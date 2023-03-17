#!/usr/bin/env python3
import sys
import os
from datetime import datetime

import pyfdb

from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability
from pproc.prob.parameter import create_parameter
from pproc.prob.window_manager import ThresholdWindowManager


def write_grib(cfg, fdb, filename, template, data):
    output_file = os.path.join(cfg.options["root_dir"], filename)
    target = common.target_factory(cfg.options["target"], out_file=output_file, fdb=fdb)
    common.write_grib(target, template, data)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    parser = common.default_parser("Compute instantaneous and period probabilites")
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    parser.add_argument(
        "--write_ensemble",
        action="store_true",
        default=False,
        help="write ensemble members to fdb/file",
    )
    args = parser.parse_args()
    cfg = common.Config(args)

    date = datetime.strptime(args.date, "%Y%m%d%H")
    nensembles = int(cfg.options.get("number_of_ensembles", 50))
    global_input_cfg = cfg.options.get("global_input_keys", {})
    global_output_cfg = cfg.options.get("global_output_keys", {})

    fdb = pyfdb.FDB()
    recovery = common.Recovery(cfg.options["root_dir"], args.config, date, args.recover)
    last_checkpoint = recovery.last_checkpoint()
    last_checkpoint_step = -1

    for param_name, param_cfg in sorted(cfg.options["parameters"].items()):
        param = create_parameter(date, global_input_cfg, param_cfg, nensembles)
        window_manager = ThresholdWindowManager(param_cfg, global_output_cfg)
        if last_checkpoint and recovery.existing_checkpoint(param_name, window_manager.unique_steps[0]):
            if param_name not in last_checkpoint:
                print(f"Recovery: skipping completed param {param_name}")
                continue
            last_checkpoint_step = int(recovery.checkpoint_identifiers(last_checkpoint)[1])
            window_manager.update_from_checkpoint(last_checkpoint_step)
            print(f"Recovery: param {param_name} looping from step {window_manager.unique_steps[0]}")
        
        for step in window_manager.unique_steps:
            with common.ResourceMeter(f"Parameter {param_name}, step {step}"):
                message_template, data = param.retrieve_data(fdb, step)

                completed_windows = window_manager.update_windows(step, data)
                for window in completed_windows:
                    if args.write_ensemble:
                        for index in range(len(window.step_values)):
                            data_type, number = param.type_and_number(index)
                            print(
                                f"Writing window values for param {param_name} and output "
                                + f"type {data_type}, number {number} for step(s) {window.name}"
                            )
                            template = construct_message(
                                message_template, window.grib_header()
                            )
                            template.set({"type": data_type, "number": number})
                            write_grib(
                                cfg,
                                fdb,
                                f"{param_name}_type{data_type}_number{number}_step{window.name}.grib",
                                template,
                                window.step_values[index],
                            )
                    for threshold in window_manager.thresholds(window):
                        window_probability = ensemble_probability(
                            window.step_values, threshold
                        )

                        print(
                            f"Writing probability for input param {param_name} and output "
                            + f"param {threshold['out_paramid']} for step(s) {window.name}"
                        )
                        write_grib(
                            cfg,
                            fdb,
                            f"{param_name}_{threshold['out_paramid']}_step{window.name}.grib",
                            construct_message(
                                message_template, window.grib_header(), threshold
                            ),
                            window_probability,
                        )
                
            if step > last_checkpoint_step:
                recovery.add_checkpoint(param_name, step)
        last_checkpoint_step = -1

    fdb.flush()
    recovery.clean()


if __name__ == "__main__":
    main(sys.argv)
