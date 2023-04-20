import sys
import os
import datetime

from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability
from pproc.prob.window_manager import AnomalyWindowManager
from pproc.prob.climatology import Climatology
from pproc.prob.multiprocess import parallel_data_retrieval


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    parser = common.default_parser(
        "Compute instantaneous and period probabilites for anomalies"
    )
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    parser.add_argument(
        "-p",
        "--processes",
        default=2,
        type=int,
        help=f"number of processes for reading files, default: 2",
    )
    args = parser.parse_args()
    cfg = common.Config(args)

    date = datetime.datetime.strptime(args.date, "%Y%m%d%H")
    n_ensembles = int(cfg.options.get("number_of_ensembles", 50))
    global_input_cfg = cfg.options.get("global_input_keys", {})
    global_output_cfg = cfg.options.get("global_output_keys", {})

    fdb = common.io.fdb(create=True)
    recovery = common.Recovery(cfg.options["root_dir"], args.config, date, args.recover)
    last_checkpoint = recovery.last_checkpoint()

    for param_name, param_cfg in sorted(cfg.options["parameters"].items()):
        param = common.create_parameter(date, global_input_cfg, param_cfg, n_ensembles)
        clim = Climatology(date, param_cfg["in_paramid"], global_input_cfg, param_cfg)

        window_manager = AnomalyWindowManager(param_cfg, global_output_cfg)
        if last_checkpoint and recovery.existing_checkpoint(
            param_name, window_manager.unique_steps[0]
        ):
            if param_name not in last_checkpoint:
                print(f"Recovery: skipping completed param {param_name}")
                continue
            last_checkpoint_step = int(
                recovery.checkpoint_identifiers(last_checkpoint)[1]
            )
            window_manager.update_from_checkpoint(last_checkpoint_step)
            print(
                f"Recovery: param {param_name} looping from step {window_manager.unique_steps[0]}"
            )

        for step, retrieved_data in parallel_data_retrieval(
            args.processes, window_manager.unique_steps, [param, clim], 0
        ):

            with common.ResourceMeter(f"Process step {step}"):
                message_template, data = retrieved_data[0]
                clim_grib_header, clim_data = retrieved_data[1]

                completed_windows = window_manager.update_windows(
                    step, data, clim_data[0], clim_data[1]
                )
                for window in completed_windows:
                    for threshold in window_manager.thresholds(window):
                        window_probability = ensemble_probability(
                            window.step_values, threshold
                        )

                        print(
                            f"Writing probability for {param_name} output "
                            + f"param {threshold['out_paramid']} for step(s) {window.name}"
                        )
                        output_file = os.path.join(
                            cfg.options["root_dir"],
                            f"{param_name}_{threshold['out_paramid']}_step{window.name}.grib",
                        )
                        target = common.target_factory(
                            cfg.options["target"], out_file=output_file, fdb=fdb
                        )
                        common.write_grib(
                            target,
                            construct_message(
                                message_template,
                                window.grib_header(),
                                threshold,
                                clim_grib_header,
                            ),
                            window_probability,
                        )
                fdb.flush()
                recovery.add_checkpoint(param_name, step)

    recovery.clean_file()


if __name__ == "__main__":
    main(sys.argv)
