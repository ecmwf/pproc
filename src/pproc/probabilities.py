#!/usr/bin/env python3
import sys
import os
from datetime import datetime

import numpy as np

import pyfdb

from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability
from pproc.prob.parameter import create_parameter
from pproc.prob.window_manager import ThresholdWindowManager

MISSING_VALUE = 9999


def main(args=None):

    parser = common.default_parser("Compute instantaneous and period probabilites")
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    args = parser.parse_args()
    cfg = common.Config(args)

    date = datetime.strptime(args.date, "%Y%m%d%H")
    leg = cfg.options.get("leg")
    nensembles = cfg.options.get("number_of_ensembles", 50)
    global_input_cfg = cfg.options.get("global_input_keys", {})
    global_output_cfg = cfg.options.get("global_output_keys", {})

    fdb = pyfdb.FDB()

    for param_name, param_cfg in cfg.options["parameters"].items():
        param = create_parameter(date, global_input_cfg, param_cfg, nensembles)
        window_manager = ThresholdWindowManager(param_cfg)

        for step in window_manager.unique_steps:
            message_template, data = param.retrieve_data(fdb, step)
            assert message_template is not None
            message_template.set(global_output_cfg)

            completed_windows = window_manager.update_windows(step, data)
            for window in completed_windows:
                for threshold in window_manager.thresholds(window):
                    window_probability = ensemble_probability(
                        window.step_values, threshold
                    )

                    print(
                        f"Writing probability for input param {param_name} and output "
                        + f"param {threshold['out_paramid']} for step(s) {window.name}"
                    )
                    output_file = os.path.join(
                        cfg.options["root_dir"],
                        f"{param_name}_{threshold['out_paramid']}_{leg}_step{window.name}.grib",
                    )
                    target = common.target_factory(
                        cfg.options["target"], out_file=output_file, fdb=fdb
                    )
                    common.write_grib(
                        target,
                        construct_message(
                            message_template, window.grib_header(leg), threshold
                        ),
                        window_probability,
                    )

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
