#!/usr/bin/env python3
import sys
from datetime import datetime

import numpy as np

import pyfdb

from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability
from pproc.prob.parameter import Parameter

MISSING_VALUE = 9999


def main(args=None):

    parser = common.default_parser("Compute instantaneous and period probabilites")
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    args = parser.parse_args()
    cfg = common.Config(args)

    date = datetime.strptime(args.date, "%Y%m%d%H")
    leg = cfg.options.get("leg")
    nensembles = cfg.options.get("number_of_ensembles", 50)

    fdb = pyfdb.FDB()

    for param_cfg in cfg.options["parameters"]:
        param_id = param_cfg["in_paramid"]
        param = Parameter(date, param_id, param_cfg, nensembles)
        second_param = None
        if "second_parameter" in param_cfg:
            second_param_cfg = param_cfg["second_parameter"]
            second_param = Parameter(
                date, second_param_cfg["paramid"], param_cfg, nensembles
            )

        window_manager = common.WindowManager(param_cfg)

        for step in window_manager.unique_steps:
            message_template, data = param.retrieve_data(fdb, step)
            assert message_template is not None

            if second_param:
                _, data2 = second_param.retrieve_data(fdb, step)
                if second_param_cfg["combine_operation"] == "norm":
                    data = np.linalg.norm([data, data2], axis=0)
                else:
                    data = getattr(np, second_param_cfg["combine_operation"])(
                        [data, data2], axis=0
                    )

            completed_windows = window_manager.update_windows(step, data)
            for window in completed_windows:
                for threshold in window.thresholds:
                    window_probability = ensemble_probability(
                        window.step_values, threshold
                    )

                    print(
                        f"Writing probability for input param {param_id} and output "
                        + f"param {threshold['out_paramid']} for step(s) {window.name}"
                    )
                    common.write_grib(
                        common.FDBTarget(fdb),
                        construct_message(
                            message_template, window.grib_header(leg), threshold
                        ),
                        window_probability,
                    )

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
