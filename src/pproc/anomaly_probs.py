import sys
import os
import datetime
from typing import Tuple, Dict, Iterator
import numpy as np

import pyfdb
from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability
from pproc.prob.parameter import create_parameter, Parameter
from pproc.prob.window_manager import ThresholdWindowManager
from pproc.prob.model_constants import LAST_MODEL_STEP, CLIM_INTERVAL


class Climatology(Parameter):
    """
    Retrieves data for mean and standard deviation of climatology
    """

    def __init__(
        self, dt: datetime.datetime, param_id: int, global_input_cfg, param_cfg: Dict
    ):
        Parameter.__init__(self, dt, param_id, global_input_cfg, param_cfg, 0)
        self.base_request.pop("number")
        self.base_request["date"] = self.get_climatology_date(dt.date())
        self.base_request["time"] = "00"
        self.base_request["stream"] = param_cfg["climatology"]["stream"]
        self.base_request["type"] = "em/es"  # Order of these is important
        self.time = dt.time()
        self.steps = param_cfg["steps"]

    def clim_step(self, step: int):
        """
        Nearest step with climatology data to step,
        taking into account diurnal variation in climatology
        which requires climatology step time to be same
        as step
        """
        if self.time == datetime.time(0):
            return step
        if self.time == datetime.time(12):
            if step == LAST_MODEL_STEP:
                return step - CLIM_INTERVAL
            return step + CLIM_INTERVAL

    @classmethod
    def get_climatology_date(cls, date: datetime.date) -> str:
        """
        Assumes climatology run on Monday and Thursday and retrieves most recent
        date climatology is available
        """
        dow = date.weekday()
        if dow >= 0 and dow < 3:
            return (date - datetime.timedelta(days=dow)).strftime("%Y%m%d")
        return (date - datetime.timedelta(days=(dow - 3))).strftime("%Y%m%d")

    @classmethod
    def grib_header(cls, grib_msg):
        """
        Get climatology period from grib message
        """
        return {
            "climateDateFrom": grib_msg.get("climateDateFrom"),
            "climateDateTo": grib_msg.get("climateDateTo"),
            "referenceDate": grib_msg.get("referenceDate"),
        }

    def retrieve_data(self, fdb, step: int) -> Tuple[Dict, Tuple[np.array, np.array]]:
        """
        Retrieves data for climatology mean and standard deviation,
        taking into account possible shift required between data and
        nearest climatology step

        :param fdb:
        :param step: model step
        :return: tuple containing climatology period dates as Dict
        and
        """
        cstep = self.clim_step(step)
        temp_message, ret = super().retrieve_data(fdb, cstep)
        return self.grib_header(temp_message), ret


class AnomalyWindowManager(ThresholdWindowManager):
    def __init__(self, parameter, global_config):
        self.standardised_anomaly_windows = []
        ThresholdWindowManager.__init__(self, parameter, global_config)

    def create_windows(self, parameter, global_config):
        super().create_windows(parameter, global_config)
        if "std_anomaly_windows" in parameter:
            # Create windows for standard anomaly
            for window_config in parameter["std_anomaly_windows"]:
                window_operations = self.window_operation_from_config(window_config)

                for operation, thresholds in window_operations.items():
                    for period in window_config["periods"]:
                        new_window = common.create_window(period, operation)
                        new_window.config_grib_header = global_config.copy()
                        new_window.config_grib_header.update(
                            window_config.get("grib_set", {})
                        )
                        self.standardised_anomaly_windows.append(new_window)
                        self.window_thresholds[new_window] = thresholds

    def update_windows(
        self, step, data: np.array, clim_mean: np.array, clim_std: np.array
    ) -> Iterator[common.Window]:
        """
        Updates all windows that include step with either the anomaly with clim_mean
        or standardised anomaly including clim_std. Function modifies input data array.

        :param step: new step
        :param data: data for step
        :param clim_mean: mean from climatology
        :param clim_std: standard deviation from climatology
        :return: generator for completed windows
        """
        data = data - clim_mean
        new_anom_windows = []
        for window in self.windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield window
            else:
                new_anom_windows.append(window)
        self.windows = new_anom_windows

        new_std_anom_windows = []
        data = data / clim_std
        for window in self.standardised_anomaly_windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield window
            else:
                new_std_anom_windows.append(window)
        self.standardised_anomaly_windows = new_std_anom_windows


def main(args=None):

    parser = common.default_parser(
        "Compute instantaneous and period probabilites for anomalies"
    )
    parser.add_argument("-d", "--date", required=True, help="Forecast date")
    args = parser.parse_args()
    cfg = common.Config(args)

    date = datetime.datetime.strptime(args.date, "%Y%m%d%H")
    n_ensembles = cfg.options.get("number_of_ensembles", 50)
    leg = cfg.options.get("leg")
    global_input_cfg = cfg.options.get("global_input_keys", {})
    global_output_cfg = cfg.options.get("global_output_keys", {})

    fdb = pyfdb.FDB()

    for param_name, param_cfg in cfg.options["parameters"].items():
        param = create_parameter(date, global_input_cfg, param_cfg, n_ensembles)
        clim = Climatology(date, param_cfg["in_paramid"], global_input_cfg, param_cfg)

        window_manager = AnomalyWindowManager(param_cfg, global_output_cfg)

        for step in window_manager.unique_steps:
            message_template, data = param.retrieve_data(fdb, step)
            clim_grib_header, clim_data = clim.retrieve_data(fdb, step)

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
                        f"{param_name}_{threshold['out_paramid']}_{leg}_step{window.name}.grib",
                    )
                    target = common.target_factory(
                        cfg.options["target"], out_file=output_file, fdb=fdb
                    )
                    common.write_grib(
                        target,
                        construct_message(
                            message_template,
                            window.grib_header(leg),
                            threshold,
                            clim_grib_header,
                        ),
                        window_probability,
                    )

    fdb.flush()


if __name__ == "__main__":
    main(sys.argv)
