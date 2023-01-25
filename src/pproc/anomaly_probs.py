import sys
import datetime
from typing import Tuple, Dict, Iterator
import numpy as np

import pyfdb
from pproc import common
from pproc.prob.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability
from pproc.prob.parameter import Parameter

LAST_MODEL_STEP = 360
CLIM_STEP_INTERVAL = 12


class CombinedForecasts(Parameter):
    def retrieve_data(self, fdb, step: int):
        """
        Retrieves data at step for perturbed and control forecast and
        concatenates them together into one array
        """
        new_request = self.base_request.copy()
        new_request["step"] = step

        # pf
        new_request["type"] = "pf"
        message_temp, pf_data = common.fdb_read_with_template(
            fdb, new_request, self.interpolation_keys
        )

        # cf
        new_request["type"] = "cf"
        new_request.pop("number")
        _, cf_data = common.fdb_read_with_template(
            fdb, new_request, self.interpolation_keys
        )

        return message_temp, np.concatenate((pf_data, cf_data), axis=0)


class Climatology(Parameter):
    """
    Retrieves data for mean and standard deviation of climatology
    """

    def __init__(self, dt: datetime.datetime, param_id: int, cfg: Dict):
        Parameter.__init__(self, dt, param_id, cfg, 0)
        self.base_request.pop("number")
        self.base_request["date"] = self.get_climatology_date(dt.date())
        self.base_request["time"] = "00"
        self.base_request["stream"] = cfg["climatology"]["stream"]
        self.base_request["type"] = "em/es"  # Order of these is important
        self.time = dt.time()
        self.steps = cfg["steps"]

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
                return step - CLIM_STEP_INTERVAL
            return step + CLIM_STEP_INTERVAL

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
        new_request = self.base_request.copy()
        new_request["step"] = cstep
        ret = []
        for type in self.base_request["type"].split("/"):
            new_request["type"] = type
            temp_message, data = common.fdb_read_with_template(
                fdb, new_request, self.interpolation_keys
            )
            ret.append(data)
        return self.grib_header(temp_message), ret


class AnomalyWindowManager(common.WindowManager):
    def __init__(self, parameter):
        super().__init__(parameter)
        self.standardised_anomaly_windows = []
        if "std_anomaly_windows" in parameter:
            # Create windows for standard anomaly
            for window_config in parameter["std_anomaly_windows"]:
                window_operation = self.window_operation_from_config(window_config)

                for period in window_config["periods"]:
                    new_window = common.create_window(period, window_operation)
                    new_window.config_grib_header = window_config.get("grib_set", {})
                    new_window.thresholds = window_config["thresholds"]
                    self.standardised_anomaly_windows.append(new_window)

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
        np.subtract(data, clim_mean, out=data)
        new_anom_windows = []
        for window in self.windows:
            window.add_step_values(step, data)

            if window.reached_end_step(step):
                yield window
            else:
                new_anom_windows.append(window)
        self.windows = new_anom_windows

        new_std_anom_windows = []
        np.divide(data, clim_std, out=data)
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

    fdb = pyfdb.FDB()

    for param_cfg in cfg.options["parameters"]:
        param_id = param_cfg["in_paramid"]
        param = CombinedForecasts(date, param_id, param_cfg, n_ensembles)
        clim = Climatology(date, param_id, param_cfg)

        window_manager = AnomalyWindowManager(param_cfg)

        for step in window_manager.unique_steps:
            message_template, data = param.retrieve_data(fdb, step)
            clim_grib_header, clim_data = clim.retrieve_data(fdb, step)

            completed_windows = window_manager.update_windows(
                step, data, clim_data[0], clim_data[1]
            )
            for window in completed_windows:
                for threshold in window.thresholds:
                    window_probability = ensemble_probability(
                        window.step_values, threshold
                    )

                    print(
                        f"Writing probability for {param_id} output "
                        + f"param {threshold['out_paramid']} for step(s) {window.name}"
                    )
                    common.write_grib(
                        common.FDBTarget(fdb),
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
