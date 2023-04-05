import datetime
from typing import Dict, Tuple
import numpy as np

from pproc.prob.parameter import Parameter
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
