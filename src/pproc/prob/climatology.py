import datetime
from typing import Any, Dict, Tuple
import numpy as np

from pproc.common import Parameter


class Climatology(Parameter):
    """
    Retrieves data for mean and standard deviation of climatology
    """

    def __init__(
        self,
        dt: datetime.datetime,
        param_id: int,
        global_input_cfg,
        param_cfg: Dict,
        overrides: Dict[str, Any] = {},
    ):
        Parameter.__init__(
            self, "clim", dt, param_id, global_input_cfg, param_cfg, 0, overrides
        )
        self.base_request["time"] = "00"
        assert "date" in param_cfg["climatology"]["clim_keys"]
        for key, value in param_cfg["climatology"]["clim_keys"].items():
            self.base_request[key] = value
        self.steps = param_cfg["climatology"].get("steps", None)

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

    def retrieve_data(
        self, step: int, **kwargs
    ) -> Tuple[Dict, Tuple[np.array, np.array]]:
        """
        Retrieves data for climatology mean and standard deviation

        :param fdb:
        :param step: model step
        :return: tuple containing climatology period dates as Dict
        and
        """
        cstep = step if not self.steps else self.steps[step]
        temp_message, ret = super().retrieve_data(
            step=cstep, join_dim="type", **kwargs
        )
        return self.grib_header(temp_message), ret
