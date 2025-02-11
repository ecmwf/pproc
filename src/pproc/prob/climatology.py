from typing import Dict, Tuple
import numpy as np

import eccodes

from pproc.common.param_requester import ParamRequester, ParamConfig


class Climatology(ParamRequester):
    """
    Retrieves data for mean and standard deviation of climatology
    """

    def __init__(
        self,
        param: ParamConfig,
        sources: dict,
        loc: str,
    ):
        super().__init__(param, sources, loc, 1, 2, self._index_func)
        self.steps = self.param._in_keys.pop("step", None)

    @classmethod
    def _index_func(cls, msg: eccodes.GRIBMessage) -> int:
        if msg.get("type") == "em":
            return 0
        if msg.get("type") == "es":
            return 1
        raise ValueError(f"Unexpected message type {msg.get('type')}")

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
        self, fdb, step: int, **kwargs
    ) -> Tuple[Dict, Tuple[np.array, np.array]]:
        """
        Retrieves data for climatology mean and standard deviation

        :param fdb:
        :param step: model step
        :return: tuple containing climatology period dates as Dict
        and
        """
        cstep = step if not self.steps else self.steps[step]
        temp_message, ret = super().retrieve_data(fdb, step=cstep, **kwargs)
        return self.grib_header(temp_message), ret
