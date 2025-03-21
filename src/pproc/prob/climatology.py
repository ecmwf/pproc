from typing import Dict, Tuple, Optional
import numpy as np

import eccodes

from pproc.common.param_requester import ParamRequester, ParamConfig
from pproc.config.io import SourceCollection


class Climatology(ParamRequester):
    """
    Retrieves data for mean and standard deviation of climatology
    """

    def __init__(
        self,
        param: ParamConfig,
        sources: SourceCollection,
        src_name: Optional[str] = None,
    ):
        super().__init__(
            param,
            sources,
            total=2,
            src_name=src_name,
            index_func=self._index_func,
        )
        clim_request = param.sources["clim"]["request"]
        self.steps = clim_request.get("step", None)

    @classmethod
    def _index_func(cls, msg: eccodes.GRIBMessage) -> int:
        if msg.get("type") == "em":
            return 0
        if msg.get("type") == "es":
            return 1
        raise ValueError(f"Unexpected message type {msg.get('type')}")

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
        temp_message, ret = super().retrieve_data(step=cstep, **kwargs)
        clim_grib = {
            "climateDateFrom": temp_message.get("climateDateFrom"),
            "climateDateTo": temp_message.get("climateDateTo"),
            "referenceDate": temp_message.get("referenceDate"),
        }
        return clim_grib, ret
