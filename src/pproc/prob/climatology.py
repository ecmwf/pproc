# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Dict, Tuple, Optional
import numpy as np

import eccodes

from pproc.common.param_requester import ParamRequester, ParamConfig
from pproc.config.io import InputsCollection


class NullRequester:
    def retrieve_data(
        self, step: int, **kwargs
    ) -> Tuple[list[Dict], Tuple[np.array, np.array]]:
        return [{}], []


class Climatology(ParamRequester):
    """
    Retrieves data for mean and standard deviation of climatology
    """

    def __init__(
        self,
        param: ParamConfig,
        inputs: InputsCollection,
        src_name: Optional[str] = None,
    ):
        super().__init__(
            param,
            inputs,
            total=2,
            src_name=src_name,
            index_func=self._index_func,
        )
        clim_request = param.inputs["clim"]["request"]
        if isinstance(clim_request, list):
            clim_request = clim_request[0]
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
    ) -> Tuple[list[Dict], Tuple[np.array, np.array]]:
        """
        Retrieves data for climatology mean and standard deviation

        :param fdb:
        :param step: model step
        :return: tuple containing climatology period dates as Dict
        and
        """
        cstep = step if not self.steps else self.steps[step]
        metadata, ret = super().retrieve_data(step=cstep, **kwargs)
        clim_grib = {
            "climateDateFrom": metadata[0].get("climateDateFrom"),
            "climateDateTo": metadata[0].get("climateDateTo"),
            "referenceDate": metadata[0].get("referenceDate"),
        }
        return [clim_grib], ret


def create_clim(
    param: Optional[ParamConfig],
    inputs: InputsCollection,
    src_name: Optional[str] = None,
):
    if param is None:
        return NullRequester()
    return Climatology(param, inputs, src_name)
