# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Optional, Tuple, List

import eccodes
from meters import ResourceMeter

from pproc.common.param_requester import ParamConfig, ParamRequester, IndexFunc
from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import parallel_data_retrieval
from pproc.config.param import ParamConfig
from pproc.config.io import InputsCollection


def retrieve_clim(
    param: ParamConfig,
    inputs: InputsCollection,
    src: str,
    total: int = 1,
    index_func: Optional[IndexFunc] = None,
    **additional_dims,
) -> Tuple[Accumulator, eccodes.GRIBMessage]:

    accums = param.accumulations.copy()
    for dim, value in additional_dims.items():
        accums[dim] = {"operation": "aggregation", "coords": [[value]]}
    accum_manager = AccumulationManager.create(accums, param.metadata)

    requester = ParamRequester(param, inputs, total, src, index_func)
    res_accum: Optional[Accumulator] = None
    res_template: Optional[eccodes.GRIBMessage] = None
    for keys, data in parallel_data_retrieval(1, accum_manager.dims, [requester]):
        ids = ", ".join(f"{k}={v}" for k, v in keys.items())
        metadata, clim = data[0]
        with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
            completed_windows = accum_manager.feed(keys, clim)
            del clim
            for _, accum in completed_windows:
                assert (
                    res_accum is None
                ), "Multiple climatological windows are not supported"
                res_accum = accum
                res_template = metadata[0]
    assert (
        res_accum is not None and res_template is not None
    ), f"Missing climatology for {param.name}"
    return res_accum, res_template
