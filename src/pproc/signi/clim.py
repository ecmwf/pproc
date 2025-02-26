from typing import Optional, Tuple, List

import eccodes
from meters import ResourceMeter

from pproc.common.param_requester import ParamConfig, ParamRequester, IndexFunc
from pproc.common.accumulation import Accumulator
from pproc.common.window_manager import WindowManager
from pproc.common.parallel import parallel_data_retrieval
from pproc.config.param import ParamConfig
from pproc.config.io import SourceCollection


def retrieve_clim(
    param: ParamConfig,
    sources: SourceCollection,
    src: str,
    members: int = 1,
    total: int = 1,
    index_func: Optional[IndexFunc] = None,
    **additional_dims,
) -> Tuple[Accumulator, eccodes.GRIBMessage]:

    accums = param.accumulations.copy()
    for dim, value in additional_dims.items():
        accums[dim] = {"operation": "aggregation", "coords": [[value]]}
    window_manager = WindowManager(accums, param.metadata)

    requester = ParamRequester(param, sources, members, total, src, index_func)
    res_accum: Optional[Accumulator] = None
    res_template: Optional[eccodes.GRIBMessage] = None
    for keys, data in parallel_data_retrieval(1, window_manager.dims, [requester]):
        ids = ", ".join(f"{k}={v}" for k, v in keys.items())
        template, clim = data[0]
        with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
            completed_windows = window_manager.update_windows(keys, clim)
            del clim
            for _, accum in completed_windows:
                assert (
                    res_accum is None
                ), "Multiple climatological windows are not supported"
                res_accum = accum
                res_template = template
    assert (
        res_accum is not None and res_template is not None
    ), f"Missing climatology for {param.name}"
    return res_accum, res_template
