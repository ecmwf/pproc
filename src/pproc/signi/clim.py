from typing import Optional, Tuple

import eccodes
from meters import ResourceMeter

from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.accumulation import Accumulator
from pproc.common.window_manager import WindowManager
from pproc.common.parallel import parallel_data_retrieval


def retrieve_clim(
    param: ParamConfig,
    sources: dict,
    loc: str,
    steprange: str,
    members: int = 1,
    total: Optional[int] = None,
) -> Tuple[Accumulator, eccodes.GRIBMessage]:

    win_cfg = param.window_config([])
    win_cfg.pop("windows", None)
    win_cfg.pop("steps", None)
    accums = win_cfg.setdefault("accumulations", {})
    accums["step"] = {"operation": "aggregation", "coords": [[steprange]]}
    window_manager = WindowManager(win_cfg, param.out_keys())

    requester = ParamRequester(param, sources, loc, members, total)
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
