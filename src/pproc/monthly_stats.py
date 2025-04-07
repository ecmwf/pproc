import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
import numpy as np

import eccodes
from conflator import Conflator
from meters import ResourceMeter

from pproc.config.types import MonthlyStatsConfig, AccumParamConfig
from pproc.accum.main import main as accum_main
from pproc.accum.postprocess import postprocess
from pproc.common.accumulation import Accumulator
from pproc.common.recovery import Recovery
from pproc.common.stepseq import steprange_to_fcmonth


def mstat_keys(template, out_keys: dict, interval: int):
    steprange = out_keys.pop("stepRange")
    start, end = map(int, steprange.split("-"))
    if out_keys.get("edition", template.get("edition")) == 1:
        date = datetime.strptime(template.get("dataDate:str"), "%Y%m%d")
        this_month = date + timedelta(hours=int(start))
        return {
            "localDefinitionNumber": 16,
            **out_keys,
            "stepType": "instant",
            "timeRangeIndicator": 10,
            "unitOfTimeRange": 1,
            "forecastMonth": steprange_to_fcmonth(date, steprange),
            "verifyingMonth": f"{this_month.year:02d}{this_month.month:02d}",
            "step": end,
            "averagingPeriod": interval,
        }
    out_keys.pop("unitOfTimeRange", None)
    return {
        "localDefinitionNumber": 16,
        **out_keys,
        "stepType": "instant",
        "productDefinitionTemplateNumber": 11,
        "indicatorOfUnitForTimeIncrement": 1,
        "timeIncrement": interval,
        "step": end,
        "typeOfGeneratingProcess": template.get("typeOfGeneratingProcess"), 
        "typeOfProcessedData": template.get("type"),
    }


def postproc_iteration(
    param: AccumParamConfig,
    cfg: MonthlyStatsConfig,
    recovery: Optional[Recovery],
    metadata: list[eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):
    intervals = np.diff(accum["step"].coords)
    assert np.all(intervals == intervals[0]), "Step intervals must be equal"
    out_keys = {
        **accum.grib_keys(),
        **cfg.outputs.stats.metadata,
    }
    with ResourceMeter(f"{param.name}, step {window_id}: Post-process"):
        ens = accum.values
        assert ens is not None
        postprocess(
            ens,
            metadata,
            cfg.outputs.stats.target,
            vmin=param.vmin,
            vmax=param.vmax,
            out_keys=mstat_keys(metadata[0], out_keys, intervals[0]),
        )
        cfg.outputs.stats.target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cfg = Conflator(app_name="pproc-monthly-stats", model=MonthlyStatsConfig).load()
    cfg.print()
    accum_main(cfg, postproc_iteration)


if __name__ == "__main__":
    sys.exit(main())
