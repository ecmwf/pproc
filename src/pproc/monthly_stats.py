import sys
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import eccodes
from conflator import Conflator
from meters import ResourceMeter

from pproc.config.types import MonthlyStatsConfig, AccumParamConfig
from pproc.accum.main import main as accum_main
from pproc.accum.postprocess import postprocess
from pproc.common.accumulation import Accumulator
from pproc.common.io import read_template
from pproc.common.recovery import Recovery
from pproc.common.stepseq import steprange_to_fcmonth


def mstat_keys(fcdate: datetime, step_range: str) -> Dict[str, Any]:
    start, end = map(int, step_range.split("-"))
    this_month = fcdate + timedelta(hours=int(start))
    return {
        "forecastMonth": steprange_to_fcmonth(fcdate, step_range),
        "verifyingMonth": f"{this_month.year:02d}{this_month.month:02d}",
        "step": end,
    }


def postproc_iteration(
    param: AccumParamConfig,
    cfg: MonthlyStatsConfig,
    recovery: Optional[Recovery],
    template: Union[str, eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):
    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)

    interval = param.accumulations["step"]["steps"][0]["interval"]
    date = datetime.strptime(template.get("dataDate:str"), "%Y%m%d")
    accum_keys = accum.grib_keys()
    steprange = accum_keys.pop("stepRange")
    out_keys = {
        "localDefinitionNumber": 16,
        **accum_keys,
        **cfg.outputs.stats.metadata,
        "stepType": "instant",
        "timeRangeIndicator": 10,
        "unitOfTimeRange": 1,
        **mstat_keys(date, steprange),
        "averagingPeriod": interval,
    }
    with ResourceMeter(f"{param.name}, step {window_id}: Post-process"):
        ens = accum.values
        assert ens is not None
        postprocess(
            ens,
            template,
            cfg.outputs.stats.target,
            vmin=param.vmin,
            vmax=param.vmax,
            out_keys=out_keys,
        )
        cfg.outputs.stats.target.flush()
    recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    cfg = Conflator(app_name="pproc-monthly-stats", model=MonthlyStatsConfig).load()
    print(cfg)
    accum_main(cfg, postproc_iteration)


if __name__ == "__main__":
    sys.exit(main())
