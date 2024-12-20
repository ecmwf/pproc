import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import eccodes
from earthkit.time.calendar import MonthInYear
from earthkit.time.sequence import MonthlySequence
from meters import ResourceMeter

from pproc.accum.config import AccumConfig, AccumParamConfig
from pproc.accum.main import main as accum_main
from pproc.accum.postprocess import postprocess
from pproc.common.accumulation import Accumulator
from pproc.common.config import default_parser
from pproc.common.io import Target, read_template
from pproc.common.recovery import Recovery


def mstat_keys(fcdate: datetime, step_range: str) -> Dict[str, Any]:
    start, end = map(int, step_range.split("-"))
    seq = MonthlySequence(1)
    first_month = seq.next(fcdate, False)
    this_month = fcdate + timedelta(hours=int(start))

    assert MonthInYear(this_month.year, this_month.month).length() * 24 == (end - start)

    mindex = this_month.month - first_month.month + 1
    return {
        "forecastMonth": mindex,
        "verifyingMonth": f"{this_month.year:02d}{this_month.month:02d}",
    }


def postproc_iteration(
    param: AccumParamConfig,
    target: Target,
    recovery: Optional[Recovery],
    template: Union[str, eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):
    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)

    interval = param._accumulations["step"]["steps"][0]["interval"]
    date = datetime.strptime(template.get("dataDate:str"), "%Y%m%d")
    out_keys = {
        "localDefinitionNumber": 16,
        "averagingPeriod": interval,
        **mstat_keys(date, accum.grib_keys()["stepRange"]),
        **accum.grib_keys(),
    }
    with ResourceMeter(f"{param.name}, step {window_id}: Post-process"):
        ens = accum.values
        assert ens is not None
        postprocess(
            ens,
            template,
            target,
            vmin=param.vmin,
            vmax=param.vmax,
            out_paramid=param.out_paramid,
            out_keys=out_keys,
        )
        target.flush()
    if recovery is not None:
        recovery.add_checkpoint(param.name, window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = default_parser(description="Compute monthly statistics")
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--out-stats", dest="out_accum", required=True, help="Output target")
    args = parser.parse_args(args)
    config = AccumConfig(args)
    accum_main(args, config, postproc_iteration)


if __name__ == "__main__":
    sys.exit(main())
