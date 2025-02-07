import argparse
import sys
from typing import List, Optional, Union

import eccodes
from meters import ResourceMeter

from pproc.accum.config import AccumConfig, AccumParamConfig
from pproc.accum.main import main as accum_main
from pproc.accum.postprocess import postprocess
from pproc.common.accumulation import Accumulator
from pproc.common.config import default_parser
from pproc.common.io import (
    Target,
    read_template,
)
from pproc.common.recovery import Recovery


def get_parser() -> argparse.ArgumentParser:
    description = "Compute accumulations"
    parser = default_parser(description=description)
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--out-accum", required=True, help="Output target")
    return parser


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
            out_keys=accum.grib_keys(),
        )
        target.flush()
    if recovery is not None:
        recovery.add_checkpoint(param.name, window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args(args)
    config = AccumConfig(args)
    accum_main(args, config, postproc_iteration)


if __name__ == "__main__":
    sys.exit(main())
