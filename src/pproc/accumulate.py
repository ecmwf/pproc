import sys

import eccodes
from conflator import Conflator
from meters import ResourceMeter

from pproc.config.types import AccumConfig, AccumParamConfig
from pproc.accum.main import main as accum_main
from pproc.accum.postprocess import postprocess
from pproc.common.accumulation import Accumulator
from pproc.common.recovery import Recovery


def postproc_iteration(
    param: AccumParamConfig,
    cfg: AccumConfig,
    recovery: Recovery,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"{param.name}, step {window_id}: Post-process"):
        ens = accum.values
        assert ens is not None
        postprocess(
            ens,
            template,
            cfg.outputs.accum.target,
            vmin=param.vmin,
            vmax=param.vmax,
            out_accum_key=param.out_accum_key,
            out_accum_values=param.out_accum_values,
            out_keys={
                **accum.grib_keys(),
                **cfg.outputs.accum.metadata,
            },
        )
        cfg.outputs.accum.target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cfg = Conflator(app_name="pproc-accumulate", model=AccumConfig).load()
    cfg.print()
    accum_main(cfg, postproc_iteration)


if __name__ == "__main__":
    sys.exit(main())
