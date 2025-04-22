#!/usr/bin/env python3
import sys
import functools

from meters import ResourceMeter
from conflator import Conflator

from pproc.common.parallel import create_executor, parallel_data_retrieval
from pproc.common.recovery import create_recovery
from pproc.common.param_requester import ParamRequester
from pproc.config.types import ProbConfig
from pproc.prob.parallel import prob_iteration
from pproc.prob.accumulation_manager import ThresholdAccumulationManager


def main():
    sys.stdout.reconfigure(line_buffering=True)

    cfg = Conflator(app_name="pproc-probabilities", model=ProbConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            accum_manager = ThresholdAccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )
            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = ParamRequester(param, cfg.sources, cfg.total_fields, "fc")
            prob_partial = functools.partial(
                prob_iteration, param, recovery, cfg.outputs.prob.target
            )
            for keys, data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                metadata, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = accum_manager.feed(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(
                        prob_partial,
                        metadata[0],
                        window_id,
                        accum,
                        accum_manager.thresholds(window_id),
                    )
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
