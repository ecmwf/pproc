import functools
from typing import Any

from meters import ResourceMeter

from pproc.common.config import Config
from pproc.common.parallel import create_executor, parallel_data_retrieval
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery
from pproc.common.accumulation_manager import AccumulationManager


def main(cfg: Config, postproc_iteration: Any):
    recover = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            accum_manager = AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )

            checkpointed_windows = [
                x["window"] for x in recover.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = ParamRequester(
                param,
                cfg.sources,
                cfg.total_fields,
            )
            postproc_partial = functools.partial(
                postproc_iteration, param, cfg, recover
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
                    executor.submit(postproc_partial, metadata, window_id, accum)
            executor.wait()

    recover.clean_file()
