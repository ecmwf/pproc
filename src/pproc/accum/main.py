import functools
from typing import Any

from meters import ResourceMeter

from pproc.common.config import Config
from pproc.common.parallel import create_executor, parallel_data_retrieval
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery
from pproc.common.window_manager import WindowManager


def main(cfg: Config, postproc_iteration: Any):
    recover = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            window_manager = WindowManager(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )

            checkpointed_windows = [
                x["window"] for x in recover.computed(param=param.name)
            ]
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param.name}")
                continue

            print(f"Recovery: param {param.name} starting from step {new_start}")

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
                window_manager.dims,
                [requester],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                metadata, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(postproc_partial, metadata, window_id, accum)
            executor.wait()

    recover.clean_file()
