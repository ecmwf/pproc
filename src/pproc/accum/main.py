import functools
from typing import Any

from meters import ResourceMeter

from pproc.common import parallel
from pproc.common.config import Config
from pproc.common.io import target_from_location
from pproc.common.parallel import (
    QueueingExecutor,
    SynchronousExecutor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import Recovery
from pproc.common.window_manager import WindowManager


def main(args, config: Config, postproc_iteration: Any):
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()
    target = target_from_location(args.out_accum, overrides=config.override_output)
    if config.n_par_compute > 1:
        target.enable_parallel(parallel)
    if recovery is not None and args.recover:
        target.enable_recovery()

    executor = (
        SynchronousExecutor()
        if config.n_par_compute == 1
        else QueueingExecutor(config.n_par_compute, config.window_queue_size)
    )

    with executor:
        for param in config.params:
            window_manager = WindowManager(
                param.window_config(config.windows, config.steps),
                param.out_keys(config.out_keys),
            )
            if last_checkpoint:
                if param.name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param.name}")
                    continue
                checkpointed_windows = [
                    recovery.checkpoint_identifiers(x)[1]
                    for x in recovery.checkpoints
                    if param.name in x
                ]
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                config.sources,
                args.in_ens,
                config.num_members,
                config.total_fields,
            )
            postproc_partial = functools.partial(
                postproc_iteration, param, target, recovery
            )
            for keys, data in parallel_data_retrieval(
                config.n_par_read,
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

    if recovery is not None:
        recovery.clean_file()
