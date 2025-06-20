# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
from typing import Any
import signal

from meters import ResourceMeter

from pproc.common.config import Config
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery
from pproc.common.accumulation_manager import AccumulationManager


def main(cfg: Config, postproc_iteration: Any):
    signal.signal(signal.SIGTERM, sigterm_handler)

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
                cfg.inputs,
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
