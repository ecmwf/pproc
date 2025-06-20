# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
import functools
import signal

from meters import ResourceMeter
from conflator import Conflator

from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.recovery import create_recovery
from pproc.common.param_requester import ParamRequester
from pproc.config.types import ProbConfig
from pproc.prob.parallel import prob_iteration
from pproc.prob.accumulation_manager import AnomalyAccumulationManager
from pproc.prob.climatology import Climatology


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-anomaly-probs", model=ProbConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            accum_manager = AnomalyAccumulationManager.create(
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

            requester = ParamRequester(param, cfg.inputs, cfg.total_fields, "fc")
            clim = Climatology(
                param.clim,
                cfg.inputs,
                "clim",
            )
            prob_partial = functools.partial(
                prob_iteration, param, recovery, cfg.outputs.prob
            )
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester, clim],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                metadata, ens = retrieved_data[0]
                clim_metadata, clim_data = retrieved_data[1]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = accum_manager.feed(
                        keys, ens, clim_data[0], clim_data[1]
                    )
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(
                        prob_partial,
                        metadata[0],
                        window_id,
                        accum,
                        accum_manager.thresholds(window_id),
                        clim_metadata[0],
                    )
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
