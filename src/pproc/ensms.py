#!/usr/bin/env python3
# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.
import functools
import sys
import signal

import eccodes
import numpy as np
from conflator import Conflator
from meters import ResourceMeter

from pproc.common.io import write_grib
from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.grib_helpers import fill_template_values
from pproc.config.types import EnsmsConfig


def ensms_metadata(
    accum: Accumulator,
    out_keys: dict,
):
    grib_sets = accum.grib_keys().copy()
    grib_sets.update(out_keys)
    grib_sets = fill_template_values(
        grib_sets, {"num_fields": np.prod(accum.values.shape[:-1])}
    )
    return grib_sets


def ensms_iteration(
    config: EnsmsConfig,
    param: ParamConfig,
    window_id: str,
    accum: Accumulator,
    template_ens: eccodes.GRIBMessage,
):

    ens = accum.values
    assert ens is not None

    # Compute mean/std over all dimensions except last
    axes = tuple(range(ens.ndim - 1))
    with ResourceMeter(f"Window {window_id}: write mean output"):
        mean = np.mean(ens, axis=axes)
        out_mean = config.outputs.mean
        write_grib(
            out_mean.target,
            template_ens,
            mean,
            ensms_metadata(accum, out_mean.metadata),
        )

    with ResourceMeter(f"Window {window_id}: write std output"):
        std = np.std(ens, axis=axes)
        out_std = config.outputs.std
        write_grib(
            out_std.target, template_ens, std, ensms_metadata(accum, out_std.metadata)
        )

    out_mean.target.flush()
    out_std.target.flush()
    config.recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-ensms", model=EnsmsConfig).load()
    cfg.print()

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
                x["window"] for x in cfg.recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = ParamRequester(
                param,
                cfg.inputs,
                param.total_fields,
            )
            iteration = functools.partial(ensms_iteration, cfg, param)
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                with ResourceMeter(f"Process keys {keys}"):
                    metadata, data = retrieved_data[0]

                    completed_windows = accum_manager.feed(
                        keys,
                        data,
                    )
                    for window_id, accum in completed_windows:
                        executor.submit(iteration, window_id, accum, metadata[0])
            executor.wait()

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
