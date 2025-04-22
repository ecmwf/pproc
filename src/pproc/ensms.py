#!/usr/bin/env python3
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

import eccodes
import numpy as np
from conflator import Conflator
from meters import ResourceMeter

from pproc import common
from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import create_executor, parallel_data_retrieval
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import create_recovery, BaseRecovery
from pproc.config.types import EnsmsConfig


def template_ensemble(
    template: eccodes.GRIBMessage,
    accum: Accumulator,
    out_keys: dict,
):
    template_ens = template.copy()

    grib_sets = accum.grib_keys().copy()
    grib_sets.update(out_keys)
    template_ens.set(grib_sets)
    return template_ens


def ensms_iteration(
    config: EnsmsConfig,
    param: ParamConfig,
    recovery: BaseRecovery,
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
        template_mean = template_ensemble(template_ens, accum, out_mean.metadata)
        template_mean.set_array("values", common.io.nan_to_missing(template_mean, mean))
        out_mean.target.write(template_mean)

    with ResourceMeter(f"Window {window_id}: write std output"):
        std = np.std(ens, axis=axes)
        out_std = config.outputs.std
        template_std = template_ensemble(template_ens, accum, out_std.metadata)
        template_std.set_array("values", common.io.nan_to_missing(template_std, std))
        out_std.target.write(template_std)

    out_mean.target.flush()
    out_std.target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)

    cfg = Conflator(app_name="pproc-ensms", model=EnsmsConfig).load()
    cfg.print()
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
            iteration = functools.partial(ensms_iteration, cfg, param, recover)
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    metadata, data = retrieved_data[0]

                    completed_windows = accum_manager.feed(
                        keys,
                        data,
                    )
                    for window_id, accum in completed_windows:
                        executor.submit(iteration, window_id, accum, metadata[0])
            executor.wait()

    recover.clean_file()


if __name__ == "__main__":
    sys.exit(main())
