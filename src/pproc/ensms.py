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
from typing import Union

import eccodes
import numpy as np
from conflator import Conflator
from meters import ResourceMeter

from pproc import common
from pproc.common.accumulation import Accumulator
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
    template_ens=Union[str, eccodes.GRIBMessage],
):
    if not isinstance(template_ens, eccodes.GRIBMessage):
        template_ens = common.io.read_template(template_ens)

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
        out_mean.target.flush()

    with ResourceMeter(f"Window {window_id}: write std output"):
        std = np.std(ens, axis=axes)
        out_std = config.outputs.std
        template_std = template_ensemble(template_ens, accum, out_std.metadata)
        template_std.set_array("values", common.io.nan_to_missing(template_std, std))
        out_mean.target.write(template_std)
        out_mean.target.flush()

    recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)

    cfg = Conflator(app_name="pproc-ensms", model=EnsmsConfig).load()
    cfg.print()
    recover = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            window_manager = common.WindowManager(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )

            checkpointed_windows = recover.computed(param.name)
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param.name}")
                continue

            print(f"Recovery: param {param.name} starting from step {new_start}")

            requester = ParamRequester(
                param,
                cfg.sources,
                cfg.members,
                cfg.total_fields,
            )
            iteration = functools.partial(ensms_iteration, cfg, param, recover)
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                window_manager.dims,
                [requester],
                cfg.parallelisation.n_par_compute > 1,
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    message_template, data = retrieved_data[0]

                    completed_windows = window_manager.update_windows(
                        keys,
                        data,
                    )
                    for window_id, accum in completed_windows:
                        executor.submit(iteration, window_id, accum, message_template)
            executor.wait()

    recover.clean_file()


if __name__ == "__main__":
    main(sys.argv)
