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


import sys
import functools
import numpy as np
import signal

import eccodes
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.recovery import create_recovery, Recovery
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.config.types import ExtremeParamConfig, ExtremeConfig
from pproc.extremes.grib import extreme_template
from pproc.signi.clim import retrieve_clim


def read_clim(
    config: ExtremeConfig,
    param: ExtremeParamConfig,
    accum: Accumulator,
    n_clim: int = 101,
) -> tuple[np.ndarray, eccodes.GRIBMessage]:
    grib_keys = accum.grib_keys()
    clim_step = grib_keys.get("stepRange", grib_keys.get("step", None))
    clim_request = param.sources["clim"]["request"]
    clim_request["quantile"] = ["{}:100".format(i) for i in range(n_clim)]
    step = clim_request.get("step", {}).get(clim_step, clim_step)
    clim_accum, clim_template = retrieve_clim(
        param,
        config.sources,
        "clim",
        n_clim,
        index_func=lambda x: int(x.get("quantile").split(":")[0]),
        step=step,
    )
    return clim_accum.values, clim_template


def compute_indices(
    cfg: ExtremeConfig,
    param: ExtremeParamConfig,
    recovery: Recovery,
    message_template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"Window {window_id}, computing indices"):
        clim, template_clim = read_clim(cfg, param.clim, accum)
        print(f"Climatology array: {clim.shape}")

        template_extreme = extreme_template(
            accum,
            message_template,
            template_clim,
            allow_grib1_to_grib2=param.allow_grib1_to_grib2,
        )

        ens = accum.values
        assert ens is not None

        if param.vmin is not None or param.vmax is not None:
            np.clip(ens, param.vmin, param.vmax, out=ens)

        for name, index in param.indices.items():
            target = getattr(cfg.outputs, name).target
            index.compute(clim, ens, target, message_template, template_extreme)
            target.flush()

        recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-extreme", model=ExtremeConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            accum_manager = AccumulationManager.create(
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

            indices_partial = functools.partial(compute_indices, cfg, param, recovery)
            requester = ParamRequester(param, cfg.sources, cfg.total_fields, "fc")
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                step = keys["step"]
                with ResourceMeter(f"Process step {step}"):
                    metadata, data = retrieved_data[0]
                    assert data.ndim == 2

                    completed_windows = accum_manager.feed(keys, data)
                    for window_id, accum in completed_windows:
                        executor.submit(indices_partial, metadata[0], window_id, accum)

            executor.wait()
    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
