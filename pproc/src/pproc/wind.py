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
import numpy as np
import signal

import eccodes
from meters import ResourceMeter
from conflator import Conflator
from ppcore.utils.dicts import dict_product

from pproc import common
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import parallel_processing, sigterm_handler
from pproc.common.param_requester import ParamRequester
from pproc.config.param import ParamConfig
from pproc.config.types import WindConfig
from pproc.config.targets import NullTarget


def wind_metadata(step: int, **out_keys) -> dict:
    grib_sets = {
        "bitsPerValue": 24,
        "step": step,
        **out_keys,
    }
    if step == 0:
        grib_sets["timeRangeIndicator"] = 1
    elif step > 255:
        grib_sets["timeRangeIndicator"] = 10
    else:
        grib_sets["timeRangeIndicator"] = 0

    return grib_sets


def wind_iteration(
    config: WindConfig,
    param: ParamConfig,
    dims: dict,
):
    requester = ParamRequester(
        param,
        config.inputs,
        src_name="fc",
        total=param.total_fields,
    )
    metadata, ens = requester.retrieve_data(**dims)
    template = metadata[0]
    with ResourceMeter(f"Param {param.name}, {dims}"):
        if not isinstance(config.outputs.ws.target, NullTarget):
            for number in range(ens.shape[0]):
                marstype = (
                    "pf"
                    if number > 0 and template.get("type") in ["cf", "fc"]
                    else template.get("type")
                )
                metadata = wind_metadata(
                    **dims,
                    number=number,
                    type=marstype,
                    **config.outputs.ws.metadata,
                    **param.metadata,
                )
                common.io.write_grib(
                    config.outputs.ws.target, template, ens[number], metadata
                )

        mean_keys = wind_metadata(
            **dims,
            **config.outputs.mean.metadata,
            **param.metadata,
        )
        common.io.write_grib(
            config.outputs.mean.target, template, np.mean(ens, axis=0), mean_keys
        )

        std_keys = wind_metadata(
            **dims,
            **config.outputs.std.metadata,
            **param.metadata,
        )
        common.io.write_grib(
            config.outputs.std.target, template, np.std(ens, axis=0), std_keys
        )

    for name in config.outputs.names:
        getattr(config.outputs, name).target.flush()
    config.recovery.add_checkpoint(param=param.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-wind", model=WindConfig).load()
    cfg.initialise()
    cfg.print()

    plan = []
    for param in cfg.parameters:
        accum_manager = AccumulationManager.create(param.accumulations)
        for dims in dict_product(accum_manager.dims):
            if cfg.recovery.existing_checkpoint(param=param.name, **dims):
                print(f"Recovery: skipping dims: {param.name} {dims}")
                continue
            plan.append((param, dims))

    iteration = functools.partial(wind_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    main()
