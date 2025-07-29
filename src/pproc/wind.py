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

from pproc import common
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.recovery import Recovery, create_recovery
from pproc.common.parallel import parallel_processing, sigterm_handler
from pproc.common.utils import dict_product
from pproc.common.param_requester import ParamRequester
from pproc.config.param import ParamConfig
from pproc.config.types import WindConfig
from pproc.config.targets import NullTarget


def wind_template(
    template: eccodes.GRIBMessage, step: int, **out_keys
) -> eccodes.GRIBMessage:
    new_template = template.copy()
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

    new_template.set(grib_sets)
    return new_template


def wind_iteration(
    config: WindConfig,
    recovery: Recovery,
    param: ParamConfig,
    dims: dict,
):
    requester = ParamRequester(
        param,
        config.inputs,
        src_name="fc",
        total=config.total_fields,
    )
    metadata, ens = requester.retrieve_data(**dims)
    template = metadata[0]
    assert (
        ens.shape[0] == config.total_fields
    ), f"Expected {config.total_fields}, got {ens.shape[0]}"
    with ResourceMeter(f"Param {param.name}, {dims}"):
        if not isinstance(config.outputs.ws.target, NullTarget):
            for number in range(ens.shape[0]):
                marstype = (
                    "pf"
                    if number > 0 and template.get("type") in ["cf", "fc"]
                    else template.get("type")
                )
                template = wind_template(
                    template,
                    **dims,
                    number=number,
                    type=marstype,
                    **config.outputs.ws.metadata,
                    **param.metadata,
                )
                common.io.write_grib(config.outputs.ws.target, template, ens[number])

        template_mean = wind_template(
            template,
            **dims,
            **config.outputs.mean.metadata,
            **param.metadata,
        )
        common.io.write_grib(
            config.outputs.mean.target, template_mean, np.mean(ens, axis=0)
        )

        template_std = wind_template(
            template,
            **dims,
            **config.outputs.std.metadata,
            **param.metadata,
        )
        common.io.write_grib(
            config.outputs.std.target, template_std, np.std(ens, axis=0)
        )

    for name in config.outputs.names:
        getattr(config.outputs, name).target.flush()
    recovery.add_checkpoint(param=param.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-wind", model=WindConfig).load()
    cfg.print()
    recover = create_recovery(cfg)

    plan = []
    for param in cfg.parameters:
        accum_manager = AccumulationManager.create(param.accumulations)
        for dims in dict_product(accum_manager.dims):
            if recover.existing_checkpoint(param=param.name, **dims):
                print(f"Recovery: skipping dims: {param.name} {dims}")
                continue
            plan.append((param, dims))

    iteration = functools.partial(wind_iteration, cfg, recover)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    recover.clean_file()


if __name__ == "__main__":
    main()
