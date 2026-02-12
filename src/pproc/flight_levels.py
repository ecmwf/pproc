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
import signal
import sys
import earthkit

import eccodes
import numpy as np
from conflator import Conflator
from earthkit.meteo import vertical
from earthkit.data import FieldList, SimpleFieldList
from meters import ResourceMeter

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.io import write_grib
from pproc.common.utils import dict_product
from pproc.common.parallel import (
    create_executor,
    parallel_processing,
    sigterm_handler,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.config.types import CATConfig

def cat_iteration(
    config: CATConfig,
    param: ParamConfig,
    dims: dict,
):
    fields = SimpleFieldList()
    for src_name in ["cat", "lnsp"]:
        src_param = getattr(param, src_name, param)
        requester = ParamRequester(
            src_param,
            config.inputs,
            src_param.total_fields,
            src_name,
        ) 
        metadata, data = requester.retrieve_data(**dims)
        fields += FieldList.from_array(data, [x.to_ekmetadata() for x in metadata])

    # Interate over ensemble members
    for group in fields.group_by("type", "number"):
        cat = group.sel(paramId=260290)
        lnsp = group.sel(param="lnsp")

        # A, B params (could also be read from the GRIB)
        A, B = vertical.hybrid_level_parameters(config.n_levels, model=config.model)

        # surface pressure array
        sp = np.exp(lnsp[0].values)

        # interpolate cat to target pressure levels
        # this method requires cat levels sorted in ascending order with
        # respect to model level number!
        cat = cat.order_by(level="ascending")
        cat_pl = vertical.interpolate_hybrid_to_pressure_levels(
            cat.values,
            config.target_levels,
            A,
            B,
            sp,
            interpolation=config.interp_method,
        )

        out_levels = config.outputs.levels
        for index, values in enumerate(cat_pl):
            write_grib(
                out_levels.target,
                cat[0].metadata()._handle,
                values,
                {
                    **out_levels.metadata,
                    **param.metadata,
                    "typeOfLevel": "isobaricInPa",
                    "level": config.target_levels[index],
                },
            )

    out_levels.target.flush()
    config.recovery.add_checkpoint(param=param.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-flight-levels", model=CATConfig).load()
    cfg.print()

    plan = []
    for param in cfg.parameters:
        accum_manager = AccumulationManager.create(
            param.accumulations, 
        )
        for dims in dict_product(accum_manager.dims):
            if cfg.recovery.existing_checkpoint(param=param.name, **dims):
                print(f"Recovery: skipping dims: {param.name} {dims}")
                continue
            plan.append((param, dims))

    iteration = functools.partial(cat_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
