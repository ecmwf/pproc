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
import logging

import numpy as np
from conflator import Conflator
from earthkit.meteo import vertical
from earthkit.data import FieldList, SimpleFieldList
from meters import ResourceMeter

from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.io import write_grib
from pproc.common.utils import dict_product
from pproc.common.parallel import (
    parallel_processing,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.config.types import FlightLevelsParamConfig, FlightLevelsConfig
from pproc.flightlevel.mapping import FLIGHT_TO_PRESSURE_LEVEL

logger = logging.getLogger(__name__)


def flight_level_iteration(
    config: FlightLevelsConfig,
    pconfig: FlightLevelsParamConfig,
    dims: dict,
):
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())
    fields = SimpleFieldList()
    for src_name in config.inputs.names:
        src_param = getattr(pconfig, src_name, pconfig)
        requester = ParamRequester(
            src_param,
            config.inputs,
            src_param.total_fields,
            src_name,
        )
        with ResourceMeter(f"Retrieve {src_name} {ids}"):
            metadata, data = requester.retrieve_data(**dims)
        fields += FieldList.from_array(data, [x.to_ekmetadata() for x in metadata])

    with ResourceMeter(f"Compute flight levels {ids}"):
        lnsp = fields.sel(param="lnsp").order_by(number="ascending")
        # A, B params (could also be read from the GRIB)
        A, B = vertical.hybrid_level_parameters(config.n_levels, model=config.model)
        # surface pressure array
        sp = np.exp(lnsp.values)
        logger.debug(f"Surface pressure: {sp.shape}")
        input_levels = None
        pressure_levels = None

        for param in fields.group_by("param"):
            if param.metadata("param")[0] == "lnsp":
                continue

            logger.debug(f"Inputs:\n {param.ls()}")
            param_levels = list(set(param.metadata("level")))
            param_levels.sort()
            if input_levels is None:
                input_levels = param_levels
                pressure_levels = vertical.pressure_on_hybrid_levels(
                    sp, A, B, levels=input_levels, output="full"
                )
                logger.debug(f"Pressure levels: {pressure_levels.shape}")

            assert (
                input_levels == param_levels
            ), "Input levels must be the same for all parameters"
            # interpolate cat to target pressure levels
            # this method requires cat levels sorted in ascending order with
            # respect to model level number!
            param = param.order_by(level="ascending", number="ascending")
            out_pl = vertical.interpolate_monotonic(
                param.values.reshape(pressure_levels.shape),
                pressure_levels,
                [FLIGHT_TO_PRESSURE_LEVEL[lvl] for lvl in config.target_flight_levels],
                interpolation=config.interp_method,
            )

            out_levels = config.outputs.levels
            templates = param.sel(level=input_levels[0])
            for index, lvl in enumerate(config.target_flight_levels):
                for mem_index, values in enumerate(out_pl[index]):
                    write_grib(
                        out_levels.target,
                        templates[mem_index].metadata()._handle,
                        values,
                        {
                            **out_levels.metadata,
                            **pconfig.metadata,
                            "typeOfLevel": "flightLevel",
                            "level": lvl,
                        },
                    )

    out_levels.target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-flight-levels", model=FlightLevelsConfig).load()
    cfg.initialise()
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

    iteration = functools.partial(flight_level_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
