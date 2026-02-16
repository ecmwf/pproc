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
from pproc.common.param_requester import ParamRequester
from pproc.config.types import FlightLevelsParamConfig, FlightLevelsConfig

logger = logging.getLogger(__name__)


PRESSURE_TO_FLIGHT_LEVEL = {
    84310: 50,
    81200: 60,
    78190: 70,
    75260: 80,
    72430: 90,
    69680: 100,
    67020: 110,
    64440: 120,
    61940: 130,
    59520: 140,
    57180: 150,
    54920: 160,
    52720: 170,
    50600: 180,
    48550: 190,
    46560: 200,
    44650: 210,
    42790: 220,
    41000: 230,
    39270: 240,
    37600: 250,
    35990: 260,
    34430: 270,
    32930: 280,
    31490: 290,
    30090: 300,
    28740: 310,
    27450: 320,
    26200: 330,
    25000: 340,
    23840: 350,
    22730: 360,
    21660: 370,
    20650: 380,
    19680: 390,
    18750: 400,
    17870: 410,
    17040: 420,
    16240: 430,
    15470: 440,
    14750: 450,
}


def flight_level_iteration(
    config: FlightLevelsConfig,
    pconfig: FlightLevelsParamConfig,
    dims: dict,
):
    fields = SimpleFieldList()
    for src_name in config.inputs.names:
        src_param = getattr(pconfig, src_name, pconfig)
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
        lnsp = group.sel(param="lnsp")
        # A, B params (could also be read from the GRIB)
        A, B = vertical.hybrid_level_parameters(config.n_levels, model=config.model)
        # surface pressure array
        sp = np.exp(lnsp[0].values)
        input_levels = None
        pressure_levels = None

        for param in group.group_by("param"):
            if param.metadata("param")[0] == "lnsp":
                continue

            logger.debug(f"Inputs:\n {param.ls()}")
            if input_levels is None:
                input_levels = param.metadata("level")
                input_levels.sort()
                logger.debug(f"Subset levels: {input_levels}")
                pressure_levels = vertical.pressure_on_hybrid_levels(
                    A, B, sp, levels=input_levels, output="full"
                )

            assert input_levels == sorted(
                param.metadata("level")
            ), "Input levels must be the same for all parameters"
            # interpolate cat to target pressure levels
            # this method requires cat levels sorted in ascending order with
            # respect to model level number!
            param = param.order_by(level="ascending")
            out_pl = vertical.interpolate_monotonic(
                param.values,
                pressure_levels,
                config.target_levels,
                interpolation=config.interp_method,
            )

            out_levels = config.outputs.levels
            for index, values in enumerate(out_pl):
                write_grib(
                    out_levels.target,
                    param[0].metadata()._handle,
                    values,
                    {
                        **out_levels.metadata,
                        **pconfig.metadata,
                        "typeOfLevel": "flightLevel",
                        "level": PRESSURE_TO_FLIGHT_LEVEL[config.target_levels[index]],
                    },
                )

    out_levels.target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-flight-levels", model=FlightLevelsConfig).load()
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
