# (C) Copyright 2026- ECMWF and individual contributors.

# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.

"""Dew point temperature on pressure levels"""

import functools
import logging
import signal
import sys

from conflator import Conflator
from earthkit.meteo import thermo
from earthkit.data import FieldList, SimpleFieldList
from meters import ResourceMeter
import numpy as np

from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.io import write_grib
from pproc.common.parallel import parallel_processing, sigterm_handler
from pproc.common.param_requester import ParamRequester
from pproc.config.types import ParamConfig, DewpointPLConfig
from pproc.common.utils import dict_product


logger = logging.getLogger(__name__)

# GRIB parameter: dew point temperature
DEWPOINT_PARAM_ID = 3017


def dewpoint_iteration(
    config: DewpointPLConfig,
    pconfig: ParamConfig,
    dims: dict,
):
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())

    fields = SimpleFieldList()
    for src_name in config.inputs.names:
        src_param = getattr(pconfig, src_name, pconfig)
        total = pconfig.compute_totalfields(config.inputs, src_name)
        requester = ParamRequester(src_param, config.inputs, total, src_name)
        with ResourceMeter(f"Retrieve {src_name} {ids}"):
            metadata, data = requester.retrieve_data(**dims)
        fields += FieldList.from_array(data, [x.to_ekmetadata() for x in metadata])

    with ResourceMeter(f"Compute dewpoint temperature {ids}"):
        # Specific humidity [kg/kg]
        q = fields.sel(param="q")
        q_arr = q.values

        # Pressure [Pa]
        p = q.metadata("vertical.level")  # [hPa]
        p_arr = 100. * np.asarray(p)[:,None]  # align for broadcasting

        # Dewpoint temperature [K]
        dpt_arr = thermo.array.dewpoint_from_specific_humidity(q_arr, p_arr)

        for i, values in enumerate(dpt_arr):
            out_dpt = config.outputs.dpt
            write_grib(
                out_dpt.target,
                q[i].metadata()._handle,
                values.astype(np.float32),
                {
                    **out_dpt.metadata,
                    **pconfig.metadata,
                    "paramId": DEWPOINT_PARAM_ID,
                },
            )

    config.outputs.dpt.target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-dewpoint-pl", model=DewpointPLConfig).load()
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

    iteration = functools.partial(dewpoint_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
