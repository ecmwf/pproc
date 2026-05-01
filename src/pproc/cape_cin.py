#!/usr/bin/env python3
# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import logging
import signal
import sys

import numpy as np
from conflator import Conflator
from earthkit.data import FieldList, SimpleFieldList
from meters import ResourceMeter

from earthkit.meteo import constants
from earthkit.meteo.thermo import specific_humidity_from_dewpoint

from pproc.cape.compute import compute_mucape_mucin
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.io import write_grib
from pproc.common.parallel import parallel_processing, sigterm_handler
from pproc.common.param_requester import ParamRequester
from pproc.common.utils import dict_product
from pproc.config.param import ParamConfig
from pproc.config.types import CapeConfig

logger = logging.getLogger(__name__)

# GRIB paramId for MUCAPE and MUCIN
MUCAPE_PARAM_ID = 228235
MUCIN_PARAM_ID = 228236


def cape_iteration(
    config: CapeConfig,
    pconfig: ParamConfig,
    dims: dict,
) -> None:
    """Process one (step, …) slice: retrieve data, compute MUCAPE/MUCIN, write output."""
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())

    fields = SimpleFieldList()
    for src_name in config.inputs.names:
        total = pconfig.compute_totalfields(config.inputs, src_name)
        requester = ParamRequester(
            pconfig,
            config.inputs,
            total,
            src_name,
        )
        # Surface geopotential is only available at step 0
        retrieve_dims = {**dims, "step": 0} if src_name == "zsfc" else dims
        with ResourceMeter(f"Retrieve {src_name} {ids}"):
            metadata, data = requester.retrieve_data(**retrieve_dims)
        fields += FieldList.from_array(data, [x.to_ekmetadata() for x in metadata])

    with ResourceMeter(f"Compute CAPE/CIN {ids}"):
        p_sfc = fields.sel(param="sp")
        t_sfc = fields.sel(param="2t")
        td_sfc = fields.sel(param="2d")
        z_sfc = fields.sel(param="z", levtype="sfc")

        t_pl = fields.sel(param="t").order_by(level="ascending", number="ascending")
        q_pl = fields.sel(param="q").order_by(level="ascending", number="ascending")
        z_pl = fields.sel(param="z", levtype="pl").order_by(
            level="ascending", number="ascending"
        )

        n_members = len(p_sfc)
        levels_sorted = sorted(config.pressure_levels)

        sfc_template = t_sfc[0].metadata()

        for member in range(n_members):
            # Surface arrays
            p_sfc_vals = p_sfc[member].values  # Pa
            t_sfc_vals = t_sfc[member].values  # K
            td_sfc_vals = td_sfc[member].values  # K
            z_sfc_vals = z_sfc[member].values  # m²/s²

            zh_sfc_vals = z_sfc_vals / constants.g  # m
            q_sfc_vals = specific_humidity_from_dewpoint(td_sfc_vals, p_sfc_vals)


            # PL arrays
            n_points = p_sfc_vals.shape[0]
            n_levels = len(levels_sorted)

            t_arr = np.empty((n_levels, n_points), dtype=np.float64)
            q_arr = np.empty((n_levels, n_points), dtype=np.float64)
            zh_arr = np.empty((n_levels, n_points), dtype=np.float64)

            for lev_idx, level in enumerate(levels_sorted):
                t_field = t_pl.sel(level=level)
                q_field = q_pl.sel(level=level)
                z_field = z_pl.sel(level=level)

                t_arr[lev_idx] = t_field[member].values
                q_arr[lev_idx] = q_field[member].values
                zh_arr[lev_idx] = z_field[member].values / constants.g  # geopotential → height

            # Compute MUCAPE / MUCIN
            mucape, mucin = compute_mucape_mucin(
                t=t_arr,
                q=q_arr,
                zh=zh_arr,
                p_levels_hpa=levels_sorted,
                t_sfc=t_sfc_vals,
                q_sfc=q_sfc_vals,
                zh_sfc=zh_sfc_vals,
                p_sfc=p_sfc_vals,
            )

            # Write GRIB output
            template = sfc_template._handle

            out_mucape = config.outputs.mucape
            write_grib(
                out_mucape.target,
                template,
                mucape.astype(np.float32),
                {
                    **out_mucape.metadata,
                    **pconfig.metadata,
                    "paramId": MUCAPE_PARAM_ID,
                    **dims,
                },
            )

            out_mucin = config.outputs.mucin
            write_grib(
                out_mucin.target,
                template,
                mucin.astype(np.float32),
                {
                    **out_mucin.metadata,
                    **pconfig.metadata,
                    "paramId": MUCIN_PARAM_ID,
                    **dims,
                },
            )

    config.outputs.mucape.target.flush()
    config.outputs.mucin.target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-cape", model=CapeConfig).load()
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

    iteration = functools.partial(cape_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
