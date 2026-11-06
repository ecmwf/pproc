# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools
import signal
import sys

from conflator import Conflator
from earthkit.data import FieldList, SimpleFieldList
from meters import ResourceMeter
import numpy as np
import xarray as xr

from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import (
    parallel_processing,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.utils import dict_product
from pproc.config.types import ParamConfig, SSWConfig


def ssw_iteration(config: SSWConfig, pconfig: ParamConfig, dims: dict):
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())
    target = config.outputs.ssw.target

    fields = SimpleFieldList()
    for src_name in config.inputs.names:
        src_param = getattr(pconfig, src_name, pconfig)
        total = pconfig.compute_totalfields(config.inputs, src_name)
        requester = ParamRequester(src_param, config.inputs, total, src_name)
        with ResourceMeter(f"Retrieve {src_name} {ids}"):
            metadata, data = requester.retrieve_data(**dims)
        fields += FieldList.from_array(data, [x.to_ekmetadata() for x in metadata])

    fields = fields.sel(param="u")

    kwargs = config.to_xarray_kwargs.copy()
    kwargs["add_earthkit_attrs"] = False
    ds = fields.to_xarray(**kwargs)
    ds = ds.drop_attrs(deep=False)

    for var in ds.data_vars:
        if ds[var].dims[-2:] != ("latitude", "longitude"):
            raise NotImplementedError(
                f"{var}: cos-lat-based area weighting only implemented for regular lat-lon grids"
            )

    for region, lat_bounds in config.latitude_bands.items():
        with ResourceMeter(f"Compute {region} {ids}"):
            lo, hi = sorted(lat_bounds)
            ds_region = ds.sel({"latitude": slice(hi, lo)})
            weights = np.cos(np.deg2rad(ds_region.coords["latitude"]))
            result = (
                ds_region.weighted(weights)
                .mean(dim=["latitude", "longitude"], keep_attrs=True)
                .expand_dims({"region": [region]}, axis=-1)
            )
            # Add coordinate with the region's boundary latitudes
            band_coord = xr.DataArray(
                [[lo, hi]],
                dims=("region", "bounds"),
                attrs={"units": ds.coords["latitude"].attrs["units"]},
            )
            result = result.assign_coords({"latitude_band": band_coord})
            # Add aggregation metadata to all data variables
            for var in result.data_vars:
                result[var].attrs.update(
                    {
                        "cell_methods": "area: mean",
                        "comment": "cos(latitude)-weighted mean over grid points within the region",
                    }
                )
            target.write(result)

    target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-ssw", model=SSWConfig).load()
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

    iteration = functools.partial(ssw_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
