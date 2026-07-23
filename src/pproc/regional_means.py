# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import signal
import sys

from conflator import Conflator
from earthkit.data import Field, ArrayField
from meters import ResourceMeter
import numpy as np
import pandas as pd

from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import (
    create_executor,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.utils import dict_product
from pproc.config.types import RegionalMeansParamConfig, RegionalMeansConfig


def crop(field: Field, bbox):
    """Latitude and field values cropped to a lat-lon bounding box"""
    lat, lon, values = field.data(["lat", "lon", "value"], flatten=True)
    n, w, s, e = bbox
    mask = (n >= lat) & (lat >= s) & (e >= lon) & (lon >= w)
    return lat[mask], values[mask]


def area_weighted_mean(field: Field, area) -> float:
    """Area-weighted mean of the field in a lat-lon bounding box
    
    cos(lat) weighting assumes a lat-lon grid with regularly spaced latitudes
    """
    lat, values = crop(field, area)
    area_weights = np.cos(np.deg2rad(lat))
    return np.average(values, weights=area_weights)


def regional_mean_iteration(
    config: RegionalMeansConfig,
    pconfig: RegionalMeansParamConfig,
    dims: dict
):
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())
    rows = []
    for src_name in config.inputs.names:
        src_param = getattr(pconfig, src_name, pconfig)
        total = pconfig.compute_totalfields(config.inputs, src_name)
        requester = ParamRequester(src_param, config.inputs, total, src_name)
        with ResourceMeter(f"Retrieve {src_name} {ids}"):
            metadata, data = requester.retrieve_data(**dims)
        with ResourceMeter("Compute means"):
            for md, arr in zip(metadata, data):
                field = ArrayField(arr, md.to_ekmetadata())
                rows.append([
                    *field.metadata(pconfig.out_coords),
                    *(area_weighted_mean(field, bbox) for bbox in pconfig.areas.values())
                ])
    return pd.DataFrame.from_records(rows, columns=[
        *pconfig.out_coords,
        *(f"{pconfig.name}_{name}" for name in pconfig.areas.keys())
    ])


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-regional-means", model=RegionalMeansConfig).load()
    cfg.initialise()
    cfg.print()

    with create_executor(cfg.parallelisation) as executor:
        futures = []
        for param in cfg.parameters:
            accum_manager = AccumulationManager.create(
                param.accumulations,
            )
            for dims in dict_product(accum_manager.dims):
                if cfg.recovery.existing_checkpoint(param=param.name, **dims):
                    print(f"Recovery: skipping dims: {param.name} {dims}")
                    continue  # TODO should create a future that loads already computed data

                futures.append(
                    executor.submit(regional_mean_iteration, cfg, param, dims)
                )

        # TODO needs more complexity when there are multiple parameters
        df = pd.concat([future.result() for future in concurrent.futures.as_completed(futures)])

    # Parse dates and datetimes (TODO earthkit-data 1.0 should remove the need for this)
    if "forecast_reference_time" in df.columns:
        df["forecast_reference_time"] = pd.to_datetime(df["forecast_reference_time"])
    if "step" in df.columns:
        df["step"] = pd.to_timedelta(df["step"], unit="h")

    with ResourceMeter(f"Write to {cfg.outputs.default.target.path}"):
        ds = df.set_index(param.out_coords).sort_index().to_xarray()
        # TODO: covjson output
        ds.to_netcdf(cfg.outputs.default.target.path)

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
