# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools
import signal
import sys

from conflator import Conflator
from earthkit.data import Field, ArrayField
from meters import ResourceMeter
import numpy as np
import pandas as pd

from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import (
    parallel_processing,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.utils import dict_product
from pproc.config.types import ParamConfig, RegionalMeansConfig


def crop(field: Field, bbox):
    """Latitude and field values cropped to a lat-lon bounding box."""
    lat, lon, values = field.data(["lat", "lon", "value"], flatten=True)
    n, w, s, e = bbox
    mask = (n >= lat) & (lat >= s) & (e >= lon) & (lon >= w)
    return lat[mask], values[mask]


def area_weighted_mean(field: Field, area) -> float:
    """Area-weighted mean of the field in a lat-lon bounding box.

    cos(lat) weighting assumes a lat-lon grid with regularly spaced latitudes.
    """
    lat, values = crop(field, area)
    area_weights = np.cos(np.deg2rad(lat))
    return np.average(values, weights=area_weights)


def get_metadata(field, key):
    value = field.metadata(key)
    if key == "forecast_reference_time":
        return pd.to_datetime(value)
    if key == "step":
        return pd.to_timedelta(value, unit="h")
    return value


def regional_mean_iteration(
    config: RegionalMeansConfig,
    pconfig: ParamConfig,
    dims: dict
):
    ids = ", ".join(f"{k}={v}" for k, v in dims.items())
    target = config.outputs.timeseries.target
    for src_name in config.inputs.names:
        src_param = getattr(pconfig, src_name, pconfig)
        total = pconfig.compute_totalfields(config.inputs, src_name)
        requester = ParamRequester(src_param, config.inputs, total, src_name)
        with ResourceMeter(f"Retrieve {src_name} {ids}"):
            metadata, data = requester.retrieve_data(**dims)
        with ResourceMeter("Compute means"):
            # One row output for each field input, one value/column per region
            for md, arr in zip(metadata, data):
                field = ArrayField(arr, md.to_ekmetadata())
                target.write(
                    index=[get_metadata(field, key) for key in target.columns.index],
                    values=[
                        area_weighted_mean(field, config.bbox[region])
                        for region in target.columns.values
                    ],
                )
    target.flush()
    config.recovery.add_checkpoint(param=pconfig.name, **dims)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-regional-means", model=RegionalMeansConfig).load()
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

    iteration = functools.partial(regional_mean_iteration, cfg)
    parallel_processing(
        iteration,
        plan,
        cfg.parallelisation,
    )

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
