# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools
import signal
import sys

from conflator import Conflator
from earthkit.data import FieldList, SimpleFieldList
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
from pproc.common.xr_helpers import as_multiindex
from pproc.config.types import ParamConfig, SSWConfig



def weighted_average(values, latitude, axis):
    weights = np.cos(np.deg2rad(latitude))
    return np.average(values, weights=weights, axis=axis)


def ssw_iteration(
    config: SSWConfig,
    pconfig: ParamConfig,
    dims: dict
):
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

    print(fields.ls())

    for region, lat_bounds in config.latitude_bands.items():
        with ResourceMeter(f"Compute {region} {ids}"):
            template = metadata[0]
            lats = template.get_array("latitudes")
            mask = (min(lat_bounds) <= lats) & (lats <= max(lat_bounds))
            values = weighted_average(data[...,mask], lats[mask], axis=-1)
            target.write(
                pd.Series(
                    data=values,
                    index=as_multiindex(metadata, config.ensure_dims),
                    name=template.get('shortName'))
                .to_xarray()
                .expand_dims({"region": [region]})
            )

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
