#!/usr/bin/env python3
# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Note: This script is intended only as example usage of thermofeel library.
#       It is designed to be used with ECMWF forecast data.

import logging
import sys
import functools
import signal

import earthkit.data
from earthkit.data.readers.grib.metadata import GribFieldMetadata
import numpy as np
import thermofeel as thermofeel
from meters import metered
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.parallel import create_executor, sigterm_handler
from pproc.config.types import ThermoConfig, ThermoParamConfig
from pproc.common.recovery import create_recovery, Recovery
from pproc.thermo import helpers
from pproc.thermo.indices import ComputeIndices

logger = logging.getLogger(__name__)


def load_input(config, param: ThermoParamConfig, source: str, step: int):
    sources = param.in_sources(config.sources, source, step=step)

    ret = earthkit.data.from_source("empty")
    for src in sources:
        if src.type == "null":
            continue

        for req in src.request:
            req = req.copy()
            req.update(config.sources.overrides)
            req["step"] = step

            logger.debug(f"Retrieve step {step}: source {src}")
            ds = None
            if src.type == "fdb":
                ds = earthkit.data.from_source("fdb", req, stream=True, read_all=True)
            elif src.type == "fileset":
                loc = src.path.format_map(req)
                req["paramId"] = req.pop("param")
                ds = earthkit.data.from_source("file", loc).sel(req).order_by("paramId")
            elif src.type == "mars":
                ds = earthkit.data.from_source("mars", req)
            else:
                raise ValueError(f"Unknown source {src}")

            if len(ds) == 0:
                raise ValueError(f"No data found from source {src} for step {step}")

            ret += earthkit.data.FieldList.from_array(ds.values, ds.metadata())
    return ret


def is_target_param(out_params: list[str], valid_names: set[str]) -> bool:
    return bool(set(out_params) & valid_names)


@metered("Process step", out=logger.info)
def process_step(
    config: ThermoConfig,
    param: ThermoParamConfig,
    step: int,
    window_id: str,
    accum: Accumulator,
    accum_metadata: list[GribFieldMetadata],
    recovery: Recovery,
):
    fields = load_input(config, param, "inst", step)
    if len(accum["step"].coords) > 1:
        logger.debug(f"Write out accum fields to target {config.outputs.accum}")
        # Set step range for de-accumulated fields
        step_range = "-".join(map(str, accum["step"].coords))
        accum_fields = earthkit.data.FieldList.from_array(
            accum.values,
            [x.override(stepType="diff", stepRange=step_range) for x in accum_metadata],
        )
        helpers.write(config.outputs.accum, accum_fields)
        fields += accum_fields

    assert len(fields) != 0, f"No fields retrieved for param {param}."
    helpers.check_field_sizes(fields)
    basetime, validtime = helpers.get_datetime(fields)
    time = basetime.hour

    logger.info(
        f"Compute indices step {step}, validtime {validtime.isoformat()} - "
        + f"basetime {basetime.date().isoformat()}, time {time}"
    )
    logger.debug(f"Inputs \n {fields.ls(namespace='mars')}")
    indices = ComputeIndices(config.outputs.indices.metadata)
    params = fields.indices()["param"]

    indices_target = config.outputs.indices
    # Mean Radiant Temperature - shortName mrt - ECMWF product
    if is_target_param(param.out_params, {"mrt", 261002}):
        mrt = indices.calc_field("mrt", indices.calc_mrt, fields)
        helpers.write(indices_target, mrt)

    # Univeral Thermal Climate Index - shortName utci - ECMWF product
    if is_target_param(param.out_params, {"utci", 261001}):
        utci = indices.calc_field(
            "utci",
            indices.calc_utci,
            fields,
            print_misses=config.utci_misses,
            validate=config.validateutci,
        )
        helpers.write(indices_target, utci)

    # Heat Index (adjusted) - shortName heatx - ECMWF product
    if is_target_param(param.out_params, {"heatx", 260004}):
        heatx = indices.calc_field("heatx", indices.calc_heatx, fields)
        helpers.write(indices_target, heatx)

    # Wind Chill factor - shortName wcf - ECMWF product
    if is_target_param(param.out_params, {"wcf", 260005}):
        wcf = indices.calc_field("wcf", indices.calc_wcf, fields)
        helpers.write(indices_target, wcf)

    # Apparent Temperature - shortName aptmp - ECMWF product
    if is_target_param(param.out_params, {"aptmp", 260255}):
        aptmp = indices.calc_field("aptmp", indices.calc_aptmp, fields)
        helpers.write(indices_target, aptmp)

    # Relative humidity percent at 2m - shortName 2r - ECMWF product
    if is_target_param(param.out_params, {"2r", 260242}):
        rhp = indices.calc_field("2r", indices.calc_rhp, fields)
        helpers.write(indices_target, rhp)

    # Humidex - shortName hmdx
    if is_target_param(param.out_params, {"hmdx", 261016}):
        hmdx = indices.calc_field("hmdx", indices.calc_hmdx, fields)
        helpers.write(indices_target, hmdx)

    # Normal Effective Temperature - shortName nefft
    if is_target_param(param.out_params, {"nefft", 261018}):
        nefft = indices.calc_field("nefft", indices.calc_nefft, fields)
        helpers.write(indices_target, nefft)

    # Globe Temperature - shortName gt
    if is_target_param(param.out_params, {"gt", 261015}):
        gt = indices.calc_field("gt", indices.calc_gt, fields)
        helpers.write(indices_target, gt)

    # Wet-bulb potential temperature - shortName wbpt
    if is_target_param(param.out_params, {"wbpt", 261022}):
        wbpt = indices.calc_field("wbpt", indices.calc_wbpt, fields)
        helpers.write(indices_target, wbpt)

    # Wet Bulb Globe Temperature - shortName wbgt
    if is_target_param(param.out_params, {"wbgt", 261014}):  #
        wbgt = indices.calc_field("wbgt", indices.calc_wbgt, fields)
        helpers.write(indices_target, wbgt)

    # Write out intermediate fields
    for field in ["10si", "cossza", "dsrp"]:
        sel = indices.results.sel(param=field)
        if len(sel) != 0:
            helpers.write(config.outputs.intermediate, sel)

    for name in config.outputs.names:
        getattr(config.outputs, name).target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-thermal-indices", model=ThermoConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    logger.info(f"thermofeel: {thermofeel.__version__}")
    logger.info(f"earthkit.data: {earthkit.data.__version__}")
    logger.info(f"Numpy: {np.version.version}")
    logger.info(f"Python: {sys.version}")
    logger.debug(
        f"Parallel processes: {cfg.parallelisation.n_par_compute}, queue size: {cfg.parallelisation.queue_size}"
    )

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            accum_manager = AccumulationManager.create(
                param.accumulations, {**cfg.outputs.default.metadata, **param.metadata}
            )
            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            thermo_partial = functools.partial(
                process_step, cfg, param, recovery=recovery
            )
            for step in accum_manager.dims["step"]:
                accum_fields = load_input(cfg, param, "accum", step)
                completed_windows = accum_manager.feed(
                    {"step": step},
                    np.empty((1,)) if len(accum_fields) == 0 else accum_fields.values,
                )
                for window_id, accum in completed_windows:
                    executor.submit(
                        thermo_partial,
                        step,
                        window_id,
                        accum,
                        accum_fields.metadata(),
                    )

            executor.wait()
    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
