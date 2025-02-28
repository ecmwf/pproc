#!/usr/bin/env python3

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

import earthkit.data
import numpy as np
import thermofeel as thermofeel
from meters import metered
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.window_manager import WindowManager
from pproc.common.parallel import create_executor
from pproc.config.types import ThermoConfig, ThermoParamConfig
from pproc.config.targets import NullTarget
from pproc.common.recovery import create_recovery, Recovery
from pproc.thermo import helpers
from pproc.thermo.indices import ComputeIndices

logger = logging.getLogger(__name__)


def load_input(source: str, config: ThermoConfig, step: int):
    src_config = getattr(config.sources, source)
    reqs = src_config.request
    src = src_config.type

    if isinstance(reqs, dict):
        reqs = [reqs]

    ret = earthkit.data.FieldList()
    for req in reqs:
        if len(req) == 0:
            return None
        req = req.copy()
        req.update(config.sources.overrides)
        req["step"] = [step]
        if req["type"] == "pf":
            if isinstance(config.members, int):
                req.setdefault("number", range(1, config.members + 1))
            else:
                req.setdefault(
                    "number", range(config.members.start, config.members.end + 1)
                )

        logger.debug(f"Retrieve: {req} from source {src_config}")
        if src == "fdb":
            ds = earthkit.data.from_source("fdb", req, stream=True, read_all=True)
        elif src == "fileset":
            loc = src_config.path.format_map(req)
            req["paramId"] = req.pop("param")
            ds = earthkit.data.from_source("file", loc).sel(req)
        elif src == "mars":
            ds = earthkit.data.from_source("mars", req)
        else:
            raise ValueError(f"Unknown source {source}")

        if len(ds) == 0:
            raise ValueError(f"No data found for request {req} from source {source}")

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
    accum_metadata: list,
    accum: Accumulator,
    recovery: Recovery,
):
    inst_fields = load_input("inst", config, step)
    fields = earthkit.data.FieldList.from_array(
        inst_fields.values, inst_fields.metadata()
    )
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

    helpers.check_field_sizes(fields)
    basetime, validtime = helpers.get_datetime(fields)
    step = helpers.get_step(fields)
    time = basetime.hour

    logger.info(
        f"Compute indices step {step}, validtime {validtime.isoformat()} - "
        + f"basetime {basetime.date().isoformat()}, time {time}"
    )
    logger.debug(f"Inputs \n {fields.ls(namespace='mars')}")
    indices = ComputeIndices(config.outputs.indices.metadata)
    params = fields.indices()["param"]

    if (not isinstance(config.outputs.intermediate.target, NullTarget)) and len(
        accum["step"].coords
    ) > 1:
        inter_target = config.outputs.intermediate
        # Windspeed - shortName ws
        ws = indices.calc_field("10si", indices.calc_ws, fields)
        helpers.write(inter_target, ws)

        # Cosine of Solar Zenith Angle - shortName uvcossza - ECMWF product
        cossza = indices.calc_field("uvcossza", indices.calc_cossza_int, fields)
        helpers.write(inter_target, cossza)

        # direct solar radiation - shortName dsrp - ECMWF product
        if "dsrp" not in params:
            dsrp = indices.calc_field("dsrp", indices.calc_dsrp, fields)
            helpers.write(inter_target, dsrp)

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

    # effective temperature 261017
    # standard effective temperature 261019

    for name in config.outputs.names:
        getattr(config.outputs, name).target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
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

    for param in cfg.parameters:
        window_manager = WindowManager(
            param.accumulations, {**cfg.outputs.default.metadata, **param.metadata}
        )
        checkpointed_windows = [
            x["window"] for x in recovery.computed(param=param.name)
        ]
        new_start = window_manager.delete_windows(checkpointed_windows)
        if new_start is None:
            logger.info(f"Recovery: skipping completed param {param.name}")
            continue

        logger.debug(f"Recovery: param {param.name} starting from step {new_start}")

        thermo_partial = functools.partial(process_step, cfg, param, recovery=recovery)
        with create_executor(cfg.parallelisation) as executor:
            for step in window_manager.dims["step"]:
                accum_data = load_input("accum", cfg, step)
                completed_windows = window_manager.update_windows(
                    {"step": step}, [] if accum_data is None else accum_data.values
                )
                for window_id, accum in completed_windows:

                    executor.submit(
                        thermo_partial,
                        step,
                        window_id,
                        [] if accum_data is None else accum_data.metadata(),
                        accum,
                    )

        executor.wait()
    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
