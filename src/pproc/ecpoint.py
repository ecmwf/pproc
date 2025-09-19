# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
from typing import List, Optional, Tuple
import functools
import numpy as np
import pandas as pd
import signal
import logging
import numexpr
import concurrent.futures as fut

import eccodes
from meters import ResourceMeter
from earthkit.meteo.stats import iter_quantiles
from conflator import Conflator
import earthkit.data

from pproc.config.types import ECPointConfig, ECPointParamConfig
from pproc.config.io import InputsCollection
from pproc.common.param_requester import ParamRequester, IndexFunc
from pproc.common.steps import AnyStep
from pproc.common.recovery import create_recovery, Recovery
from pproc.common.parallel import (
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.io import write_grib, GribMetadata
from pproc.common.accumulation_manager import AccumulationManager
from pproc.quantile.grib import quantiles_metadata
from pproc.ecpt.predictors import compute_predictors, to_ekmetadata

logger = logging.getLogger(__name__)


class FilteredParamRequester(ParamRequester):
    def __init__(
        self,
        param: ECPointParamConfig,
        inputs: InputsCollection,
        steps: list[int],
        total: Optional[int] = None,
        src_name: Optional[str] = None,
        index_func: Optional[IndexFunc] = None,
    ):
        super().__init__(param, inputs, total, src_name, index_func)
        self.steps = steps

    def retrieve_data(
        self, step: AnyStep, **kwargs
    ) -> Tuple[List[GribMetadata], np.ndarray]:
        if step not in self.steps:
            return ([], None)
        return super().retrieve_data(step=step, **kwargs)


def grid_bc_metadata(
    template: eccodes.GRIBMessage, out_keys: dict
) -> tuple[eccodes.GRIBMessage, dict]:
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")

    grib_keys = {}
    if edition == 2:
        grib_keys.update(
            {
                "edition": 2,
                "productDefinitionTemplateNumber": 73,
                "type": "gbf",
                "inputProcessIdentifier": template.get("generatingProcessIdentifier"),
                "inputOriginatingCentre": template.get("originatingCentre"),
                "typeOfGeneratingProcess": 13,
                "typeOfPostProcessing": 206,
                "indicatorOfUnitForTimeIncrement": 1,
                "timeIncrement": 1,
            }
        )
    grib_keys.update(out_keys)
    return template, grib_keys


def weather_types_metadata(
    template: eccodes.GRIBMessage, out_keys: dict
) -> tuple[eccodes.GRIBMessage, dict]:
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")

    grib_keys = {}
    if edition == 2:
        # `typeOfOriginalFieldValues` needs to be set separately as it is a helper key for
        # the packing, which doesn't exist any more after the packingType has been set
        template = template.copy()
        template.set({"edition": 2, "productDefinitionTemplateNumber": 73, "typeOfOriginalFieldValues": 1}, check_values=False)
        grib_keys.update(
            {
                "type": "gwt",
                "packingType": "grid_ieee",
                "inputProcessIdentifier": template.get("generatingProcessIdentifier"),
                "inputOriginatingCentre": template.get("originatingCentre"),
                "typeOfGeneratingProcess": 13,
                "typeOfPostProcessing": 206,
                "indicatorOfUnitForTimeIncrement": 1,
                "timeIncrement": 1,
            }
        )
    grib_keys.update(out_keys)
    return template, grib_keys


def point_scale_metadata(
    template: eccodes.GRIBMessage, pert_number: int, total_number: int, out_keys: dict
) -> dict:
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")

    grib_keys = {}
    if edition == 2:
        grib_keys.update(
            {
                "edition": 2,
                "productDefinitionTemplateNumber": 90,
                "type": "pfc",
                "inputProcessIdentifier": template.get("generatingProcessIdentifier"),
                "inputOriginatingCentre": template.get("originatingCentre"),
                "typeOfGeneratingProcess": 13,
                "typeOfPostProcessing": 206,
                "indicatorOfUnitForTimeIncrement": 1,
                "timeIncrement": 1,
            }
        )
    grib_keys.update(out_keys)
    return quantiles_metadata(template, pert_number, total_number, grib_keys)


def compute_single_ens(
    predictant: np.ndarray,
    predictors: np.ndarray,
    thr_inf: np.ndarray,
    thr_sup: np.ndarray,
    fer: np.ndarray,
    codes_wt: np.ndarray,
    wt_batch_size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    num_fer = fer.shape[1]
    num_wt = thr_inf.shape[0]
    num_pred, num_gp = predictors.shape

    pt_bc_allwt = np.zeros((num_fer, num_gp))
    wt_allwt = np.zeros((num_gp,))
    for index in range(0, num_wt, wt_batch_size):
        end_index = min(index + wt_batch_size, num_wt)
        logger.info(f"Weather types: {index} - {end_index - 1}")
        wt_size = end_index - index

        temp_wts = numexpr.evaluate(
            "prod(where((predictors >= thr_inf) & (predictors < thr_sup), 1, 0), axis=1)",
            local_dict={
                "predictors": np.reshape(predictors, (1, num_pred, num_gp)),
                "thr_inf": np.reshape(thr_inf[index:end_index], (wt_size, num_pred, 1)),
                "thr_sup": np.reshape(thr_sup[index:end_index], (wt_size, num_pred, 1)),
            },
        )
        temp_wts = np.where(np.any(np.isnan(predictors), axis=0), np.nan, temp_wts)

        wt_allwt += np.einsum("i,i...", codes_wt[index:end_index], temp_wts)
        wt_rain = np.reshape(predictant, (1, num_gp)) * temp_wts
        cdf_wt = numexpr.evaluate(
            "sum(wt_rain * (fer + 1), axis=0)",
            local_dict={
                "fer": np.reshape(fer[index:end_index], (wt_size, num_fer, 1)),
                "wt_rain": np.reshape(wt_rain, (wt_size, 1, num_gp)),
            },
        )
        pt_bc_allwt += cdf_wt
    return pt_bc_allwt, wt_allwt


def compute_weather_types(
    predictant: np.ndarray,
    predictors: np.ndarray,
    bp_loc: str,
    fer_loc: str,
    min_predictant: Optional[float] = None,
    wt_batch_size: int = 1,
    ens_batch_size: int = 1,
) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    # Extract variables from files
    bp_file = pd.read_csv(bp_loc, header=0, delimiter=",")
    fer_file = pd.read_csv(fer_loc, header=0, delimiter=",")
    bp = bp_file.iloc[:, 1:].to_numpy()
    fer = fer_file.iloc[:, 1:].to_numpy()
    codes_wt = bp_file.iloc[:, 0].to_numpy()
    thr_inf = bp[:, 0:-1:2]
    thr_sup = bp[:, 1::2]

    if min_predictant is not None:
        predictant = np.where(predictant < min_predictant, 0, predictant)
    ens_partial = functools.partial(
        compute_single_ens,
        thr_inf=thr_inf,
        thr_sup=thr_sup,
        codes_wt=codes_wt,
        fer=fer,
        wt_batch_size=wt_batch_size,
    )

    # inizialize field for the new post-processed ensemble (CDF)
    # built from all raw ensemble members and all WTs
    pt_bc_allens_allwt = []
    # inizialize field for the bias corrected (bc) at grid-scale
    # fields for all raw ensemble members and all WTs
    grid_bc_allens_allwt = []
    # inizialize field for the wt for all raw ensemble members and all WTs
    wt_allens_allwt = []

    with fut.ProcessPoolExecutor(
        max_workers=ens_batch_size,
        initializer=signal.signal,
        initargs=(signal.SIGTERM, signal.SIG_DFL),
    ) as executor:
        for ind_em, result in enumerate(
            executor.map(ens_partial, predictant, predictors.transpose(1, 0, 2))
        ):
            logger.info(f"Ensemble member: {ind_em}")
            pt_bc_allwt, wt_allwt = result
            grid_bc_allens_allwt.append(np.mean(pt_bc_allwt, axis=0))
            pt_bc_allens_allwt.extend(list(pt_bc_allwt))
            if min_predictant is not None:
                wt_allwt = np.where(
                    (predictant[ind_em] < min_predictant)
                    & (np.invert(np.isnan(wt_allwt))),
                    99999,
                    wt_allwt,
                )
            wt_allens_allwt.append(wt_allwt)

    return (
        np.asarray(pt_bc_allens_allwt),
        np.asarray(grid_bc_allens_allwt),
        np.asarray(wt_allens_allwt),
    )


def ecpoint_iteration(
    config: ECPointConfig,
    param: ECPointParamConfig,
    recovery: Recovery,
    window_id: str,
    input_params: earthkit.data.FieldList,
    out_keys: dict,
):
    logging.info(
        f"Processing {window_id}, fields: \n {input_params.ls(namespace='mars')}"
    )
    with ResourceMeter(f"Compute predictant and predictors: {window_id}"):
        predictant = input_params.sel(param=config.predictant).values
        predictors = compute_predictors(
            config, param, out_keys["stepRange"], input_params
        )

    with ResourceMeter(f"Compute realisations: {window_id}"):
        (
            pt_bc_allens_allwt,
            grid_bc_allens_allwt,
            wt_allens_allwt,
        ) = compute_weather_types(
            predictant,
            predictors,
            config.bp_location,
            config.fer_location,
            config.min_predictant,
            config.parallelisation.wt_batch_size,
            config.parallelisation.ens_batch_size,
        )

    # Scale outputs, needed for grib 2 rainfall in metres
    if config.scale_outputs is not None:
        pt_bc_allens_allwt *= config.scale_outputs
        grid_bc_allens_allwt *= config.scale_outputs

    # Save the grid-scale outputs and weather types for each member
    out_bs = config.outputs.bs
    out_wt = config.outputs.wt
    for index, field in enumerate(input_params.sel(param=config.predictant)):
        template = field.metadata()._handle
        bs_message, metadata = grid_bc_metadata(
            template,
            {
                **out_keys,
                **out_bs.metadata,
            },
        )
        write_grib(out_bs.target, bs_message, grid_bc_allens_allwt[index], metadata)
        out_bs.target.flush()

        wt_message, metadata = weather_types_metadata(
            template, {**out_keys, **out_wt.metadata}
        )
        write_grib(out_wt.target, wt_message, wt_allens_allwt[index], metadata)
        out_wt.target.flush()

    del grid_bc_allens_allwt
    del wt_allens_allwt

    with ResourceMeter(f"Compute the percentiles: {window_id}"):
        out_perc = config.outputs.perc
        template = input_params.sel(param=config.predictant)[0].metadata()._handle
        for i, quantile in enumerate(
            iter_quantiles(
                np.asarray(pt_bc_allens_allwt), config.quantiles, method="sort"
            )
        ):
            grib_keys = {
                **out_keys,
                **out_perc.metadata,
            }
            pert_number, total_number = config.quantile_indices(i)
            metadata = point_scale_metadata(
                template, pert_number, total_number, grib_keys
            )
            write_grib(out_perc.target, template, quantile, metadata)
        out_perc.target.flush()

    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-ecpoint", model=ECPointConfig).load()
    cfg.print()
    recover = create_recovery(cfg)

    for param in cfg.parameters:
        managers = [
            AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )
        ]
        checkpointed_windows = [x["window"] for x in recover.computed(param=param.name)]
        managers[0].delete(checkpointed_windows)
        requesters = [
            FilteredParamRequester(
                param,
                cfg.inputs,
                steps=managers[-1].dims["step"],
                total=cfg.total_fields,
            )
        ]
        dims = {k: set(val) for k, val in managers[0].dims.items()}
        static_data = earthkit.data.SimpleFieldList()
        for input_param in param.dependencies.values():
            new_manager = AccumulationManager.create(
                input_param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **input_param.metadata,
                },
            )
            if len(new_manager.dims) == 0:
                # Static data, requiring no accumulation
                requester = ParamRequester(
                    input_param, cfg.inputs, total=cfg.total_fields
                )
                metadata, data = requester.retrieve_data()
                static_data += earthkit.data.FieldList.from_array(
                    data, to_ekmetadata(metadata)
                )
            else:
                for dim, vals in new_manager.dims.items():
                    min_val = min(managers[0].dims[dim])
                    max_val = max(managers[0].dims[dim])
                    dims[dim].update([x for x in vals if x > min_val and x < max_val])
                new_requester = FilteredParamRequester(
                    input_param,
                    cfg.inputs,
                    steps=new_manager.dims["step"],
                    total=cfg.total_fields,
                )
                managers.append(new_manager)
                requesters.append(new_requester)
                for k, val in new_manager.dims.items():
                    dims[k] |= set(val)
        logger.debug(f"Expected number of inputs: {param.num_inputs}")
        logger.debug(f"Dims: {dims}")
        ecpoint_partial = functools.partial(ecpoint_iteration, cfg, param, recover)
        input_sets = []
        for keys, retrieved_data in parallel_data_retrieval(
            cfg.parallelisation.n_par_read,
            {k: sorted(list(val)) for k, val in dims.items()},
            requesters,
        ):
            ids = ", ".join(f"{k}={v}" for k, v in keys.items())
            with ResourceMeter(f"{ids}: Compute accumulation"):
                for index, param_data in enumerate(retrieved_data):
                    param_metadata, ens = param_data
                    for wid, completed_window in managers[index].feed(keys, ens):
                        if index == 0:
                            logger.debug(f"Creating input set for {wid}")
                            input_sets.append(
                                {
                                    "window_id": wid,
                                    "out_keys": completed_window.grib_keys(),
                                    "input_params": static_data,
                                }
                            )
                        new_fields = earthkit.data.FieldList.from_array(
                            completed_window.values, to_ekmetadata(param_metadata)
                        )
                        for fields in new_fields.group_by("param"):
                            field_name = fields.metadata()[0]["param"]
                            for input_set in input_sets:
                                if (
                                    len(input_set["input_params"].sel(param=field_name))
                                    == 0
                                ):
                                    input_set["input_params"] += fields
                    del ens
                checked = 0
                while checked < len(input_sets):
                    num_inputs = len(
                        input_sets[checked]["input_params"]
                        .ls(keys=["param"])["param"]
                        .unique()
                    )
                    if num_inputs == param.num_inputs:
                        ecpoint_partial(**input_sets[checked])
                        del input_sets[checked]
                    elif num_inputs >= param.num_inputs:
                        raise ValueError(
                            f"Retrieved {num_inputs} inputs, expected {param.num_inputs}"
                        )
                    else:
                        checked += 1

    recover.clean_file()


if __name__ == "__main__":
    sys.exit(main())
