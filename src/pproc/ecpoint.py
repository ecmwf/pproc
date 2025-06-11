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
import yaml
import numexpr
import concurrent.futures as fut

import eccodes
from meters import ResourceMeter
from earthkit.meteo.stats import iter_quantiles
from conflator import Conflator
import earthkit.data
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from earthkit.data.readers.grib.codes import GribCodesHandle

from pproc.config.param import ParamConfig
from pproc.config.types import ECPointConfig, ECPointParamConfig
from pproc.config.io import SourceCollection
from pproc.common.param_requester import ParamRequester, IndexFunc
from pproc.common.steps import AnyStep
from pproc.common.recovery import create_recovery, Recovery
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.io import nan_to_missing, GribMetadata
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.grib_helpers import construct_message
from pproc.quantile.grib import quantiles_template

logger = logging.getLogger(__name__)


class FilteredParamRequester(ParamRequester):
    def __init__(
        self,
        param: ECPointParamConfig,
        sources: SourceCollection,
        steps: list[int],
        total: Optional[int] = None,
        src_name: Optional[str] = None,
        index_func: Optional[IndexFunc] = None,
    ):
        super().__init__(param, sources, total, src_name, index_func)
        self.steps = steps

    def retrieve_data(
        self, step: AnyStep, **kwargs
    ) -> Tuple[List[GribMetadata], np.ndarray]:
        if step not in self.steps:
            return ([], None)
        return super().retrieve_data(step, **kwargs)


def ratio(var_num, var_den):
    den_zero = var_den == 0
    ratio_mapped = var_num / np.where(den_zero, -9999, var_den)
    return np.where(den_zero, 0, ratio_mapped)


def grid_bc_template(
    template: eccodes.GRIBMessage, out_keys: dict
) -> eccodes.GRIBMessage:
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")

    grib_keys = out_keys.copy()
    if edition == 2:
        grib_keys.update(
            {
                "productDefinitionTemplateNumber": 73,
                "type": "gbf",
                "inputProcessIdentifier": template.get("generatingProcessIdentifier"),
                "typeOfGeneratingProcess": 13,
                "typeOfPostProcessing": 206,
                "indicatorOfUnitForTimeIncrement": 1,
                "timeIncrement": 1,
            }
        )
    return construct_message(template, grib_keys)


def weather_types_template(
    template: eccodes.GRIBMessage, out_keys: dict
) -> eccodes.GRIBMessage:
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")

    grib_keys = out_keys.copy()
    if edition == 2:
        # `typeOfOriginalFieldValues` needs to be set separately as it is a helper key for
        # the packing, which doesn't exist any more after the packingType has been set
        template = template.copy()
        template.set({"edition": 2, "typeOfOriginalFieldValues": 1}, check_values=False)
        grib_keys.update(
            {
                "productDefinitionTemplateNumber": 73,
                "type": "gwt",
                "packingType": "grid_ieee",
                "inputProcessIdentifier": template.get("generatingProcessIdentifier"),
                "typeOfGeneratingProcess": 13,
                "typeOfPostProcessing": 206,
                "indicatorOfUnitForTimeIncrement": 1,
                "timeIncrement": 1,
            }
        )
    return construct_message(template, grib_keys)


def point_scale_template(
    template: eccodes.GRIBMessage, pert_number: int, total_number: int, out_keys: dict
) -> eccodes.GRIBMessage:
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")

    grib_keys = out_keys.copy()
    if edition == 2:
        grib_keys.update(
            {
                "productDefinitionTemplateNumber": 90,
                "type": "pfc",
                "inputProcessIdentifier": template.get("generatingProcessIdentifier"),
                "typeOfPostProcessing": 206,
                "indicatorOfUnitForTimeIncrement": 1,
                "timeIncrement": 1,
            }
        )
    return quantiles_template(template, pert_number, total_number, grib_keys)


def compute_single_ens(
    predictand: np.ndarray,
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
                "thr_inf": np.reshape(
                    thr_inf[index:end_index], (wt_size, num_pred, 1)
                ),
                "thr_sup": np.reshape(
                    thr_sup[index:end_index], (wt_size, num_pred, 1)
                ),
            },
        )
        temp_wts = np.where(np.any(np.isnan(predictors), axis=0), np.nan, temp_wts)
        
        wt_allwt += np.einsum("i,i...", codes_wt[index:end_index], temp_wts)
        wt_rain = numexpr.evaluate(
            "pred * wts",
            local_dict={
                "pred": np.reshape(predictand, (1, num_gp)),
                "wts": temp_wts,
            },
        )
        cdf_wt = numexpr.evaluate(
            "sum(wt_rain * (fer + 1), axis=0)",
            local_dict={
                "fer": np.reshape(fer[index:end_index], (wt_size, num_fer, 1)),
                "wt_rain": np.reshape(wt_rain, (wt_size, 1, num_gp)),
            },
        )
        numexpr.evaluate(
            "pt + cdf", local_dict={"pt": pt_bc_allwt, "cdf": cdf_wt}, out=pt_bc_allwt
        )
    return pt_bc_allwt, wt_allwt


def compute_weather_types(
    tp: np.ndarray,
    cpr: np.ndarray,
    ws700: np.ndarray,
    maxmucape: np.ndarray,
    sr: np.ndarray,
    bp_loc: str,
    fer_loc: str,
    min_predictand: float = 0.04,
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

    predictand = np.where(tp < min_predictand, 0, tp)
    predictors = np.asarray([cpr, tp, ws700, maxmucape, sr])
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
            executor.map(
                ens_partial,
                *zip(
                    *[
                        (predictand[index], predictors[:, index])
                        for index in range(len(tp))
                    ]
                ),
            )
        ):
            logger.info(f"Ensemble member: {ind_em}")
            pt_bc_allwt, wt_allwt = result
            grid_bc_allens_allwt.append(np.mean(pt_bc_allwt, axis=0))
            pt_bc_allens_allwt.extend(list(pt_bc_allwt))
            wt_allens_allwt.append(
                np.where((predictand[ind_em] < min_predictand) & (np.invert(np.isnan(wt_allwt))), 99999, wt_allwt)
            )

    return pt_bc_allens_allwt, grid_bc_allens_allwt, wt_allens_allwt


def to_ekmetadata(metadata: list[GribMetadata]) -> list[StandAloneGribMetadata]:
    return [
        StandAloneGribMetadata(
            GribCodesHandle(eccodes.codes_clone(x._handle), None, None)
        )
        for x in metadata
    ]


def retrieve_sr24(
    config: ECPointConfig, param: ParamConfig, step: int
) -> earthkit.data.FieldList:
    requester = ParamRequester(param, config.sources, config.total_fields, "fc")
    end_step = max(step, 24)
    _, start_data = requester.retrieve_data(end_step - 24)
    metadata, end_data = requester.retrieve_data(end_step)
    return earthkit.data.FieldList.from_array(
        end_data - start_data, to_ekmetadata(metadata)
    )


def ecpoint_iteration(
    config: ECPointConfig,
    param: ECPointParamConfig,
    recovery: Recovery,
    window_id: str,
    input_params: earthkit.data.FieldList,
    out_keys: dict,
):
    if len(input_params.sel(param="cdir")) == 0:
        # Fetch solar radiation if not present. This is to handle the special case of step ranges where 
        # the end step is < 24 (e.g. 0-12) but uses solar radiation over 24hr window and therefore the end 
        # step of the solar radiation window does not match the end step of the tp step interval
        input_params += retrieve_sr24(config, param.cdir, int(window_id.split("-")[1]))

    logging.info(
        f"Processing {window_id}, fields: \n {input_params.ls(namespace='mars')}"
    )
    with ResourceMeter(f"Compute predictant and predictors: {window_id}"):
        tp = input_params.sel(param="tp").values
        maxmucape = input_params.sel(param="mxcape6").values
        sr = input_params.sel(param="cdir").values
        cpr = ratio(input_params.sel(param="cp").values, tp)
        ws700 = np.sqrt(
            input_params.sel(param="u").values ** 2
            + input_params.sel(param="v").values ** 2
        )

    with ResourceMeter(f"Compute realisations: {window_id}"):
        (
            pt_bc_allens_allwt,
            grid_bc_allens_allwt,
            wt_allens_allwt,
        ) = compute_weather_types(
            tp,
            cpr,
            ws700,
            maxmucape,
            sr,
            config.bp_location,
            config.fer_location,
            config.min_predictand,
            config.parallelisation.wt_batch_size,
            config.parallelisation.ens_batch_size,
        )

    # Save the grid-scale outputs and weather types for each member
    out_bs = config.outputs.bs
    out_wt = config.outputs.wt
    for index, field in enumerate(input_params.sel(param="tp")):
        template = field.metadata()._handle
        bs_message = grid_bc_template(
            template,
            {
                **out_keys,
                **out_bs.metadata,
            },
        )
        bs_message.set_array(
            "values", nan_to_missing(bs_message, grid_bc_allens_allwt[index])
        )
        out_bs.target.write(bs_message)
        out_bs.target.flush()

        wt_message = weather_types_template(template, {**out_keys, **out_wt.metadata})
        wt_message.set_array(
            "values", nan_to_missing(wt_message, wt_allens_allwt[index])
        )
        out_wt.target.write(wt_message)
        out_wt.target.flush()

    del grid_bc_allens_allwt
    del wt_allens_allwt

    with ResourceMeter(f"Compute the percentiles: {window_id}"):
        out_perc = config.outputs.perc
        template = input_params.sel(param="tp")[0].metadata()._handle
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
            message = point_scale_template(
                template, pert_number, total_number, grib_keys
            )
            message.set_array("values", nan_to_missing(message, quantile))
            out_perc.target.write(message)
        out_perc.target.flush()

    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-ecpoint", model=ECPointConfig).load()
    logger.info(yaml.dump(cfg.model_dump(by_alias=True)))
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
        checkpointed_windows = [
            x["window"] for x in recover.computed(param=param.name)
        ]
        managers[0].delete(checkpointed_windows)
        requesters = [
            FilteredParamRequester(
                param,
                cfg.sources,
                steps=managers[-1].dims["step"],
                total=cfg.total_fields,
            )
        ]
        dims = {k: set(val) for k, val in managers[0].dims.items()}
        for input_param in param.dependencies:
            managers.append(
                AccumulationManager.create(
                    input_param.accumulations,
                    {
                        **cfg.outputs.default.metadata,
                        **input_param.metadata,
                    },
                )
            )
            for dim, vals in managers[-1].dims.items():
                min_val = min(managers[0].dims[dim])
                max_val = max(managers[0].dims[dim])
                dims[dim].update([x for x in vals if x > min_val and x < max_val])
            requesters.append(
                FilteredParamRequester(
                    input_param,
                    cfg.sources,
                    steps=managers[-1].dims["step"],
                    total=cfg.total_fields,
                )
            )
        ecpoint_partial = functools.partial(ecpoint_iteration, cfg, param, recover)
        for keys, retrieved_data in parallel_data_retrieval(
            cfg.parallelisation.n_par_read,
            {k: sorted(list(val)) for k, val in dims.items()},
            requesters,
        ):
            ids = ", ".join(f"{k}={v}" for k, v in keys.items())
            window_id = None
            fields = None
            out_keys = {}
            with ResourceMeter(f"{ids}: Compute accumulation"):
                for index, param_data in enumerate(retrieved_data):
                    param_metadata, ens = param_data
                    for wid, completed_window in managers[index].feed(keys, ens):
                        if index == 0:
                            window_id = wid
                            out_keys = completed_window.grib_keys()
                            fields = earthkit.data.FieldList()
                        fields += earthkit.data.FieldList.from_array(
                            completed_window.values, to_ekmetadata(param_metadata)
                        )
                    del ens
                if fields is not None:
                    ecpoint_partial(window_id, fields, out_keys)

    recover.clean_file()


if __name__ == "__main__":
    sys.exit(main())
