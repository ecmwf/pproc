import sys
from typing import List, Optional, Tuple
import functools
import numpy as np
import pandas as pd
import signal
import logging
import yaml
import numexpr

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
    var_den_bitmap = np.where(var_den == 0, -9999, var_den)
    ratio_bitmap = var_num / var_den_bitmap
    output = np.where(ratio_bitmap == -9999, 0, ratio_bitmap)
    return output


def compute_weather_types(
    tp: np.ndarray,
    cpr: np.ndarray,
    ws700: np.ndarray,
    maxmucape: np.ndarray,
    sr: np.ndarray,
    bp_loc: str,
    fer_loc: str,
    min_predictand: float = 0.04,
    wt_batch_size: int = 40,
    ens_batch_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp_file = pd.read_csv(bp_loc, header=0, delimiter=",")
    fer_file = pd.read_csv(fer_loc, header=0, delimiter=",")
    bp = bp_file.iloc[:, 1:].to_numpy(dtype=np.float32)
    fer = fer_file.iloc[:, 1:].to_numpy(dtype=np.float32)
    codes_wt = bp_file.iloc[:, 0].to_numpy(dtype=np.float32)
    num_wt = bp.shape[0]
    num_pred = int(bp.shape[1] / 2)
    num_fer = fer.shape[1]

    num_gp = tp.shape[1]
    num_ens = tp.shape[0]
    # inizialize field for the new post-processed ensemble (CDF)
    # built from all raw ensemble members and all WTs
    pt_bc_allens_allwt = np.array([]).reshape(0, num_gp)
    # inizialize field for the bias corrected (bc) at grid-scale
    # fields for all raw ensemble members and all WTs
    grid_bc_allens_allwt = np.array([]).reshape(0, num_gp)
    # inizialize field for the wt for all raw ensemble members and all WTs
    wt_allens_allwt = np.array([]).reshape(0, num_gp)

    with ResourceMeter("Compute realisations"):
        for ind_em in range(0, num_ens, ens_batch_size):
            end_ind_em = min(num_ens, ind_em + ens_batch_size)
            logger.info(f"Ensemble member: {ind_em} - {end_ind_em - 1}")

            predictand = tp[ind_em:end_ind_em]
            predictand = np.where(predictand < min_predictand, 0, predictand)
            predictors = np.asarray(
                [
                    cpr[ind_em:end_ind_em],
                    tp[ind_em:end_ind_em],
                    ws700[ind_em:end_ind_em],
                    maxmucape[ind_em:end_ind_em],
                    sr[ind_em:end_ind_em],
                ]
            )  # (num_pred, ens_batch_size, num_gp)

            predictors = predictors.reshape(
                1, *predictors.shape
            )  # (1, num_pred, ens_batch_size, num_gp)
            print("predictors", predictors.shape)
            thr_inf2 = bp[:, 0:-1:2].reshape(
                num_wt, num_pred, 1, 1
            )  # (num_wt, num_pred, 1, 1)
            thr_sup2 = bp[:, 1::2].reshape(
                num_wt, num_pred, 1, 1
            )  # (num_wt, num_pred, 1, 1)
            print("thr_inf2/thr_sup2", thr_inf2.shape, thr_sup2.shape)
            temp_wts = numexpr.evaluate(
                "prod(where((predictors >= thr_inf2) & (predictors < thr_sup2), 1, 0), axis=1)",
                local_dict={
                    "predictors": predictors,
                    "thr_inf2": thr_inf2,
                    "thr_sup2": thr_sup2,
                },
            )  # (num_wt, ens_batch_size, num_gp)
            print("temp_wts", temp_wts.shape)

            wt_allwt = np.einsum(
                "i,i...", codes_wt, temp_wts
            )  # (ens_batch_size, num_gp)
            print("wt_allwt", wt_allwt.shape)
            wt_rain_allwt = numexpr.evaluate(
                "pred * wts",
                local_dict={"pred": predictand[None, ...], "wts": temp_wts},
            )  # (num_wt, ens_batch_size, num_gp)
            print("wt_rain_allwt", wt_rain_allwt.shape)

            pt_bc_allwt = np.zeros(
                (predictand.shape[0], num_fer, num_gp)
            )  # (ens_batch_size, num_fer, num_gp)
            for index in range(0, num_wt, wt_batch_size):
                end_index = min(index + wt_batch_size, num_wt)
                wt_size = end_index - index
                fer_reshaped = fer[index:end_index].reshape(wt_size, 1, num_fer, 1)
                wt_rain_allwt_reshaped = wt_rain_allwt[index:end_index].reshape(
                    wt_size, ens_batch_size, 1, num_gp
                )
                cdf_wtbatch = numexpr.evaluate(
                    "sum((fer + 1) * wt_rain, axis=0)",
                    local_dict={"fer": fer_reshaped, "wt_rain": wt_rain_allwt_reshaped},
                )  # (ens_batch_size, num_fer, num_gp)
                pt_bc_allwt = numexpr.evaluate(
                    "pt + cdf", local_dict={"pt": pt_bc_allwt, "cdf": cdf_wtbatch}
                )
            print("pt_bc_allwt", pt_bc_allwt.shape)

            grid_bc_allens_allwt = np.vstack(
                (grid_bc_allens_allwt, np.mean(pt_bc_allwt, axis=1))
            )
            pt_bc_allens_allwt = np.vstack(
                (
                    pt_bc_allens_allwt,
                    pt_bc_allwt.reshape(ens_batch_size * num_fer, num_gp, order="C"),
                )
            )
            wt_allens_allwt = np.vstack(
                (wt_allens_allwt, np.where(predictand == 0, 9999, wt_allwt))
            )

    print("grid_bc_allens_allwt", grid_bc_allens_allwt.shape)
    print("pt_bc_allens_allwt", pt_bc_allens_allwt.shape)
    print("wt_allens_allwt", wt_allens_allwt.shape)
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
        # Fetch solar radiation if not present, which can be the case because it always has to use 24hr windows
        input_params += retrieve_sr24(config, param.cdir, int(window_id.split("-")[1]))

    logging.info(
        f"Processing {window_id}, fields: \n {input_params.ls(namespace='mars')}"
    )
    with ResourceMeter("Compute predictant and predictors"):
        tp = input_params.sel(param="tp").values
        maxmucape = input_params.sel(param="mxcape6").values
        sr = input_params.sel(param="cdir").values
        cpr = ratio(input_params.sel(param="cp").values, tp)
        ws700 = np.sqrt(
            input_params.sel(param="u").values ** 2
            + input_params.sel(param="v").values ** 2
        )

    pt_bc_allens_allwt, grid_bc_allens_allwt, wt_allens_allwt = compute_weather_types(
        tp,
        cpr,
        ws700,
        maxmucape,
        sr,
        config.bp_location,
        config.fer_location,
        config.min_predictand,
    )

    # Save the grid-scale outputs
    out_bs = config.outputs.bs
    out_wt = config.outputs.wt
    for index, field in enumerate(input_params.sel(param="tp")):
        template = field.metadata()._handle
        bs_message = construct_message(template, {**out_bs.metadata, **out_keys})
        bs_message.set_array(
            "values", nan_to_missing(bs_message, grid_bc_allens_allwt[index])
        )
        out_bs.target.write(bs_message)
        out_bs.target.flush()

        wt_message = construct_message(template, {**out_wt.metadata, **out_keys})
        wt_message.set_array(
            "values", nan_to_missing(wt_message, wt_allens_allwt[index])
        )
        out_wt.target.write(wt_message)
        out_wt.target.flush()

    del grid_bc_allens_allwt
    del wt_allens_allwt

    with ResourceMeter("Compute the percentiles"):
        out_perc = config.outputs.perc
        template = input_params.sel(param="tp")[0].metadata()._handle
        for quantile in iter_quantiles(
            pt_bc_allens_allwt, config.quantiles, method="sort"
        ):
            # TODO: check keys that should be set for each percentile
            message = construct_message(template, {**out_perc.metadata, **out_keys})
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

    with create_executor(cfg.parallelisation) as executor:
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
                        executor.submit(ecpoint_partial, window_id, fields, out_keys)
            executor.wait()

    recover.clean_file()


if __name__ == "__main__":
    sys.exit(main())
