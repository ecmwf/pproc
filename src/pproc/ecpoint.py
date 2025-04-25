import sys
from typing import List, Dict, Optional, Tuple
import argparse
import functools
import numpy as np
import datetime
import pandas as pd
import signal

import eccodes
from meters import ResourceMeter
from earthkit.meteo.stats import iter_quantiles
from conflator import Conflator

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
from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.grib_helpers import construct_message


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
    var_den_bitmap = np.where(var_den == 0, 9999, var_den)
    ratio_bitmap = var_num / var_den_bitmap
    output = np.where(ratio_bitmap == 9999, 0, ratio_bitmap)
    return output


def compute_weather_types(
    tp: np.ndarray,
    cpr: np.ndarray,
    ws700: np.ndarray,
    maxmucape: np.ndarray,
    sr: np.ndarray,
    bp_loc: str,
    fer_loc: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bp_file = pd.read_csv(bp_loc, header=0, delimiter=",")
    fer_file = pd.read_csv(fer_loc, header=0, delimiter=",")
    bp = bp_file.iloc[:, 1:]
    fer = fer_file.iloc[:, 1:]
    codes_wt = bp_file.iloc[:, 0]
    num_wt = bp.shape[0]
    num_pred = int(bp.shape[1] / 2)
    num_fer = fer.shape[1]

    num_gp = tp.shape[1]
    num_ens = tp.shape[0]
    # inizialize field for the new post-processed ensemble (CDF) built from all raw ensemble members and all WTs
    pt_bc_allens_allwt = np.array([]).reshape(0, num_gp)
    # inizialize field for the bias corrected (bc) at grid-scale fields for all raw ensemble members and all WTs
    grid_bc_allens_allwt = np.array([]).reshape(0, num_gp)
    # inizialize field for the wt for all raw ensemble members and all WTs
    wt_allens_allwt = np.array([]).reshape(0, num_gp)

    with ResourceMeter("Compute realisations"):
        for ind_em in range(num_ens):
            predictand = tp[ind_em]
            predictors = [
                cpr[ind_em],
                tp[ind_em],
                ws700[ind_em],
                maxmucape[ind_em],
                sr[ind_em],
            ]
            predictand = np.where(predictand < 0.04, 0, predictand)

            pt_bc_singleens_allwt = np.zeros(
                (num_fer, predictand.shape[0])
            )  # inizializes the field that will contain the new post-processed ensemble (CDF) built from each raw ensemble member and all WTs
            wt_singleens_allwt = 0  # initializes the field that will locate all the WTs within the grid boxes.

            for ind_wt in range(num_wt):
                print(" - EM n.", ind_em + 1, " and WT n.", ind_wt + 1)

                ind_inf = 0
                ind_sup = 1
                tempWT = np.ones_like(predictand)

                for ind_pred in range(num_pred):
                    thr_inf = bp.iloc[ind_wt, ind_inf]
                    thr_sup = bp.iloc[ind_wt, ind_sup]
                    tempWT1 = np.where(
                        (predictors[ind_pred] >= thr_inf)
                        & (predictors[ind_pred] < thr_sup),
                        1,
                        0,
                    )  # combination with a single predictor
                    tempWT = tempWT * tempWT1  # combination with all predictors
                    ind_inf = ind_inf + 2
                    ind_sup = ind_inf + 1

                wt_singleens_allwt = wt_singleens_allwt + (codes_wt[ind_wt] * tempWT)

                WT_RainVals_SingleENS_SingleWT = predictand * tempWT
                CDF_SingleENS_SingleWT = []
                for ind_fer in range(num_fer):
                    CDF_SingleENS_SingleWT.append(
                        WT_RainVals_SingleENS_SingleWT * (fer.iloc[ind_wt, ind_fer] + 1)
                    )

                pt_bc_singleens_allwt = pt_bc_singleens_allwt + np.array(
                    CDF_SingleENS_SingleWT
                )
                CDF_SingleENS_SingleWT = None

            grid_bc_allens_allwt = np.vstack(
                (grid_bc_allens_allwt, np.mean(pt_bc_singleens_allwt, axis=0))
            )
            pt_bc_allens_allwt = np.vstack((pt_bc_allens_allwt, pt_bc_singleens_allwt))
            wt_allens_allwt = np.vstack((wt_allens_allwt, wt_singleens_allwt))

    return pt_bc_allens_allwt, grid_bc_allens_allwt, wt_allens_allwt


def ecpoint_iteration(
    config: ECPointConfig,
    param: ECPointParamConfig,
    recovery: Recovery,
    window_id: str,
    metadata: list[eccodes.GRIBMessage],
    input_params: Dict[str, Accumulator],
):
    with ResourceMeter("Compute predictant and predictors"):
        tp = input_params["tp"].values
        maxmucape = input_params["mxcape6"].values
        sr = input_params["cdir"].values
        cpr = ratio(input_params["cp"].values, tp)
        ws700 = np.sqrt(input_params["u"].values ** 2 + input_params["v"].values ** 2)

    pt_bc_allens_allwt, grid_bc_allens_allwt, wt_allens_allwt = compute_weather_types(
        tp, cpr, ws700, maxmucape, sr, config.bp_location, config.fer_location
    )

    # Save the grid-scale outputs
    out_bs = config.outputs.bs
    out_wt = config.outputs.wt
    for index, template in enumerate(metadata):
        bs_message = construct_message(
            template, {**out_bs.metadata, **input_params["tp"].grib_keys()}
        )
        bs_message.set_array(
            "values", nan_to_missing(bs_message, grid_bc_allens_allwt[index])
        )
        out_bs.target.write(bs_message)
        out_bs.target.flush()

        wt_message = construct_message(
            template, {**out_wt.metadata, **input_params["tp"].grib_keys()}
        )
        wt_message.set_array(
            "values", nan_to_missing(wt_message, wt_allens_allwt[index])
        )
        out_wt.target.write(wt_message)
        out_wt.target.flush()

    del grid_bc_allens_allwt
    del wt_allens_allwt

    with ResourceMeter("Compute the percentiles"):
        out_perc = config.outputs.perc
        for quantile in iter_quantiles(
            pt_bc_allens_allwt, config.quantiles, method="sort"
        ):
            # TODO: check keys that should be set for each percentile
            message = construct_message(
                metadata[0], {**out_perc.metadata, **input_params["tp"].grib_keys()}
            )
            message.set_array("values", nan_to_missing(message, quantile))
            out_perc.target.write(message)
            out_perc.target.flush()

    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-ecpoint", model=ECPointConfig).load()
    cfg.print()
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
                completed_windows = {}
                window_id = None
                metadata = None
                with ResourceMeter(f"{ids}: Compute accumulation"):
                    for index, param_data in enumerate(retrieved_data):
                        param_metadata, ens = param_data
                        for wid, completed_window in managers[index].feed(keys, ens):
                            if index == 0:
                                window_id = wid
                                metadata = param_metadata
                            completed_windows[
                                metadata[0]["shortName"]
                            ] = completed_window
                        del ens

                if len(completed_windows) != 0:
                    assert len(completed_windows) == len(
                        requesters
                    ), f"Expected {len(requesters)}, got {completed_windows.keys()}."
                    executor.submit(
                        ecpoint_partial, window_id, metadata, completed_windows
                    )
            executor.wait()

    recover.clean_file()


if __name__ == "__main__":
    sys.exit(main())
