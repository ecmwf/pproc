import sys
from typing import List, Dict, Optional, Tuple
import argparse
import functools
import numpy as np
import datetime
import pandas as pd

import eccodes
from meters import ResourceMeter
from earthkit.meteo.stats import iter_quantiles

from pproc.common.param_requester import ParamConfig, ParamRequester, IndexFunc
from pproc.common.steps import AnyStep
from pproc.common.config import default_parser, Config
from pproc.common.recovery import Recovery
from pproc.common import parallel
from pproc.common.parallel import (
    SynchronousExecutor,
    QueueingExecutor,
    parallel_data_retrieval,
)
from pproc.common.io import target_from_location, nan_to_missing, GribMetadata
from pproc.common.accumulation import Accumulator
from pproc.common.window_manager import WindowManager
from pproc.common.grib_helpers import construct_message


def get_parser() -> argparse.ArgumentParser:
    description = "Compute quantiles of an ensemble"
    parser = default_parser(description=description)
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--bp-loc", required=True, help="Location of BP CSV file")
    parser.add_argument("--fer-loc", required=True, help="Location of FER CSV file")
    parser.add_argument(
        "--out-bs",
        required=True,
        help="Bias corrected rainfall at grid-scale output target",
    )
    parser.add_argument("--out-wt", required=True, help="Weather types output target")
    parser.add_argument(
        "--out-perc", required=True, help="Rainfall percentile output target"
    )
    return parser


class FilteredParamRequester(ParamRequester):
    def __init__(
        self,
        param: ParamConfig,
        sources: dict,
        loc: str,
        members: int,
        steps: list[int],
        total: Optional[int] = None,
        index_func: Optional[IndexFunc] = None,
    ):
        super().__init__(param, sources, loc, members, total, index_func)
        self.steps = steps

    def retrieve_data(
        self, fdb, step: AnyStep, **kwargs
    ) -> Tuple[List[GribMetadata], np.ndarray]:
        if step not in self.steps:
            return ([], None)
        return super().retrieve_data(fdb, step, **kwargs)


class ECPointConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.num_members)
        self.out_keys = self.options.get("out_keys", {})
        self.quantiles = self.options.get("quantiles", 100)

        self.params = [
            ParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["parameters"].items()
        ]
        self.sources = self.options.get("sources", {})

        date = self.options.get("date")
        self.date = None if date is None else datetime.strptime(str(date), "%Y%m%d%H")
        self.root_dir = self.options.get("root_dir", None)

        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)

        self.bp_location = args.bp_loc
        self.fer_location = args.fer_loc

        for attr in ["out_bs", "out_wt", "out_perc"]:
            location = getattr(args, attr)
            target = target_from_location(location, overrides=self.override_output)
            if self.n_par_compute > 1:
                target.enable_parallel(parallel)
            if args.recover:
                target.enable_recovery()
            self.__setattr__(attr, target)


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
    recovery: Recovery,
    window_id: str,
    metadata: list[eccodes.GRIBMessage],
    params: Dict[str, Accumulator],
):
    with ResourceMeter("Compute predictant and predictors"):
        tp = params["tp"].values
        maxmucape = params["mxcape6"].values
        sr = params["cdir"].values
        cpr = ratio(params["cp"].values, tp)
        ws700 = np.sqrt(params["u"].values ** 2 + params["v"].values ** 2)

    pt_bc_allens_allwt, grid_bc_allens_allwt, wt_allens_allwt = compute_weather_types(
        tp, cpr, ws700, maxmucape, sr, config.bp_location, config.fer_location
    )

    # Save the grid-scale outputs
    for index, template in enumerate(metadata):
        message = construct_message(
            template, {**config.out_keys, **params["tp"].grib_keys()}
        )
        message.set_array(
            "values", nan_to_missing(message, grid_bc_allens_allwt[index])
        )
        config.out_bs.write(message)
        config.out_bs.flush()

        message.set_array("values", nan_to_missing(message, wt_allens_allwt[index]))
        config.out_wt.write(message)
        config.out_wt.flush()

    del grid_bc_allens_allwt
    del wt_allens_allwt

    with ResourceMeter("Compute the percentiles"):
        for quantile in iter_quantiles(
            pt_bc_allens_allwt, config.quantiles, method="sort"
        ):
            # TODO: check keys that should be set for each percentile
            message = construct_message(
                metadata[0], {**config.out_keys, **params["tp"].grib_keys()}
            )
            message.set_array("values", nan_to_missing(message, quantile))
            config.out_perc.write(message)
            config.out_perc.flush()

    recovery.add_checkpoint(window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)

    parser = get_parser()
    args = parser.parse_args()
    config = ECPointConfig(args)
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()

    executor = (
        SynchronousExecutor()
        if config.n_par_compute == 1
        else QueueingExecutor(config.n_par_compute, config.window_queue_size)
    )

    with executor:
        managers = []
        requesters = []
        dims = {}
        for param in config.params:
            managers.append(
                WindowManager(
                    param.window_config([]),
                    param.out_keys(config.out_keys),
                )
            )
            for dim, vals in managers[-1].dims.items():
                dim_vals = dims.setdefault(dim, set(vals))
                dim_vals.update(vals)
            requesters.append(
                FilteredParamRequester(
                    param,
                    config.sources,
                    args.in_ens,
                    config.num_members,
                    steps=managers[-1].dims["step"],
                    total=config.total_fields,
                )
            )

        if last_checkpoint:
            checkpointed_windows = [
                recovery.checkpoint_identifiers(x)[1] for x in recovery.checkpoints
            ]
            managers[0].delete_windows(checkpointed_windows)

        ecpoint_partial = functools.partial(ecpoint_iteration, config, recovery)
        for keys, data in parallel_data_retrieval(
            config.n_par_read,
            {k: sorted(list(val)) for k, val in dims.items()},
            requesters,
        ):
            ids = ", ".join(f"{k}={v}" for k, v in keys.items())
            completed_windows = {}
            window_id = None
            with ResourceMeter(f"{ids}: Compute accumulation"):
                for index, param_data in enumerate(data):
                    metadata, ens = param_data
                    for wid, completed_window in managers[index].update_windows(
                        keys, ens
                    ):
                        if index == 0:
                            window_id = wid
                        completed_windows[metadata[0]["shortName"]] = completed_window
                    del ens

            if len(completed_windows) != 0:
                assert len(completed_windows) == len(
                    requesters
                ), f"Expected {len(requesters)}, got {completed_windows.keys()}."
                executor.submit(ecpoint_partial, window_id, metadata, completed_windows)
        executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
