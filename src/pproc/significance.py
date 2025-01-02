import argparse
from datetime import datetime
import functools
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import mannwhitneyu

import eccodes
from meters import ResourceMeter

from pproc.common.accumulation import Accumulator
from pproc.common.config import Config, default_parser
from pproc.common.grib_helpers import construct_message
from pproc.common.io import Target, nan_to_missing, read_template, target_from_location
from pproc.common import parallel
from pproc.common.parallel import (
    QueueingExecutor,
    SynchronousExecutor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import Recovery
from pproc.common.window_manager import WindowManager
from pproc.signi.clim import retrieve_clim


def signi(
    fc: np.ndarray,
    clim: np.ndarray,
    template: eccodes.GRIBMessage,
    clim_template: eccodes.GRIBMessage,
    target: Target,
    out_paramid: Optional[str] = None,
    out_keys: Optional[Dict[str, Any]] = None,
    epsilon: Optional[float] = None,
    epsilon_is_abs: bool = True,
):
    """Compute significance

    NOTE 1: In line with the legacy code, the result is actually the p-value
    associated with the WMW test, between 0 and 100.

    NOTE 2: If ``epsilon`` is set, the ``fc`` and ``clim`` are modified in
    place.

    Parameters
    ----------
    fc: numpy array (..., npoints)
        Forecast data (all dimensions but the last are squashed together)
    clim: numpy array (..., npoints)
        Climatology data (all dimensions but the last are squashed together)
    template: eccodes.GRIBMessage
        GRIB template for output (from forecast)
    clim_template: eccodes.GRIBMessage
        GRIB template for output (from climatology)
    target: Target
        Target to write to
    out_paramid: str, optional
        Parameter ID to set on the output
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    epsilon: float, optional
        If set, set forecast and climatology values below this threshold to 0
    epsilon_is_abs: bool
        If True (default), the absolute value of the forecast and climatology is
        compared to ``epsilon``. Otherwise, the signed value is compared.
    """
    assert (
        fc.shape[-1] == clim.shape[-1]
    ), "Forecast and climatology are on different grids"

    if epsilon is not None:
        if epsilon_is_abs:
            fc[np.abs(fc) <= epsilon] = 0.0
            clim[np.abs(clim) <= epsilon] = 0.0
        else:
            fc[fc <= epsilon] = 0.0
            clim[clim <= epsilon] = 0.0

    result = mannwhitneyu(
        fc.reshape((-1, fc.shape[-1])),
        clim.reshape((-1, clim.shape[-1])),
        alternative="two-sided",
        method="asymptotic",
        use_continuity=False,
    )
    pvalue = result.pvalue
    pvalue *= 100.0

    # If there is no signal whatsoever (e.g. forecast and climatology all zero)
    # the variance of the test will be zero, leading to the p-value being
    # undefined (NaN). We set it to 0 instead.
    zero_variance = np.logical_and(
        np.isnan(pvalue), np.logical_not(np.isnan(result.statistic))
    )
    pvalue[zero_variance] = 0.0

    if out_keys is None:
        out_keys = {}
    grib_keys = out_keys.copy()
    grib_keys.setdefault("type", "taem")
    if out_paramid is not None:
        grib_keys["paramId"] = out_paramid

    clim_keys = {key: clim_template.get(key) for key in []}
    grib_keys.update(clim_keys)
    message = construct_message(template, grib_keys)
    message.set_array("values", nan_to_missing(message, pvalue))
    target.write(message)


def get_parser() -> argparse.ArgumentParser:
    description = "Compute significance using a Wilcoxon-Mann-Whitney test"
    parser = default_parser(description=description)
    parser.add_argument("--in-fc", required=True, help="Input forecast")
    parser.add_argument("--in-clim", required=True, help="Input climatology")
    parser.add_argument(
        "--in-clim-em", default=None, help="Input climatology ensemble mean"
    )
    parser.add_argument("--out-sig", required=True, help="Output target")
    return parser


class SigniParamConfig(ParamConfig):
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        super().__init__(name, options, overrides)

        clim_options = options.copy()
        if "clim" in options:
            clim_options.update(clim_options.pop("clim"))
        self.clim_param = ParamConfig(f"clim_{name}", clim_options, overrides)
        clim_options = options.copy()
        if "clim_em" in options:
            clim_options.update(options.pop("clim_em"))
        elif "clim" in options:
            clim_options.update(options.pop("clim"))
        self.clim_em_param = ParamConfig(f"clim_em_{name}", clim_options, overrides)

        self.epsilon = options.get("epsilon", None)
        if self.epsilon is not None:
            self.epsilon = float(self.epsilon)
        self.epsilon_is_abs = options.get("epsilon_is_abs", True)


class SigniConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.num_members)

        self.clim_loc: str = args.in_clim
        self.clim_em_loc: Optional[str] = args.in_clim_em
        self.clim_num_members = self.options.get("clim_num_members", 11)
        self.clim_total_fields = self.options.get(
            "clim_total_fields", self.clim_num_members
        )

        self.out_keys = self.options.get("out_keys", {})

        self.params = [
            SigniParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["params"].items()
        ]

        self.sources = self.options.get("sources", {})

        date = self.options.get("date")
        self.date = None if date is None else datetime.strptime(str(date), "%Y%m%d%H")
        self.root_dir = self.options.get("root_dir", None)

        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)


def signi_iteration(
    config: SigniConfig,
    param: SigniParamConfig,
    target: Target,
    recovery: Optional[Recovery],
    template: Union[str, eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):

    with ResourceMeter(f"{param.name}, window {window_id}: Retrieve climatology"):
        steprange = accum.grib_keys()["stepRange"]
        clim_accum, clim_template = retrieve_clim(
            param.clim_param,
            config.sources,
            config.clim_loc,
            config.clim_num_members,
            config.clim_total_fields,
            step=steprange,
        )
        clim = clim_accum.values
        assert clim is not None
        if config.clim_em_loc is not None:
            clim_em_accum, _ = retrieve_clim(
                param.clim_em_param, config.sources, config.clim_em_loc, step=steprange
            )
            clim_em = clim_em_accum.values
            assert clim_em is not None
            # Assumed clim_em shape: (ndates, 1, npoints)
            # Assumed clim shape: (ndates, nhdates*members, npoints)
            exp_shape = (clim.shape[0], 1, clim.shape[-1])
            assert (
                clim_em.shape == exp_shape
            ), f"Wrong ensemble mean shape {clim_em.shape}, expected {exp_shape}"
            clim -= clim_em

    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)
    with ResourceMeter(f"{param.name}, window {window_id}: Compute significance"):
        fc = accum.values
        assert fc is not None
        signi(
            fc,
            clim,
            template,
            clim_template,
            target,
            out_paramid=param.out_paramid,
            out_keys=accum.grib_keys(),
            epsilon=param.epsilon,
            epsilon_is_abs=param.epsilon_is_abs,
        )
        target.flush()
    if recovery is not None:
        recovery.add_checkpoint(param.name, window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args()
    config = SigniConfig(args)
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()
    target = target_from_location(args.out_sig, overrides=config.override_output)
    if config.n_par_compute > 1:
        target.enable_parallel(parallel)
    if recovery is not None and args.recover:
        target.enable_recovery()

    executor = (
        SynchronousExecutor()
        if config.n_par_compute == 1
        else QueueingExecutor(config.n_par_compute, config.window_queue_size)
    )

    with executor:
        for param in config.params:
            window_manager = WindowManager(
                param.window_config([]),
                param.out_keys(config.out_keys),
            )
            if last_checkpoint:
                if param.name not in last_checkpoint:
                    print(f"Recovery: skipping completed param {param.name}")
                    continue
                checkpointed_windows = [
                    recovery.checkpoint_identifiers(x)[1]
                    for x in recovery.checkpoints
                    if param.name in x
                ]
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                config.sources,
                args.in_fc,
                config.num_members,
                config.total_fields,
            )
            signi_partial = functools.partial(
                signi_iteration, config, param, target, recovery
            )
            for keys, data in parallel_data_retrieval(
                config.n_par_read,
                window_manager.dims,
                [requester],
                config.n_par_compute > 1,
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                template, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(signi_partial, template, window_id, accum)
            executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
