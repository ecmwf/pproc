import argparse
from datetime import datetime
import functools
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

import eccodes
from meters import ResourceMeter

from pproc.common.config import Config, default_parser
from pproc.common.grib_helpers import construct_message
from pproc.common.io import (
    Target,
    nan_to_missing,
    read_template,
    target_from_location,
)
from pproc.common import parallel
from pproc.common.parallel import (
    QueueingExecutor,
    SynchronousExecutor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import Recovery
from pproc.common.window import Window
from pproc.common.window_manager import WindowManager


def postprocess(
    ens: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out_paramid: Optional[str] = None,
    out_keys: Optional[Dict[str, Any]] = None,
):
    """Post-process data and write to target

    Parameters
    ----------
    ens: numpy array (nens, npoints)
        Ensemble data
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    vmin: float, optional
        Minimum output value
    vmax: float, optional
        Maximum output value
    out_paramid: str, optional
        Parameter ID to set on the output
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    for i, field in enumerate(ens):
        if vmin is not None or vmax is not None:
            np.clip(field, vmin, vmax, out=field)

        grib_keys = {
            **out_keys,
            "perturbationNumber": i,
        }
        if template.get("type") == "cf" and i > 0:
            grib_keys.setdefault("type", "pf")
        if out_paramid is not None:
            grib_keys["paramId"] = out_paramid
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, field))
        target.write(message)


def get_parser() -> argparse.ArgumentParser:
    description = "Compute accumulations"
    parser = default_parser(description=description)
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--out-accum", required=True, help="Output target")
    return parser


class AccumParamConfig(ParamConfig):
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        super().__init__(name, options, overrides)
        self.vmin = options.get("vmin", None)
        self.vmax = options.get("vmax", None)


class AccumConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        self.total_fields = self.options.get("total_fields", self.num_members)

        self.out_keys = self.options.get("out_keys", {})

        self.params = [
            AccumParamConfig(pname, popt, overrides=self.override_input)
            for pname, popt in self.options["params"].items()
        ]
        self.steps = self.options.get("steps", [])
        self.windows = self.options.get("windows", [])

        self.sources = self.options.get("sources", {})

        date = self.options.get("date")
        self.date = None if date is None else datetime.strptime(str(date), "%Y%m%d%H")
        self.root_dir = self.options.get("root_dir", None)

        self.n_par_read = self.options.get("n_par_read", 1)
        self.n_par_compute = self.options.get("n_par_compute", 1)
        self.window_queue_size = self.options.get("queue_size", self.n_par_compute)


def postproc_iteration(
    param: AccumParamConfig,
    target: Target,
    recovery: Optional[Recovery],
    template: Union[str, eccodes.GRIBMessage],
    window_id: str,
    window: Window,
):
    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)
    with ResourceMeter(f"{param.name}, step {window.name}: Post-process"):
        postprocess(
            window.step_values,
            template,
            target,
            vmin=param.vmin,
            vmax=param.vmax,
            out_paramid=param.out_paramid,
            out_keys=window.grib_header(),
        )
    if recovery is not None:
        recovery.add_checkpoint(param.name, window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args(args)
    config = AccumConfig(args)
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()
    target = target_from_location(args.out_accum, overrides=config.override_output)
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
                param.window_config(config.windows, config.steps),
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
                window_manager.delete_windows(checkpointed_windows)
                print(
                    f"Recovery: param {param.name} looping from step {window_manager.unique_steps[0]}"
                )
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                config.sources,
                args.in_ens,
                config.num_members,
                config.total_fields,
            )
            postproc_partial = functools.partial(
                postproc_iteration, param, target, recovery
            )
            for keys, data in parallel_data_retrieval(
                config.n_par_read,
                {"step": window_manager.unique_steps},
                [requester],
                config.n_par_compute > 1,
            ):
                step = keys["step"]
                template, ens = data[0]
                with ResourceMeter(f"{param.name}, step {step}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(step, ens)
                    del ens
                for window_id, window in completed_windows:
                    executor.submit(postproc_partial, template, window_id, window)
            executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
