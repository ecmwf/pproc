import argparse
from datetime import datetime
import functools
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

import eccodes
from earthkit.meteo.stats import iter_quantiles
from meters import ResourceMeter

from pproc.common.accumulation import Accumulator
from pproc.common.config import Config, default_parser
from pproc.common.grib_helpers import construct_message
from pproc.common.io import (
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
from pproc.common.window_manager import WindowManager
from pproc.config.targets import Target


def do_quantiles(
    ens: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    out_paramid: Optional[str] = None,
    n: Union[int, List[float]] = 100,
    out_keys: Optional[Dict[str, Any]] = None,
):
    """Compute quantiles

    Parameters
    ----------
    ens: numpy array (..., npoints)
        Ensemble data (all dimensions but the last are squashed together)
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    out_paramid: str, optional
        Parameter ID to set on the output
    n: int or list of floats
        List of quantiles to compute, e.g. `[0., 0.25, 0.5, 0.75, 1.]`, or
        number of evenly-spaced intervals (default 100 = percentiles).
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    even_spacing = isinstance(n, int) or np.all(np.diff(n) == n[1] - n[0])
    num_quantiles = n if isinstance(n, int) else (len(n) - 1)
    total_number = num_quantiles if even_spacing else 100
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")
    for i, quantile in enumerate(
        iter_quantiles(ens.reshape((-1, ens.shape[-1])), n, method="sort")
    ):
        pert_number = i if even_spacing else int(n[i] * 100)
        grib_keys = {**out_keys}
        if edition == 1:
            grib_keys.update(
                {
                    "totalNumber": total_number,
                    "perturbationNumber": pert_number,
                }
            )
        else:
            grib_keys.setdefault("productDefinitionTemplateNumber", 86)
            grib_keys.update(
                {
                    "totalNumberOfQuantiles": total_number,
                    "quantileValue": pert_number,
                }
            )
        grib_keys.setdefault("type", "pb")
        if out_paramid is not None:
            grib_keys["paramId"] = out_paramid
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, quantile))
        target.write(message)
        target.flush()


def get_parser() -> argparse.ArgumentParser:
    description = "Compute quantiles of an ensemble"
    parser = default_parser(description=description)
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--out-quantiles", required=True, help="Output target")
    return parser


class QuantilesConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        if "quantiles" in self.options:
            if "num_quantiles" in self.options:
                raise ValueError("Cannot specify both num_quantiles and quantiles")
            self.quantiles = self.options["quantiles"]
        else:
            self.quantiles = self.options.get("num_quantiles", 100)
        self.total_fields = self.options.get("total_fields", self.num_members)

        self.out_keys = self.options.get("out_keys", {})

        self.params = [
            ParamConfig(pname, popt, overrides=self.override_input)
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


def quantiles_iteration(
    config: QuantilesConfig,
    param: ParamConfig,
    target: Target,
    recovery: Optional[Recovery],
    template: Union[str, eccodes.GRIBMessage],
    window_id: str,
    accum: Accumulator,
):
    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)
    with ResourceMeter(f"{param.name}, step {window_id}: Quantiles"):
        ens = accum.values
        assert ens is not None
        do_quantiles(
            ens,
            template,
            target,
            param.out_paramid,
            n=config.quantiles,
            out_keys=accum.grib_keys(),
        )
        target.flush()
    if recovery is not None:
        recovery.add_checkpoint(param.name, window_id)


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args(args)
    config = QuantilesConfig(args)
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()
    target = target_from_location(args.out_quantiles, overrides=config.override_output)
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
                new_start = window_manager.delete_windows(checkpointed_windows)
                print(f"Recovery: param {param.name} looping from step {new_start}")
                last_checkpoint = None  # All remaining params have not been run

            requester = ParamRequester(
                param,
                config.sources,
                args.in_ens,
                config.num_members,
                config.total_fields,
            )
            quantiles_partial = functools.partial(
                quantiles_iteration, config, param, target, recovery
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
                    executor.submit(quantiles_partial, template, window_id, accum)
            executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
