import argparse
from datetime import datetime
import functools
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import eccodes
from earthkit.meteo.stats import iter_quantiles
from meters import ResourceMeter

from pproc.common.config import Config, default_parser
from pproc.common.dataset import open_multi_dataset
from pproc.common.grib_helpers import construct_message
from pproc.common.io import (
    Target,
    missing_to_nan,
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
from pproc.common.recovery import Recovery
from pproc.common.steps import AnyStep
from pproc.common.window import Window
from pproc.common.window_manager import WindowManager


def read_ensemble(
    sources: dict, loc: str, members: int, dtype=np.float32, **kwargs
) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
    """Read data from a GRIB file as a single array

    Parameters
    ----------
    sources: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    members: int
        Number of ensemble members to expect
    dtype: numpy data type
        Data type for the result array (default float32)
    kwargs: any
        Extra arguments for backends that support them

    Returns
    -------
    eccodes.GRIBMessage
        GRIB template (first message read)
    numpy array (nfields, npoints)
        Read data
    """

    def set_number(keys):
        if keys.get("type") == "pf":
            keys["number"] = range(1, members)

    readers = open_multi_dataset(sources, loc, update=set_number, **kwargs)
    template = None
    data = None
    n_read = 0
    for reader in readers:
        with reader:
            message = reader.peek()
            if message is None:
                raise EOFError(f"No data in {loc!r}")
            if template is None:
                template = message
                data = np.empty(
                    (members, template.get("numberOfDataPoints")), dtype=dtype
                )
            for message in reader:
                i = message.get("perturbationNumber", 0)
                data[i, :] = missing_to_nan(message)
                n_read += 1
    if n_read != members:
        raise EOFError(f"Expected {members} fields in {loc!r}, got {n_read}")
    return template, data


def do_quantiles(
    ens: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    out_paramid: Optional[str] = None,
    n: int = 100,
    out_keys: Optional[Dict[str, Any]] = None,
):
    """Compute quantiles

    Parameters
    ----------
    ens: numpy array (nens, npoints)
        Ensemble data
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    out_paramid: str, optional
        Parameter ID to set on the output
    n: int
        Number of quantiles (default 100 = percentiles)
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    for i, quantile in enumerate(iter_quantiles(ens, n, method="sort")):
        grib_keys = {
            **out_keys,
            "type": "pb",
            "numberOfForecastsInEnsemble": n,
            "perturbationNumber": i,
        }
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


def parse_paramids(pid):
    if isinstance(pid, int):
        return [pid]
    if isinstance(pid, str):
        return pid.split("/")
    if isinstance(pid, list):
        if not all(isinstance(p, (int, str)) for p in pid):
            raise TypeError("Lists of paramids can contain only ints or strings")
        return pid
    raise TypeError(f"Invalid paramid type {type(pid)}")


class ParamConfig:
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        self.name = name
        self.in_paramids = parse_paramids(options["in"])
        self.combine = options.get("combine_operation", None)
        self.scale = options.get("scale", 1.0)
        self.out_paramid = options.get("out", None)
        self._in_keys = options.get("in_keys", {})
        self._out_keys = options.get("out_keys", {})
        self._steps = options.get("steps", None)
        self._windows = options.get("windows", None)
        self._in_overrides = overrides

    def in_keys(self, base: Optional[Dict[str, Any]] = None, **kwargs):
        keys = base.copy() if base is not None else {}
        keys.update(self._in_keys)
        keys.update(kwargs)
        keys.update(self._in_overrides)
        keys_list = []
        for pid in self.in_paramids:
            keys["param"] = pid
            keys_list.append(keys.copy())
        return keys_list

    def out_keys(self, base: Optional[Dict[str, Any]] = None, **kwargs):
        keys = base.copy() if base is not None else {}
        keys.update(self._out_keys)
        keys.update(kwargs)
        return keys

    def window_config(self, base: List[dict], base_steps: Optional[List[dict]] = None):
        if self._windows is not None:
            config = {"windows": self._windows}
            if self._steps is not None:
                config["steps"] = self._steps
            return config

        windows = []
        for coarse_cfg in base:
            coarse_window = Window(coarse_cfg)
            periods = [{"range": [step, step]} for step in coarse_window.steps]
            windows.append(
                {
                    "window_operation": "none",
                    "periods": periods,
                }
            )
        config = {"windows": windows}
        if base_steps:
            config["steps"] = base_steps

        return config


class ParamRequester:
    def __init__(self, param: ParamConfig, sources: dict, loc: str, members: int):
        self.param = param
        self.sources = sources
        self.loc = loc
        self.members = members

    def combine_data(self, data_list: List[np.ndarray]) -> np.ndarray:
        if self.param.combine is None:
            assert (
                len(data_list) == 1
            ), "Multiple input fields require a combine operation"
            return data_list[0]
        if self.param.combine == "norm":
            return np.linalg.norm(data_list, axis=0)
        return getattr(np, self.param.combine)(data_list, axis=0)

    def retrieve_data(
        self, fdb, step: AnyStep
    ) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
        data_list = []
        for in_keys in self.param.in_keys(step=str(step)):
            template, data = read_ensemble(
                self.sources, self.loc, self.members, **in_keys
            )
            data_list.append(data)
        return template, self.combine_data(data_list) * self.param.scale

    @property
    def name(self):
        return self.param.name


class QuantilesConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)
        self.num_quantiles = self.options.get("num_quantiles", 100)

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
    window: Window,
):
    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)
    with ResourceMeter(f"{param.name}, step {window.name}: Quantiles"):
        do_quantiles(
            window.step_values,
            template,
            target,
            param.out_paramid,
            n=config.num_quantiles,
            out_keys=window.grib_header(),
        )
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
                param.window_config(config.windows, config.steps), param.out_keys(config.out_keys)
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
                param, config.sources, args.in_ens, config.num_members
            )
            quantiles_partial = functools.partial(
                quantiles_iteration, config, param, target, recovery
            )
            for step, data in parallel_data_retrieval(
                config.n_par_read,
                window_manager.unique_steps,
                [requester],
                config.n_par_compute > 1,
            ):
                template, ens = data[0]
                with ResourceMeter(f"{param.name}, step {step}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(step, ens)
                    del ens
                for window_id, window in completed_windows:
                    executor.submit(quantiles_partial, template, window_id, window)
            executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
