import argparse
from datetime import datetime
import functools
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

import eccodes
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
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import Recovery
from pproc.common.steps import AnyStep
from pproc.common.window_manager import WindowManager


def write_histogram(
    hist: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    normalise: bool = True,
    scale: Optional[float] = None,
    out_paramid: Optional[str] = None,
    out_keys: Optional[Dict[str, Any]] = None,
):
    """Write histogram

    Parameters
    ----------
    hist: numpy array (nbins, npoints)
        Histogram data
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    normalise: bool
        If True, normalise the histogram
    scale: float or None
        If set, multiply values by this number
    out_paramid: str, optional
        Parameter ID to set on the output
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    hist = hist.astype(np.float32)
    if normalise:
        hist /= hist.sum(axis=0)
    if scale is not None:
        hist *= scale
    nbins = hist.shape[0]

    for i, hist_bin in enumerate(hist):
        grib_keys = {
            **out_keys,
            "totalNumber": nbins,
            "perturbationNumber": i + 1,
        }
        grib_keys.setdefault("type", "pd")
        if out_paramid is not None:
            grib_keys["paramId"] = out_paramid
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, hist_bin))
        target.write(message)


def get_parser() -> argparse.ArgumentParser:
    description = "Compute a histogram of an ensemble"
    parser = default_parser(description=description)
    parser.add_argument("--in-ens", required=True, help="Input ensemble source")
    parser.add_argument("--out-histogram", required=True, help="Output target")
    return parser


class HistParamConfig(ParamConfig):
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        super().__init__(name, options, overrides)
        self.bins = np.asarray(options["bins"])
        self.mod = options.get("mod", None)
        self.normalise = options.get("normalise", True)
        self.scale_out = options.get("scale_out", None)


def iter_ensemble(
    sources: dict,
    loc: str,
    dtype=np.float32,
    **kwargs,
) -> Iterator[Tuple[eccodes.GRIBMessage, np.ndarray]]:
    """Iterate over GRIB data, in arbitrary order

    Parameters
    ----------
    sources: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    dtype: numpy data type
        Data type for the result array (default float32)
    kwargs: any
        Extra arguments for backends that support them

    Yields
    ------
    eccodes.GRIBMessage
        GRIB message
    numpy array (npoints)
        Field data
    """

    readers = open_multi_dataset(sources, loc, **kwargs)
    for reader in readers:
        with reader:
            message = reader.peek()
            if message is None:
                raise EOFError(f"No data in {loc!r}")
            for message in reader:
                data = missing_to_nan(message)
                yield message, data.astype(dtype)


class HistParamRequester(ParamRequester):
    def __init__(
        self,
        param: HistParamConfig,
        sources: dict,
        loc: str,
    ):
        super().__init__(param, sources, loc, 0)

    def retrieve_data(
        self, fdb, step: AnyStep, **kwargs
    ) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
        assert isinstance(self.param, HistParamConfig)
        iterators = tuple(
            iter_ensemble(
                self.sources,
                self.loc,
                update=self._set_number,
                **in_keys,
            )
            for in_keys in self.param.in_keys(step=str(step), **kwargs)
        )
        nbins = len(self.param.bins) - 1
        template = None
        hist = None
        for result_list in zip(*iterators):
            if template is None:
                template = result_list[0][0]
                hist = np.zeros((nbins, result_list[0][1].shape[0]))
            data_list = [data for _, data in result_list]
            data = self.combine_data(data_list) * self.param.scale
            if self.param.mod is not None:
                data %= self.param.mod
            ind = np.digitize(data, self.param.bins) - 1
            if self.param.mod is not None:
                ind[ind < 0] = nbins - 1
                ind[ind >= nbins] = 0
            for i in range(nbins):
                hist[i, ind == i] += 1
        return template, hist


class HistogramConfig(Config):
    def __init__(self, args: argparse.Namespace, verbose: bool = True):
        super().__init__(args, verbose=verbose)

        self.num_members = self.options.get("num_members", 51)

        self.out_keys = self.options.get("out_keys", {})

        self.params = [
            HistParamConfig(pname, popt, overrides=self.override_input)
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


def write_iteration(
    config: HistogramConfig,
    param: HistParamConfig,
    target: Target,
    recovery: Optional[Recovery],
    template: Union[str, eccodes.GRIBMessage],
    step: AnyStep,
    hist: np.ndarray,
):
    if not isinstance(template, eccodes.GRIBMessage):
        template = read_template(template)
    with ResourceMeter(f"{param.name}, step {step!s}: Write histogram"):
        write_histogram(
            hist,
            template,
            target,
            param.normalise,
            param.scale_out,
            param.out_paramid,
            out_keys=param.out_keys(config.out_keys),
        )
    if recovery is not None:
        recovery.add_checkpoint(param.name, str(step))


def main(args: List[str] = sys.argv[1:]):
    sys.stdout.reconfigure(line_buffering=True)
    parser = get_parser()
    args = parser.parse_args(args)
    config = HistogramConfig(args)
    if config.root_dir is None or config.date is None:
        print("Recovery disabled. Set root_dir and date in config to enable.")
        recovery = None
        last_checkpoint = None
    else:
        recovery = Recovery(config.root_dir, args.config, config.date, args.recover)
        last_checkpoint = recovery.last_checkpoint()
    target = target_from_location(args.out_histogram, overrides=config.override_output)
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
            print(f"Processing {param.name}")
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

            requester = HistParamRequester(param, config.sources, args.in_ens)
            write_partial = functools.partial(
                write_iteration, config, param, target, recovery
            )
            for keys, data in parallel_data_retrieval(
                config.n_par_read,
                {"step": window_manager.unique_steps},
                [requester],
                config.n_par_compute > 1,
            ):
                step = keys["step"]
                print(f"Processing step {step}")
                template, hist = data[0]
                executor.submit(write_partial, template, step, hist)
            executor.wait()

    if recovery is not None:
        recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
