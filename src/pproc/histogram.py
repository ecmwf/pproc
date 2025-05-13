# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import sys
from typing import Any, Dict, Iterator, Optional, Tuple
import signal

import numpy as np

import eccodes
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.dataset import open_multi_dataset
from pproc.common.grib_helpers import construct_message
from pproc.common.io import (
    missing_to_nan,
    nan_to_missing,
    GribMetadata,
)
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery, BaseRecovery
from pproc.common.steps import AnyStep
from pproc.config.types import HistogramConfig, HistParamConfig
from pproc.config.targets import Target
from pproc.config.io import SourceCollection, Source


def write_histogram(
    hist: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    normalise: bool = True,
    scale: Optional[float] = None,
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
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, hist_bin))
        target.write(message)


def iter_ensemble(
    source: Source,
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
    loc = source.location()
    readers = open_multi_dataset(source.legacy_config(), loc, **kwargs)
    for reader in readers:
        with reader:
            message = reader.peek()
            if message is None:
                raise EOFError(f"No data in {source!r}")
            for message in reader:
                data = missing_to_nan(message)
                yield message, data.astype(dtype)


class HistParamRequester(ParamRequester):
    def __init__(
        self,
        param: HistParamConfig,
        sources: SourceCollection,
        total: int,
    ):
        super().__init__(param, sources, total)

    def retrieve_data(
        self, step: AnyStep, **kwargs
    ) -> Tuple[list[GribMetadata], np.ndarray]:
        assert isinstance(self.param, HistParamConfig)
        sources = self.param.in_sources(self.sources, "fc", step=str(step), **kwargs)
        metadata = [src.base_request() for src in sources]

        iterators = tuple(
            iter_ensemble(
                source,
                dtype=self.param.dtype,
            )
            for source in sources
        )
        nbins = len(self.param.bins) - 1
        template = None
        hist = None
        for result_list in zip(*iterators):
            if template is None:
                template = result_list[0][0]
                hist = np.zeros((nbins, result_list[0][1].shape[0]))
            data_list = [data for _, data in result_list]
            _, data_list = self.param.preprocessing.apply(metadata, data_list)
            data = data_list[0]
            if self.param.mod is not None:
                data %= self.param.mod
            ind = np.digitize(data, self.param.bins) - 1
            if self.param.mod is not None:
                ind[ind < 0] = nbins - 1
                ind[ind >= nbins] = 0
            for i in range(nbins):
                hist[i, ind == i] += 1
        return [GribMetadata(template)], hist


def write_iteration(
    param: HistParamConfig,
    target: Target,
    recovery: BaseRecovery,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"{param.name}, window {window_id!s}: Write histogram"):
        hist = accum.values
        assert hist is not None
        write_histogram(
            hist,
            template,
            target,
            param.normalise,
            param.scale_out,
            out_keys=accum.grib_keys(),
        )
    recovery.add_checkpoint(param=param.name, window=str(window_id))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-histogram", model=HistogramConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            accum_manager = AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.histogram.metadata,
                    **param.metadata,
                },
            )
            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = HistParamRequester(param, cfg.sources, cfg.total_fields)
            write_partial = functools.partial(
                write_iteration, param, cfg.outputs.histogram.target, recovery
            )
            for keys, data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                metadata, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = accum_manager.feed(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(write_partial, metadata[0], window_id, accum)

            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
