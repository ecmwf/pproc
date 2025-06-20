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
from typing import Any, Dict, Optional
import signal

import numpy as np

import eccodes
from earthkit.meteo.stats import iter_quantiles
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.io import nan_to_missing
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import create_recovery, BaseRecovery
from pproc.config.types import QuantilesConfig
from pproc.quantile.grib import quantiles_template


def do_quantiles(
    config: QuantilesConfig,
    ens: np.ndarray,
    template: eccodes.GRIBMessage,
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
    n: int or list of floats
        List of quantiles to compute, e.g. `[0., 0.25, 0.5, 0.75, 1.]`, or
        number of evenly-spaced intervals (default 100 = percentiles).
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    for i, quantile in enumerate(
        iter_quantiles(
            ens.reshape((-1, ens.shape[-1])), config.quantiles, method="sort"
        )
    ):
        pert_number, total_number = config.quantile_indices(i)
        message = quantiles_template(template, pert_number, total_number, out_keys)
        message.set_array("values", nan_to_missing(message, quantile))
        config.outputs.quantiles.target.write(message)


def quantiles_iteration(
    config: QuantilesConfig,
    param: ParamConfig,
    recovery: BaseRecovery,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"{param.name}, step {window_id}: Quantiles"):
        ens = accum.values
        assert ens is not None
        do_quantiles(
            config,
            ens,
            template,
            out_keys=accum.grib_keys(),
        )
        config.outputs.quantiles.target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-quantiles", model=QuantilesConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            accum_manager = AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.quantiles.metadata,
                    **param.metadata,
                },
            )

            checkpointed_windows = [
                x["window"] for x in recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = ParamRequester(
                param,
                cfg.inputs,
                cfg.total_fields,
            )
            quantiles_partial = functools.partial(
                quantiles_iteration, cfg, param, recovery
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
                    executor.submit(quantiles_partial, metadata[0], window_id, accum)
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
