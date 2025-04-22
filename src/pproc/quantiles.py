import functools
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

import eccodes
from earthkit.meteo.stats import iter_quantiles
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.grib_helpers import construct_message
from pproc.common.io import nan_to_missing
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import create_recovery, BaseRecovery
from pproc.config.targets import Target
from pproc.config.types import QuantilesConfig


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
    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")
    for i, quantile in enumerate(
        iter_quantiles(
            ens.reshape((-1, ens.shape[-1])), config.quantiles, method="sort"
        )
    ):
        pert_number, total_number = config.quantile_indices(i)
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
        message = construct_message(template, grib_keys)
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
                cfg.sources,
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
