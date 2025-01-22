import functools
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np

import eccodes
from earthkit.meteo.stats import iter_quantiles
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.grib_helpers import construct_message
from pproc.common.io import (
    nan_to_missing,
    read_template,
)
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
)
from pproc.common.param_requester import ParamConfig, ParamRequester
from pproc.common.recovery import create_recovery, BaseRecovery
from pproc.common.window_manager import WindowManager
from pproc.config.targets import Target
from pproc.config.types import QuantilesConfig


def do_quantiles(
    ens: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
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
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, quantile))
        target.write(message)
        target.flush()


def quantiles_iteration(
    config: QuantilesConfig,
    param: ParamConfig,
    recovery: BaseRecovery,
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
            config.outputs.quantiles.target,
            n=config.quantiles,
            out_keys=accum.grib_keys(),
        )
        config.outputs.quantiles.target.flush()
    recovery.add_checkpoint(param.name, window_id)


def main(args=None):
    sys.stdout.reconfigure(line_buffering=True)
    cfg = Conflator(app_name="pproc-quantiles", model=QuantilesConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            window_manager = WindowManager(
                param.accumulations,
                {
                    **cfg.outputs.quantiles.metadata,
                    **param.metadata,
                },
            )

            checkpointed_windows = recovery.computed(param.name)
            new_start = window_manager.delete_windows(checkpointed_windows)
            if new_start is None:
                print(f"Recovery: skipping completed param {param.name}")
                continue

            print(f"Recovery: param {param.name} starting from step {new_start}")

            requester = ParamRequester(
                param,
                cfg.sources,
                cfg.members,
                cfg.total_fields,
            )
            quantiles_partial = functools.partial(
                quantiles_iteration, cfg, param, recovery
            )
            for keys, data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                window_manager.dims,
                [requester],
                cfg.parallelisation.n_par_compute > 1,
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                template, ens = data[0]
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    completed_windows = window_manager.update_windows(keys, ens)
                    del ens
                for window_id, accum in completed_windows:
                    executor.submit(quantiles_partial, template, window_id, accum)
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
