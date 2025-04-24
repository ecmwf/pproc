import functools
import sys
from typing import Any, Dict, Optional
import signal

import numpy as np
from scipy.stats import mannwhitneyu

import eccodes
from meters import ResourceMeter
from conflator import Conflator

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.grib_helpers import construct_message
from pproc.common.io import nan_to_missing
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.common.recovery import create_recovery, Recovery
from pproc.config.types import SigniConfig, SigniParamConfig
from pproc.config.targets import Target
from pproc.signi.clim import retrieve_clim


def signi(
    fc: np.ndarray,
    clim: np.ndarray,
    template: eccodes.GRIBMessage,
    clim_template: eccodes.GRIBMessage,
    target: Target,
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

    clim_keys = {key: clim_template.get(key) for key in []}
    grib_keys.update(clim_keys)
    message = construct_message(template, grib_keys)
    message.set_array("values", nan_to_missing(message, pvalue))
    target.write(message)


def signi_iteration(
    config: SigniConfig,
    param: SigniParamConfig,
    recovery: Recovery,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):

    with ResourceMeter(f"{param.name}, window {window_id}: Retrieve climatology"):
        steprange = accum.grib_keys()["stepRange"]
        clim_accum, clim_template = retrieve_clim(
            param.clim,
            config.sources,
            "clim",
            config.clim_total_fields,
            step=steprange,
        )
        clim = clim_accum.values
        assert clim is not None
        if config.use_clim_anomaly:
            clim_em_accum, _ = retrieve_clim(
                param.clim_em,
                config.sources,
                "clim_em",
                step=steprange,
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

    with ResourceMeter(f"{param.name}, window {window_id}: Compute significance"):
        fc = accum.values
        assert fc is not None
        signi(
            fc,
            clim,
            template,
            clim_template,
            config.outputs.signi.target,
            out_keys=accum.grib_keys(),
            epsilon=param.epsilon,
            epsilon_is_abs=param.epsilon_is_abs,
        )
        config.outputs.signi.target.flush()
    recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(app_name="pproc-significance", model=SigniConfig).load()
    cfg.print()
    recovery = create_recovery(cfg)

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            accum_manager = AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.signi.metadata,
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
                "fc",
            )
            signi_partial = functools.partial(signi_iteration, cfg, param, recovery)
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
                    executor.submit(signi_partial, metadata[0], window_id, accum)
            executor.wait()

    recovery.clean_file()


if __name__ == "__main__":
    sys.exit(main())
