# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import signal
import sys
from fractions import Fraction
from typing import Any, Dict, Optional, Set

import eccodes
import numpy as np
from meters import ResourceMeter
from conflator import Conflator
from earthkit.meteo.stats import iter_quantiles

from pproc.common.accumulation import Accumulator
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.io import write_grib
from pproc.common.parallel import (
    create_executor,
    parallel_data_retrieval,
    sigterm_handler,
)
from pproc.common.param_requester import ParamRequester
from pproc.config.types import QuantileProbConfig, QuantileProbParamConfig
from pproc.config.targets import Target
from pproc.signi.clim import retrieve_clim


def compute_boundaries(
    denominators: list[int],
    clim: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    out_keys: Optional[Dict[str, Any]] = None,
) -> Dict[int, np.ndarray]:
    """Compute climatological quantile boundaries

    Parameters
    ----------
    denominators: list of int
        List of quantile denominators (3 for terciles, 5 for quintiles, etc.)
    clim: numpy array (..., npoints)
        Climatology data (all dimensions but the last are squashed together)
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    out_keys: dict, optional
        Extra GRIB keys to set on the output

    Returns
    -------
    dict[int, numpy array]
        Quantile boundaries for each denominator. ``bounds[d]`` has shape ``(d - 1, npoints)``
    """
    clim = clim.reshape((-1, clim.shape[-1]))
    clim -= clim.mean(axis=0)

    all_qs: Set[Fraction] = set()
    for d in denominators:
        all_qs.update(Fraction(n, d) for n in range(1, d))
    qs = sorted(all_qs)
    bounds_f = {q: quantile for q, quantile in zip(qs, iter_quantiles(clim, qs))}
    bounds: Dict[int, np.ndarray] = {}
    for d in denominators:
        bounds[d] = np.stack([bounds_f[Fraction(n, d)] for n in range(1, d)])

    if out_keys is None:
        out_keys = {}
    edition = out_keys.get("edition", template["edition"])
    for d, qbounds in bounds.items():
        for n, bound in enumerate(qbounds, start=1):
            metadata = out_keys.copy()
            if edition == 1:
                metadata["numberOfForecastsInEnsemble"] = d
                metadata["perturbationNumber"] = n
            elif edition == 2:
                metadata["totalNumberOfQuantiles"] = d
                metadata["quantileValue"] = n
            else:
                raise ValueError(f"Unsupported GRIB edition {edition}")
            write_grib(target, template, bound, metadata)

    return bounds


def compute_probabilities(
    fc: np.ndarray,
    bounds: Dict[int, np.ndarray],
    template: eccodes.GRIBMessage,
    target: Target,
    out_keys: Optional[Dict[str, Any]],
    eps: float = 1e-10,
):
    """Compute quantiles probabilities

    Parameters
    ----------
    fc: numpy array (..., npoints)
        Forecast data (all dimensions but the last are squashed together)
    bounds: dict[int, numpy array]
        Quantile boundaries for each denominator. ``bounds[d]`` must have shape ``(d - 1, npoints)``
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    eps: float (default 1e-10)
        If the first and last boundary values are closer than this value,
        consider the distribution degenerate and output uniform probabilities
        across all quantiles
    """
    if out_keys is None:
        out_keys = {}

    edition = out_keys.get("edition", template["edition"])

    # TODO: add safeguard: if quantile boundaries are too close to each other
    # (e.g. 10**-10), output an uniform distribution

    fc = fc.reshape((-1, fc.shape[-1]))
    for d, qbounds in bounds.items():
        assert (
            qbounds.shape[-1] == fc.shape[-1]
        ), "Forecast and climatology are on different grids"

        degenerate = np.abs(qbounds[-1, :] - qbounds[0, :]) < eps
        missing = np.any(np.isnan(qbounds), axis=0)
        cdf = np.zeros_like(qbounds)
        for member in fc:
            missing |= np.isnan(member)
            cdf[qbounds > member] += 100.0 / fc.shape[0]

        for n in range(d):
            low = 0.0 if n == 0 else cdf[n - 1, :]
            high = 100.0 if n == d - 1 else cdf[n, :]
            prob = high - low
            prob[degenerate] = 1 / d
            prob[missing] = np.nan

            metadata = out_keys.copy()
            if edition == 1:
                metadata["numberOfForecastsInEnsemble"] = d
                metadata["perturbationNumber"] = n + 1
            elif edition == 2:
                metadata["probabilityType"] = 10
                # When using entry 10, the lower limit is used to encode the
                # quantile q (must be an integer between 0 and Q) while the
                # upper limit is used to encode the total number of quantiles Q
                # (WMO manual on codes, code table 4.9 "Probability type")
                metadata["lowerLimit"] = n + 1
                metadata["upperLimit"] = d
            else:
                raise ValueError(f"Unsupported GRIB edition {edition}")
            write_grib(target, template, prob, metadata)


def qprob_iteration(
    config: QuantileProbConfig,
    param: QuantileProbParamConfig,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
):
    with ResourceMeter(f"{param.name}, window {window_id}: Retrieve climatology"):
        steprange = accum.grib_keys()["stepRange"]
        clim_accum, _ = retrieve_clim(
            param.clim,
            config.inputs,
            "clim",
            param.clim.total_fields,
            step=steprange,
        )
        clim = clim_accum.values
        assert clim is not None

    with ResourceMeter(
        f"{param.name}, window {window_id}: Compute quantile boundaries"
    ):
        bounds = compute_boundaries(
            param.denominators,
            clim,
            template,
            config.outputs.bound.target,
            accum.grib_keys() | config.outputs.bound.metadata,
        )
        config.outputs.bound.target.flush()

    with ResourceMeter(f"{param.name}, window {window_id}: Compute probabilities"):
        fc = accum.values
        assert fc is not None
        compute_probabilities(
            fc,
            bounds,
            template,
            config.outputs.prob.target,
            accum.grib_keys() | config.outputs.prob.metadata,
        )
        config.outputs.prob.target.flush()
    config.recovery.add_checkpoint(param=param.name, window=window_id)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGTERM, sigterm_handler)

    cfg = Conflator(
        app_name="pproc-quantile-probabilities", model=QuantileProbConfig
    ).load()
    cfg.initialise()
    cfg.print()

    with create_executor(cfg.parallelisation) as executor:
        for param in cfg.parameters:
            print(f"Processing {param.name}")
            accum_manager = AccumulationManager.create(
                param.accumulations,
                {
                    **cfg.outputs.default.metadata,
                    **param.metadata,
                },
            )
            checkpointed_windows = [
                x["window"] for x in cfg.recovery.computed(param=param.name)
            ]
            accum_manager.delete(checkpointed_windows)

            requester = ParamRequester(param, cfg.inputs, param.total_fields, "fc")
            prob_partial = functools.partial(qprob_iteration, cfg, param)
            for keys, retrieved_data in parallel_data_retrieval(
                cfg.parallelisation.n_par_read,
                accum_manager.dims,
                [requester],
            ):
                ids = ", ".join(f"{k}={v}" for k, v in keys.items())
                with ResourceMeter(f"{param.name}, {ids}: Compute accumulation"):
                    metadata, data = retrieved_data[0]
                    completed_windows = accum_manager.feed(keys, data)
                    del data

                for window_id, accum in completed_windows:
                    executor.submit(
                        prob_partial,
                        metadata[0],
                        window_id,
                        accum,
                    )
            executor.wait()

    cfg.clean()


if __name__ == "__main__":
    sys.exit(main())
