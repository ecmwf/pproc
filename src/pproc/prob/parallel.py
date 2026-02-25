# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from meters import ResourceMeter
from typing import Union, Optional
import numpy as np
import numexpr

import eccodes

from pproc.config.param import ParamConfig
from pproc.config.io import Output
from pproc.config.recovery import BaseRecovery
from pproc.common.accumulation import Accumulator
from pproc.common.io import write_grib
from pproc.prob.threshold import ThresholdConfig, SingleThreshold


def ensemble_probability(data: np.array, thconfig: ThresholdConfig) -> np.array:
    """Ensemble Probabilities:

    Computes the probability of a given parameter crossing a given threshold,
    by checking how many times it occurs across all ensembles.
    e.g. the chance of temperature being less than 0C

    """
    thresholds = thconfig.param_thresholds
    data = data.reshape((len(thresholds), -1) + data.shape[1:])
    is_nan = 0
    comp = 1
    for index, threshold in enumerate(thresholds):
        param_data = data[index]
        # Find all locations where np.nan appears as an ensemble value
        is_nan |= np.isnan(param_data).any(axis=0)

        # Read threshold configuration and compute probability
        if isinstance(threshold, SingleThreshold):
            comp &= numexpr.evaluate(
                f"data {threshold.comparison} {threshold.value}",
                local_dict={"data": param_data},
            )
        else:
            comp &= numexpr.evaluate(
                f"data {threshold.lower_comparison} {threshold.lower_value}",
                local_dict={"data": param_data},
            ) & numexpr.evaluate(
                f"data {threshold.upper_comparison} {threshold.upper_value}",
                local_dict={"data": param_data},
            )

    probability = np.where(comp, 100, 0).mean(axis=0)
    # Put in missing values
    probability = np.where(is_nan, np.nan, probability)
    return probability


def prob_iteration(
    param: ParamConfig,
    recovery: BaseRecovery,
    out_prob: Output,
    template: eccodes.GRIBMessage,
    window_id: str,
    accum: Accumulator,
    thresholds: list[ThresholdConfig],
    clim_metadata: Optional[dict] = None,
):
    with ResourceMeter(f"Window {window_id}, computing threshold probs"):

        ens = accum.values
        assert ens is not None

        for threshold in thresholds:
            window_probability = ensemble_probability(ens, threshold)

            print(
                f"Writing probability for input param {param.name} and output "
                + f"param {threshold.out_paramid} for step(s) {window_id}"
            )
            grib_set = out_prob.metadata.copy()
            grib_set.update(accum.grib_keys())
            grib_set.update(
                threshold.grib_keys(
                    grib_set.get("edition", template["edition"]), clim_metadata
                )
            )
            write_grib(out_prob.target, template, window_probability, grib_set)

        out_prob.target.flush()
        recovery.add_checkpoint(param=param.name, window=window_id)
