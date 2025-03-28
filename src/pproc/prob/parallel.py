import eccodes
from meters import ResourceMeter
from typing import Dict
import numpy as np
import numexpr

from pproc import common
from pproc.config.param import ParamConfig
from pproc.config.targets import Target
from pproc.common.recovery import BaseRecovery
from pproc.common.accumulation import Accumulator
from pproc.common.grib_helpers import construct_message


def threshold_grib_headers(
    edition: int, threshold: Dict, climatology_headers: Dict = {}
) -> Dict:
    """
    Creates dictionary of threshold related grib headers
    """
    threshold_dict = {"paramId": threshold["out_paramid"]}
    scale_factor = threshold.get("local_scale_factor", 0)
    threshold_value = round(threshold["value"] * 10**scale_factor, 0)
    comparison = threshold["comparison"].strip("=")
    if edition == 1 and comparison == "<":
        grib_keys = {
            "localDecimalScaleFactor": scale_factor,
            "thresholdIndicator": 2,
            "upperThreshold": threshold_value,
        }
    elif edition == 1 and comparison == ">":
        grib_keys = {
            "localDecimalScaleFactor": scale_factor,
            "thresholdIndicator": 1,
            "lowerThreshold": threshold_value,
        }
    elif edition == 2:
        # GRIB 2 has probability types above/below upper/lower limits (see Code Table 4.9)
        # where the threshold value can correspond to either limit. The default limit type
        # is upper for "<" and lower for ">", consistent with the GRIB 1 to GRIB 2 conversion
        # assumption.
        prob_types = {"<": {"upper": 4, "lower": 0}, ">": {"upper": 1, "lower": 3}}
        if comparison == "<":
            limit_type = threshold.get("limit_type", "upper")
            probability_type = prob_types[comparison][limit_type]
        elif comparison == ">":
            limit_type = threshold.get("limit_type", "lower")
            probability_type = prob_types[comparison][limit_type]
        missing = "Upper" if limit_type == "lower" else "Lower"
        grib_keys = {
            f"scaleFactorOf{limit_type.capitalize()}Limit": scale_factor,
            f"scaledValueOf{limit_type.capitalize()}Limit": threshold_value,
            "probabilityType": probability_type,
            f"scaleFactorOf{missing}Limit": "MISSING",
            f"scaledValueOf{missing}Limit": "MISSING",
            **climatology_headers,
        }
    else:
        raise ValueError(
            f"Unsupported threshold comparison {comparison} for grib edition {edition}"
        )

    threshold_dict.update(grib_keys)
    threshold_dict.update(threshold.get("metadata", {}))
    return threshold_dict


def ensemble_probability(data: np.array, threshold: Dict) -> np.array:
    """Ensemble Probabilities:

    Computes the probability of a given parameter crossing a given threshold,
    by checking how many times it occurs across all ensembles.
    e.g. the chance of temperature being less than 0C

    """

    # Find all locations where np.nan appears as an ensemble value
    is_nan = np.isnan(data).any(axis=0)

    # Read threshold configuration and compute probability
    comparison = threshold["comparison"]
    comp = numexpr.evaluate(
        "data " + comparison + str(threshold["value"]), local_dict={"data": data}
    )
    probability = np.where(comp, 100, 0).mean(axis=0)

    # Put in missing values
    probability = np.where(is_nan, np.nan, probability)

    return probability


def prob_iteration(
    param: ParamConfig,
    recovery: BaseRecovery,
    out_prob: Target,
    template_filename: str,
    window_id: str,
    accum: Accumulator,
    thresholds: list,
    climatology_headers={},
):
    with ResourceMeter(f"Window {window_id}, computing threshold probs"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        ens = accum.values
        assert ens is not None

        for threshold in thresholds:
            window_probability = ensemble_probability(ens, threshold)

            print(
                f"Writing probability for input param {param.name} and output "
                + f"param {threshold['out_paramid']} for step(s) {window_id}"
            )
            grib_set = accum.grib_keys()
            grib_set.update(
                threshold_grib_headers(
                    grib_set.get("edition", 1), threshold, climatology_headers
                )
            )
            common.io.write_grib(
                out_prob,
                construct_message(
                    message_template,
                    grib_set,
                ),
                window_probability,
            )

        out_prob.flush()
        recovery.add_checkpoint(param=param.name, window=window_id)
