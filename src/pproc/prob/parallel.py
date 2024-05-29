import eccodes
from meters import ResourceMeter
from typing import Dict

from pproc import common
from pproc.common.grib_helpers import construct_message
from pproc.prob.math import ensemble_probability


def threshold_grib_headers(
    edition: int, threshold: Dict, climatology_headers: Dict = {}
) -> Dict:
    """
    Creates dictionary of threshold related grib headers
    """
    threshold_dict = {"paramId": threshold["out_paramid"]}
    scale_factor = threshold.get("localDecimalScaleFactor", 0)
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
    elif edition == 2 and comparison == "<" and threshold_value >= 0:
        grib_keys = {
            "scaleFactorOfUpperLimit": scale_factor,
            "scaledValueOfUpperLimit": threshold_value,
            "probabilityType": 4,
            "scaleFactorOfLowerLimit": "MISSING",
            "scaledValueOfLowerLimit": "MISSING",
            **climatology_headers,
        }
    elif edition == 2 and comparison == "<" and threshold_value < 0:
        grib_keys = {
            "scaleFactorOfLowerLimit": scale_factor,
            "scaledValueOfLowerLimit": threshold_value,
            "probabilityType": 0,
            "scaleFactorOfUpperLimit": "MISSING",
            "scaledValueOfUpperLimit": "MISSING",
            **climatology_headers,
        }
    elif edition == 2 and comparison == ">":
        grib_keys = {
            "scaleFactorOfLowerLimit": scale_factor,
            "scaledValueOfLowerLimit": threshold_value,
            "probabilityType": 3,
            "scaleFactorOfUpperLimit": "MISSING",
            "scaledValueOfUpperLimit": "MISSING",
            **climatology_headers,
        }
    else:
        raise ValueError(
            f"Unsupported threshold comparison {comparison} for grib edition {edition}"
        )

    threshold_dict.update(grib_keys)
    threshold_dict.update(threshold.get("grib_set", {}))
    return threshold_dict


def prob_iteration(
    param,
    recovery,
    out_ensemble,
    out_prob,
    template_filename,
    window_id,
    window,
    thresholds,
    climatology_headers={},
):

    with ResourceMeter(f"Window {window.name}, computing threshold probs"):
        message_template = (
            template_filename
            if isinstance(template_filename, eccodes.highlevel.message.GRIBMessage)
            else common.io.read_template(template_filename)
        )

        if not isinstance(out_ensemble, common.io.NullTarget):
            for index in range(len(window.step_values)):
                data_type, number = param.type_and_number(index)
                print(
                    f"Writing window values for param {param.name} and output "
                    + f"type {data_type}, number {number} for step(s) {window.name}"
                )
                template = construct_message(message_template, window.grib_header())
                template.set({"type": data_type, "number": number})
                common.write_grib(out_ensemble, template, window.step_values[index])

        for threshold in thresholds:
            window_probability = ensemble_probability(window.step_values, threshold)

            print(
                f"Writing probability for input param {param.name} and output "
                + f"param {threshold['out_paramid']} for step(s) {window.name}"
            )
            grib_set = window.grib_header()
            grib_set.update(
                threshold_grib_headers(
                    grib_set.get("edition", 1), threshold, climatology_headers
                )
            )
            common.write_grib(
                out_prob,
                construct_message(
                    message_template,
                    grib_set,
                ),
                window_probability,
            )

        out_ensemble.flush()
        out_prob.flush()
        recovery.add_checkpoint(param.name, window_id)
