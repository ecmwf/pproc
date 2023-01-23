from typing import Dict


def threshold_grib_headers(threshold) -> Dict:
    threshold_dict = {"paramId": threshold["out_paramid"]}
    threshold_value = threshold["value"]
    if "localDecimalScaleFactor" in threshold:
        scale_factor = threshold["localDecimalScaleFactor"]
        threshold_dict["localDecimalScaleFactor"] = scale_factor
        threshold_value = round(threshold["value"] * 10**scale_factor, 0)

    comparison = threshold["comparison"]
    if "<" in comparison:
        threshold_dict.update(
            {"thresholdIndicator": 2, "upperThreshold": threshold_value}
        )
    elif ">" in comparison:
        threshold_dict.update(
            {"thresholdIndicator": 1, "lowerThreshold": threshold_value}
        )
    return threshold_dict


def construct_message(template_grib, window_grib_headers, threshold):
    # Copy an input GRIB message and modify headers for writing probability
    # field
    out_grib = template_grib.copy()
    key_values = {
        "type": "ep",
        "localDefinitionNumber": 5,
        "bitsPerValue": 8,  # Set equal to accuracy used in mars compute
    }
    key_values.update(window_grib_headers)
    key_values.update(threshold_grib_headers(threshold))

    out_grib.set(key_values, check_values=True)
    return out_grib
