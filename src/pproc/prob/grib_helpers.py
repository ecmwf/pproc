from typing import Dict


def threshold_grib_headers(threshold) -> Dict:
    """
    Creates dictionary of threshold related grib headers
    """
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


def construct_message(
    template_grib, window_grib_headers, threshold=None, climatology_headers: Dict = None
):
    """
    Sets grib headers into template message using headers specified in
    config, from the threshold and climatology date period
    """
    # Copy an input GRIB message and modify headers for writing probability
    # field
    out_grib = template_grib.copy()
    key_values = window_grib_headers.copy()
    set_missing = [
        key for key, value in window_grib_headers.items() if value == "MISSING"
    ]
    for missing_key in set_missing:
        key_values.pop(missing_key)

    # Set grib 1 and grib 2 keys separately as value check can fail when
    # grib 1 keys are removed in the switch to grib 2
    if key_values.get("edition", 1) == 2:
        if threshold:
            key_values.update({"paramId": threshold["out_paramid"]})
        if climatology_headers:
            key_values.update(climatology_headers)
        keys = list(key_values.keys())
        grib2_start_index = keys.index("edition")
        out_grib.set(
            {key: key_values[key] for key in keys[:grib2_start_index]},
            check_values=True,
        )
        out_grib.set(
            {key: key_values[key] for key in keys[grib2_start_index:]},
            check_values=True,
        )
    else:
        if threshold:
            key_values.update(threshold_grib_headers(threshold))
        out_grib.set(key_values, check_values=True)

    for missing_key in set_missing:
        out_grib.set_missing(missing_key)
    return out_grib
