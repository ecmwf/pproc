from typing import Dict


def construct_message(template_grib, window_grib_headers: Dict):
    """
    Sets grib headers into template message using headers specified in
    config, from the threshold and climatology date period
    """
    out_grib = template_grib.copy()
    key_values = window_grib_headers.copy()
    set_missing = [
        key for key, value in window_grib_headers.items() if value == "MISSING"
    ]
    for missing_key in set_missing:
        key_values.pop(missing_key)

    template_edition = out_grib.get("edition")
    if key_values.get("edition", template_edition) != template_edition:
        # Set grib 1 and grib 2 keys separately as value check can fail when
        # grib 1 keys are removed in the switch to grib 2, or vice versa
        keys = list(key_values.keys())
        edition_index = keys.index("edition")
        out_grib.set(
            {key: key_values[key] for key in keys[:edition_index]},
            check_values=True,
        )
        out_grib.set(
            {key: key_values[key] for key in keys[edition_index:]},
            check_values=True,
        )
    else:
        out_grib.set(key_values, check_values=True)

    for missing_key in set_missing:
        out_grib.set_missing(missing_key)
    return out_grib
