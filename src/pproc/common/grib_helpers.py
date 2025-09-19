# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Dict
import numpy as np
import re


def construct_message(template_grib, metadata: dict):
    out_grib = template_grib.copy()
    key_values = metadata.copy()
    arr_grib_keys = {
        key: value for key, value in metadata.items() if np.ndim(value) > 0
    }
    for arr_key in arr_grib_keys.keys():
        key_values.pop(arr_key)

    template_edition = out_grib.get("edition")
    if key_values.get("edition", template_edition) == 2:
        key_values.setdefault("packingType", "grid_ccsds")
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

    for key, value in arr_grib_keys.items():
        out_grib.set_array(key, value)
    return out_grib


_TEMPLATE_RE = re.compile("^{([a-z_]*)}:?([a-z]*)$", re.I)
_TYPES = {
    "int": int,
    "str": str,
    "float": float,
}


def fill_template_values(metadata: dict, template_map: dict) -> dict:
    updates = {}
    for key, val in metadata.items():
        if isinstance(val, str):
            m = _TEMPLATE_RE.fullmatch(val)
            if m is None:
                continue
            value, tp = m.groups()
            if value not in template_map:
                continue

            updates[key] = (
                template_map[value] if len(tp) == 0 else _TYPES[tp](template_map[value])
            )
    metadata.update(updates)
    return metadata
