# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import re

_TEMPLATE_RE = re.compile("^{([a-z_]*)}:?([a-z]*)$", re.I)
_TYPES = {
    "int": int,
    "str": str,
    "float": float,
}


def fill_template_value(val: str, template_map: dict):
    m = _TEMPLATE_RE.fullmatch(val)
    if m is None:
        return val
    value, tp = m.groups()
    if value not in template_map:
        return val

    return template_map[value] if len(tp) == 0 else _TYPES[tp](template_map[value])


def fill_template_values(metadata: dict, template_map: dict) -> dict:
    metadata = metadata.copy()
    for key, val in metadata.items():
        if isinstance(val, str):
            metadata[key] = fill_template_value(val, template_map)
    return metadata
