# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import re

_TEMPLATE_RE = re.compile("^{([a-z_]*)}:?([a-z]*)$", re.I)
_EXPR_TEMPLATE_RE = re.compile(r"^\{([^{}]+)\}$")
_TYPES = {
    "int": int,
    "str": str,
    "float": float,
}


def fill_template_value(val: str, template_map: dict):
    m = _TEMPLATE_RE.fullmatch(val)
    if m is not None:
        value, tp = m.groups()
        if value not in template_map:
            return val
        return template_map[value] if len(tp) == 0 else _TYPES[tp](template_map[value])

    expr_match = _EXPR_TEMPLATE_RE.fullmatch(val)
    in_map = any(k in val for k in template_map.keys())
    if expr_match is None or not in_map:
        return val
    return eval(
        expr_match.group(1).strip(),
        {"__builtins__": {}},
        {**_TYPES, **template_map},
    )


def fill_template_values(metadata: dict, template_map: dict) -> dict:
    metadata = metadata.copy()
    for key, val in metadata.items():
        if isinstance(val, str):
            metadata[key] = fill_template_value(val, template_map)
        if isinstance(val, list):
            metadata[key] = [
                fill_template_value(v, template_map) if isinstance(v, str) else v
                for v in val
            ]
    return metadata
