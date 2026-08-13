# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def parse_vars(items: list[str]) -> dict:
    """
    Parse a series of key-value pairs and return a dictionary
    """
    return dict(map(lambda s: s.split("="), items))


def parse_var_strs(items: list[str]) -> dict:
    """
    Parse a list of comma-separated lists of key-value pairs and return a dictionary
    """
    return parse_vars(sum((s.split(",") for s in items if s), start=[]))


def validate_overrides(data: Any) -> Any:
    if isinstance(data, list):
        return parse_var_strs(data)
    return data
