# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Optional, Iterator
import copy
import numpy as np
import pandas as pd
import itertools

from pproc.common.utils import dict_product


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    return dict(map(lambda s: s.split("="), items))


def parse_var_strs(items):
    """
    Parse a list of comma-separated lists of key-value pairs and return a dictionary
    """
    return parse_vars(sum((s.split(",") for s in items if s), start=[]))


def _get(obj, attr, default=None):
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)


def _set(obj, attr, value):
    if isinstance(obj, dict):
        obj[attr] = value
    else:
        setattr(obj, attr, value)


def model_update(original: dict, update: Any) -> dict:
    for key, value in update.items():
        default = object()
        if isinstance(value, dict) and _get(original, key, default) != default:
            _set(original, key, model_update(_get(original, key), value))
        else:
            _set(original, key, value)
    return original


def validate_overrides(data: Any) -> Any:
    if isinstance(data, list):
        return parse_var_strs(data)
    return data
