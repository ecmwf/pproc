# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any


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
