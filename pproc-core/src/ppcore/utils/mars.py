# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from datetime import datetime, date
from math import ceil

METADATA_KEYS = {"param": "paramId", "date": "dataDate"}


def _val_to_mars(val):
    if isinstance(val, bytes):
        return val
    elif isinstance(val, (list, tuple)):
        return b"/".join(_val_to_mars(v) for v in val)
    elif isinstance(val, str):
        pass
    elif isinstance(val, int):
        val = str(val)
    elif isinstance(val, (datetime, date)):
        val = val.strftime("%Y%m%d")
    elif isinstance(val, range):
        first = val.start
        last = val.start + (ceil((val.stop - val.start) / val.step) - 1) * val.step
        step = val.step
        val = f"{first}/to/{last}"
        if step != 1:
            val += f"/by/{step}"
    elif isinstance(val, float) and val.is_integer():
        val = str(int(val))
    else:
        raise TypeError(f"Cannot convert {type(val)} to MARS request")
    return val.encode("utf-8")


def to_mars(verb: bytes, req: dict) -> bytes:
    def _gen_req():
        yield verb
        for key, val in req.items():
            key = key.encode("utf-8")
            val = _val_to_mars(val)
            yield key + b"=" + val

    return b",".join(_gen_req())


def extract_mars(keys: dict, additional: Optional[list[str]] = None) -> dict:
    additional = additional or []
    for key, metadata_key in METADATA_KEYS.items():
        if metadata_key in keys:
            keys[key] = keys.pop(metadata_key)
    mars_namespace = [
        "class",
        "type",
        "stream",
        "expver",
        "model",
        "levtype",
        "levelist",
        "param",
        "date",
        "year",
        "month",
        "hdate",
        "fcmonth",
        "fcperiod",
        "time",
        "step",
        "number",
        "domain",
        "quantile",
        "method",
        "origin",
        "system",
    ]
    return {k: v for k, v in keys.items() if (k in mars_namespace) or (k in additional)}
