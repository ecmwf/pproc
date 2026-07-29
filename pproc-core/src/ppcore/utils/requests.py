# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools
from typing import Iterator, Optional, Iterable, Any

import numpy as np
import pandas as pd
from qubed import Qube

from ppcore.utils.dicts import deep_update
from ppcore.utils.dicts import dict_product
from ppcore.utils.helpers import to_list


VALUE_TYPES = {
    "param": str,
    "paramId": int,
    "levelist": int,
    "step": int,
    "fcmonth": int,
    "number": int,
    "dataDate": int,
    "date": str,
}


def validate_request(request: dict) -> dict:
    """
    Format request into desired format for schema and config generation
    """
    out = copy.deepcopy(request)
    # Map types
    for key, value in request.items():
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if key in VALUE_TYPES:
            value_type = VALUE_TYPES[key]
            try:
                value = (
                    value_type(value)
                    if np.ndim(value) == 0
                    else list(map(value_type, value))
                )
            except ValueError:
                pass
        out[key] = value
    # Format time
    if times := out.get("time", None):
        if isinstance(times, (int, str)):
            times = [times]
        out_times = []
        for ti in times:
            out_times.append(f"{int(ti):02d}".ljust(4, "0"))
        out["time"] = out_times if len(out_times) > 1 else out_times[0]
    return out


def expand(
    requests: dict[str, Any] | Iterable[dict[str, Any]],
    dim: Optional[str | list[str]] = None,
    exclude: list[str] = [],
) -> Iterator[dict]:
    requests: Iterable[dict[str, Any]] = (
        [requests] if isinstance(requests, dict) else requests
    )  # type: ignore

    for request in requests:
        request = copy.deepcopy(request)
        # Expand all if no dimension is specified
        if dim is None:
            dims = [x for x in request.keys() if x not in exclude]
        elif isinstance(dim, str):
            dims = [dim]
        else:
            dims = dim

        expansion = {}
        for d in dims:
            coords = request.pop(d, None)
            if coords is None:
                continue
            expansion[d] = to_list(coords)

        for vals in dict_product(expansion):
            yield {**request, **vals}


def squeeze(reqs: list[dict], dims: list[str]) -> Iterator[dict]:
    df = pd.DataFrame(reqs)
    drop_dims = df.drop(dims, axis=1, errors="ignore").drop_duplicates()
    for _, row in drop_dims.iterrows():
        req = row.dropna().to_dict()
        condition = np.logical_and.reduce([df[k] == v for k, v in req.items()])
        cond_reqs = df.loc[condition].to_dict("records")
        for dim in dims:
            val = cond_reqs[0].get(dim, np.nan)
            if val is None:
                continue
            if isinstance(val, str) or not np.isnan(val):
                req[dim] = sorted(list({x[dim] for x in cond_reqs}))
        yield req


def datacubes(reqs: list[dict]) -> Iterator[dict]:
    qube = Qube.empty()
    for req in reqs:
        qube = qube | Qube.from_datacube(req)
    for dataqube in qube.datacubes():
        yield {key: val if len(val) > 1 else val[0] for key, val in dataqube.items()}


def update_request(
    base: dict | list[dict], update: dict | list[dict], method: str = "map", **kwargs
):
    if isinstance(base, dict):
        base = [base]
    if isinstance(update, dict):
        update = [update]

    if len(update) == 0:
        return copy.deepcopy(base)
    if len(base) == 0:
        return copy.deepcopy(update)
    if method == "map":
        if len(base) == len(update):
            combinations = zip(base, update)
        else:
            assert len(base) == 1 or len(update) == 1
            combinations = itertools.product(base, update)
    elif method == "product":
        combinations = itertools.product(base, update)
    else:
        raise ValueError(
            f"Unknown method for combining requests: {method}. Supported methods are 'map' and 'product'"
        )
    new_requests = [
        deep_update(copy.deepcopy(breq), {**ureq, **kwargs})
        for breq, ureq in combinations
    ]
    # Remove duplicates
    deduplicated = []
    for inp in new_requests:
        if inp not in deduplicated:
            deduplicated.append(inp)
    return deduplicated
