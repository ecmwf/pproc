from typing import Any, Optional, Iterator
import copy
import numpy as np
import pandas as pd

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


def deep_update(original: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(original.get(key, None), dict):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original


def update_request(base: dict | list[dict], update: dict | list[dict], **kwargs):
    if isinstance(base, dict):
        base = [base]
    if isinstance(update, dict):
        update = [update]

    if len(update) == 0:
        return copy.deepcopy(base)
    if len(base) != len(update):
        broadcast_len = max(len(base), len(update))
        if len(base) == 0:
            return [copy.deepcopy(up) for up in update]
        if len(base) == 1:
            base = [copy.deepcopy(base[0]) for _ in range(broadcast_len)]
        if len(update) == 1:
            update = update * broadcast_len
    return [deep_update(breq, {**ureq, **kwargs}) for breq, ureq in zip(base, update)]


def expand(
    requests: dict | list[dict],
    dim: Optional[str | list[str]] = None,
    exclude: list[str] = [],
) -> Iterator[dict]:
    if isinstance(requests, dict):
        requests = [requests]

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
            coords = request.pop(d, [])
            if coords is None:
                continue
            if np.ndim(coords) == 0:
                coords = [coords]
            expansion[d] = coords

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
            if isinstance(val, str) or not np.isnan(val):
                req[dim] = sorted(list({x[dim] for x in cond_reqs}))
        yield req


def extract_mars(keys: dict) -> dict:
    if "paramId" in keys:
        keys["param"] = keys.pop("paramId")
    mars_namespace = [
        "class",
        "stream",
        "expver",
        "date",
        "time",
        "param",
        "levtype",
        "levelist",
        "type",
        "number",
        "step",
        "hdate",
        "domain",
    ]
    return {k: v for k, v in keys.items() if k in mars_namespace}
