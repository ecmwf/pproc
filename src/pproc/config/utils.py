from typing import Any


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


def expand(request: dict, dim: str):
    coords = request.pop(dim, None)
    if coords is None:
        return [request]
    if not isinstance(coords, list):
        coords = [coords]
    return [{**request, dim: coord} for coord in coords]


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
        "domain"
    ]
    return {k: v for k, v in keys.items() if k in mars_namespace}
