import copy
from typing import Any
import numpy as np

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
    if time := out.get("time", None):
        if isinstance(time, list):
            raise ValueError("Only single value of time supported per request")
        if isinstance(time, int):
            time = f"{time:02d}"
        out["time"] = time.ljust(4, "0")
    return out
