from typing import Any

import numpy as np


def to_list(value: Any) -> list[Any]:
    if np.ndim(value) == 0:
        return [value]
    return list(value)
