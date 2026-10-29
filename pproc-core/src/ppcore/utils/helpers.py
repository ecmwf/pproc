# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np


def to_list(value: Any) -> list[Any]:
    if np.ndim(value) == 0:
        return [value]
    return list(value)
