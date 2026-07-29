# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import model_validator
from typing import Any, Optional, Literal

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel

Comparisons = Literal["<", ">", "<=", ">="]


class Threshold(PProcBaseModel):
    select: Optional[dict] = None
    lower_scale_factor: int = 0
    lower_comparison: Comparisons
    lower_value: float
    upper_scale_factor: int = 0
    upper_comparison: Optional[Comparisons] = None
    upper_value: Optional[float] = None

    @model_validator(mode="before")
    def validate_threshold(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "comparison" in data:
                data["lower_comparison"] = data.pop("comparison")
                data["lower_value"] = data.pop("value")
                data["lower_scale_factor"] = data.pop("scale_factor", 0)
            if any(k in data for k in ["upper_comparison", "upper_value"]) and not all(
                k in data for k in ["upper_comparison", "upper_value"]
            ):
                raise ValueError(
                    "Both upper_comparison and upper_value must be provided together"
                )
            # Derive scale factors for custom thresholds
            for value in ["lower_value", "upper_value"]:
                upper_or_lower = value.split("_")[0]
                if (
                    isinstance(data.get(value), float)
                    and f"{upper_or_lower}_scale_factor" not in data
                ):
                    scale_factor = 0
                    while abs(data[value] * 10**scale_factor) < 1:
                        scale_factor += 1
                    data[f"{upper_or_lower}_scale_factor"] = scale_factor
                    data[value] = round(data[value] * 10**scale_factor)
        return data
