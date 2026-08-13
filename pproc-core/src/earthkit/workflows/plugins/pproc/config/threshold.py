# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import model_validator
from typing import Any, Optional, Literal

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel

Comparisons = Literal["<", ">", "<=", ">="]


class Threshold(PProcBaseModel):
    select: Optional[dict] = None
    lower_scale_factor: int
    lower_comparison: Comparisons
    lower_value: float
    upper_scale_factor: Optional[int] = None
    upper_comparison: Optional[Comparisons] = None
    upper_value: Optional[float] = None

    @model_validator(mode="before")
    def validate_threshold(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "comparison" in data:
                data["lower_comparison"] = data.pop("comparison")
                for key in ["value", "scale_factor"]:
                    if key in data:
                        data[f"lower_{key}"] = data.pop(key)
            if any(k in data for k in ["upper_comparison", "upper_value"]):
                if not all(k in data for k in ["upper_comparison", "upper_value"]):
                    raise ValueError(
                        "Both upper_comparison and upper_value must be provided together"
                    )
                if "lower_comparison" not in data or "lower_value" not in data:
                    raise ValueError(
                        "Upper threshold should not be used without lower threshold. If only a single threshold is needed, use lower_comparison and lower_value."
                    )

            # Derive scale factors, if not specified
            for value in ["lower_value", "upper_value"]:
                upper_or_lower = value.split("_")[0]
                if value in data and f"{upper_or_lower}_scale_factor" not in data:
                    scale_factor = 0
                    threshold = data[value]
                    while not threshold.is_integer():
                        scale_factor += 1
                        threshold = threshold * 10**scale_factor
                    data[f"{upper_or_lower}_scale_factor"] = scale_factor
        return data
