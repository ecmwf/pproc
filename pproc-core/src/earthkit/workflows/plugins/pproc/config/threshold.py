from pydantic import BaseModel, model_validator
from typing import Any, Optional, Literal

Comparisons = Literal["<", ">", "<=", ">="]


class Threshold(BaseModel):
    select: Optional[dict] = None
    lower_scale_factor: int = 0
    lower_comparison: Comparisons
    lower_value: float
    upper_scale_factor: int = 0
    upper_comparison: Optional[Comparisons] = None
    upper_value: Optional[float] = None

    @model_validator(mode="before")
    def validate_threshold(cls, data: Any) -> Any:
        if "comparison" in data and "value" in data:
            data["lower_comparison"] = data.pop("comparison")
            data["lower_value"] = data.pop("value")
            data["lower_scale_factor"] = data.get("scale_factor", 0)
        if any(k in data for k in ["upper_comparison", "upper_value"]) and not all(
            k in data for k in ["upper_comparison", "upper_value"]
        ):
            raise ValueError(
                "Both upper_comparison and upper_value must be provided together"
            )
        return data
