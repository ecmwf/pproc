from typing import Literal, Any
from pydantic import Field, model_validator

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from earthkit.workflows.plugins.pproc.config.threshold import Threshold


class Statistics(PProcBaseModel):
    metadata: dict = Field(default_factory=dict)


class Quantiles(Statistics):
    operation: Literal["quantiles"] = "quantiles"
    quantiles: int | list[float] | str = 100


class EFI(Statistics):
    operation: Literal["efi"] = "efi"
    eps: float = -1.0


class SOT(Statistics):
    operation: Literal["sot"] = "sot"
    eps: float = -1.0
    sot: list[int]


class Mean(Statistics):
    operation: Literal["mean"] = "mean"


class StandardDeviation(Statistics):
    operation: Literal["std"] = "std"


class ThresholdProbability(Statistics):
    operation: Literal["threshold_probability"] = "threshold_probability"
    thresholds: list[Threshold] = Field(default_factory=list)

    @model_validator(mode="before")
    def validate_thresholds(cls, data: Any) -> Any:
        if isinstance(data, dict) and "threshold" in data:
            data.setdefault("thresholds", []).append(data.pop("threshold"))
        return data
