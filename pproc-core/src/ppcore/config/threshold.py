from pydantic import BaseModel, Field, model_validator
from typing import Any, Optional, Literal, List

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

    def grib_keys(self, edition: int) -> dict:
        if self.upper_comparison is None:
            threshold_value = round(self.lower_value * 10**self.lower_scale_factor, 0)
            comparison = self.lower_comparison.strip("=")
            if edition == 1 and comparison == "<":
                grib_keys = {
                    "localDefinitionNumber": 5,
                    "localDecimalScaleFactor": self.lower_scale_factor,
                    "thresholdIndicator": 2,
                    "upperThreshold": threshold_value,
                }
            elif edition == 1 and comparison == ">":
                grib_keys = {
                    "localDefinitionNumber": 5,
                    "localDecimalScaleFactor": self.lower_scale_factor,
                    "thresholdIndicator": 1,
                    "lowerThreshold": threshold_value,
                }
            elif edition == 2:
                # GRIB 2 has probability types above/below upper/lower limits (see Code Table 4.9)
                # Default is to use limit_type=lower probability types
                # where the threshold value can correspond to either limit. The default limit type
                # is upper for "<" and lower for ">", consistent with the GRIB 1 to GRIB 2 conversion
                # assumption.
                limit_type = "lower"
                prob_types = {
                    "<": {"upper": 4, "lower": 0},
                    ">": {"upper": 1, "lower": 3},
                }
                probability_type = prob_types[comparison][limit_type]
                missing = "Upper" if limit_type == "lower" else "Lower"
                grib_keys = {
                    f"scaleFactorOf{limit_type.capitalize()}Limit": self.lower_scale_factor,
                    f"scaledValueOf{limit_type.capitalize()}Limit": threshold_value,
                    "probabilityType": probability_type,
                    f"scaleFactorOf{missing}Limit": "MISSING",
                    f"scaledValueOf{missing}Limit": "MISSING",
                }
            else:
                raise ValueError(
                    f"Unsupported threshold comparison {comparison} for grib edition {edition}"
                )
            return grib_keys
        if edition != 2:
            raise ValueError("Threshold ranges are only supported for GRIB edition 2")
        return {
            "scaleFactorOfLowerLimit": self.lower_scale_factor,
            "scaledValueOfLowerLimit": round(
                self.lower_value * 10**self.lower_scale_factor, 0
            ),
            "probabilityType": 2,
            "scaleFactorOfUpperLimit": self.upper_scale_factor,
            "scaledValueOfUpperLimit": round(
                self.upper_value * 10**self.upper_scale_factor, 0
            ),
        }


class ThresholdConfig(BaseModel):
    param_thresholds: List[Threshold] = Field(default_factory=list)
    metadata: dict = {}

    @model_validator(mode="before")
    def validate_thresholds(cls, data: Any) -> Any:
        if isinstance(data, dict) and "param_thresholds" not in data:
            data["param_thresholds"] = [
                {k: data[k] for k in data if k not in cls.model_fields}
            ]
        return data

    def grib_keys(self, edition: int, clim_metadata: Optional[dict] = None) -> dict:
        """
        Creates dictionary of threshold related grib headers
        """
        threshold_dict = {}
        if "paramId" in self.metadata:
            threshold_dict["paramId"] = self.metadata["paramId"]
        threshold_dict.update(self.param_thresholds[0].grib_keys(edition))
        if edition == 2 and clim_metadata:
            threshold_dict.update(clim_metadata)
        threshold_dict.update(self.metadata)
        return threshold_dict
