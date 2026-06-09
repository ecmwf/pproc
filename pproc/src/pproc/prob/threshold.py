from pydantic import BaseModel, Field, Discriminator, Tag, model_validator
from typing import Union, Any, Optional, Literal, Annotated, List


class SingleThreshold(BaseModel):
    type_: Literal["single"] = Field("single", alias="type")
    scale_factor: int = 0
    comparison: str
    value: float
    limit_type: str = "lower"

    def grib_keys(self, edition: int) -> dict:
        threshold_value = round(self.value * 10**self.scale_factor, 0)
        comparison = self.comparison.strip("=")
        if edition == 1 and comparison == "<":
            grib_keys = {
                "localDefinitionNumber": 5,
                "localDecimalScaleFactor": self.scale_factor,
                "thresholdIndicator": 2,
                "upperThreshold": threshold_value,
            }
        elif edition == 1 and comparison == ">":
            grib_keys = {
                "localDefinitionNumber": 5,
                "localDecimalScaleFactor": self.scale_factor,
                "thresholdIndicator": 1,
                "lowerThreshold": threshold_value,
            }
        elif edition == 2:
            # GRIB 2 has probability types above/below upper/lower limits (see Code Table 4.9)
            # where the threshold value can correspond to either limit. The default limit type
            # is upper for "<" and lower for ">", consistent with the GRIB 1 to GRIB 2 conversion
            # assumption.
            prob_types = {"<": {"upper": 4, "lower": 0}, ">": {"upper": 1, "lower": 3}}
            probability_type = prob_types[comparison][self.limit_type]
            missing = "Upper" if self.limit_type == "lower" else "Lower"
            grib_keys = {
                f"scaleFactorOf{self.limit_type.capitalize()}Limit": self.scale_factor,
                f"scaledValueOf{self.limit_type.capitalize()}Limit": threshold_value,
                "probabilityType": probability_type,
                f"scaleFactorOf{missing}Limit": "MISSING",
                f"scaledValueOf{missing}Limit": "MISSING",
            }
        else:
            raise ValueError(
                f"Unsupported threshold comparison {comparison} for grib edition {edition}"
            )
        return grib_keys


class RangeThreshold(BaseModel):
    type_: Literal["range"] = Field("range", alias="type")
    lower_scale_factor: int = 0
    lower_comparison: str = ">="
    lower_value: float
    upper_scale_factor: int = 0
    upper_comparison: str = "<"
    upper_value: float

    def grib_keys(self, edition: int) -> dict:
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


def _discriminator(config: Any):
    return config.get("type", "single")


class ThresholdConfig(BaseModel):
    param_thresholds: List[
        Annotated[
            Union[
                Annotated[SingleThreshold, Tag("single")],
                Annotated[RangeThreshold, Tag("range")],
            ],
            Discriminator(_discriminator),
        ]
    ] = Field(default_factory=list)
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
