from pydantic import BaseModel, model_validator
from typing import Union, Any, Optional


class Threshold(BaseModel):
    comparison: str
    value: float


class ThresholdConfig(BaseModel):
    out_paramid: int
    local_scale_factor: int = 0
    param_thresholds: list[Threshold]
    metadata: dict = {}
    limit_type: str = "lower"

    @model_validator(mode="before")
    def validate_thresholds(cls, data: Any) -> Any:
        if isinstance(data, dict) and "param_thresholds" not in data:
            value = data.pop("value")
            comparison = data.pop("comparison")
            data["param_thresholds"] = [{"value": value, "comparison": comparison}]
        return data

    def grib_keys(self, edition: int, clim_metadata: Optional[dict] = None) -> dict:
        """
        Creates dictionary of threshold related grib headers
        """
        threshold_dict = {"paramId": self.out_paramid}
        scale_factor = self.local_scale_factor
        threshold_value = round(self.param_thresholds[0].value * 10**scale_factor, 0)
        comparison = self.param_thresholds[0].comparison.strip("=")
        if edition == 1 and comparison == "<":
            grib_keys = {
                "localDefinitionNumber": 5,
                "localDecimalScaleFactor": scale_factor,
                "thresholdIndicator": 2,
                "upperThreshold": threshold_value,
            }
        elif edition == 1 and comparison == ">":
            grib_keys = {
                "localDefinitionNumber": 5,
                "localDecimalScaleFactor": scale_factor,
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
                f"scaleFactorOf{self.limit_type.capitalize()}Limit": scale_factor,
                f"scaledValueOf{self.limit_type.capitalize()}Limit": threshold_value,
                "probabilityType": probability_type,
                f"scaleFactorOf{missing}Limit": "MISSING",
                f"scaledValueOf{missing}Limit": "MISSING",
            }
            if clim_metadata:
                grib_keys.update(clim_metadata)
        else:
            raise ValueError(
                f"Unsupported threshold comparison {comparison} for grib edition {edition}"
            )

        threshold_dict.update(grib_keys)
        threshold_dict.update(self.metadata)
        return threshold_dict
