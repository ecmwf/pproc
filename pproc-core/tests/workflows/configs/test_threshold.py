import pytest
from typing import Any
from pydantic import ValidationError

from earthkit.workflows.plugins.pproc.config.threshold import Threshold

# class Threshold(PProcBaseModel):
#     select: Optional[dict] = None
#     lower_scale_factor: int
#     lower_comparison: Comparisons
#     lower_value: float
#     upper_scale_factor: int = 0
#     upper_comparison: Optional[Comparisons] = None
#     upper_value: Optional[float] = None


@pytest.mark.parametrize(
    "config, expected", [
        [{"lower_value": 1, "lower_comparison": "<", "lower_scale_factor": 1}, {"lower_value": 1, "lower_comparison": "<", "lower_scale_factor": 1}], 
        [{"lower_value": 0.01, "lower_comparison": "<"}, {"lower_value": 0.01, "lower_comparison": "<", "lower_scale_factor": 2}],
        [{"comparison": "<", "value": 10.}, {"lower_value": 10., "lower_comparison": "<", "lower_scale_factor": 0, "upper_value": None, "upper_comparison": None, "upper_scale_factor": 0}],
        [{"comparison": "<", "value": 273.15}, {"lower_value": 273.15, "lower_comparison": "<", "lower_scale_factor": 2}],
        [{"lower_value": 0, "lower_comparison": ">", "upper_value": 1, "upper_comparison": "<", "upper_scale_factor": 1}, {"upper_value": 1, "upper_comparison": "<", "upper_scale_factor": 1}], 
        [{"lower_value": 0.01, "lower_comparison": ">", "upper_value": 0.01, "upper_comparison": "<"}, {"lower_value": 0.01, "lower_comparison": ">", "lower_scale_factor": 2, "upper_value": 0.01, "upper_comparison": "<", "upper_scale_factor": 2}],
    ], 
    ids=["all-explicit", "derive-lower-scale-factor", "legacy-comparison", "legacy-comparison-with-float", "upper-threshold", "derive-upper-scale-factor"]
)
def test_threshold_validation(config: dict[str, Any], expected: dict[str, Any]):
    threshold = Threshold(**config)
    for key, value in expected.items():
        assert getattr(threshold, key) == value


@pytest.mark.parametrize(
    "config, error, match", [
        [{"comparison": "<"}, ValidationError, "2 validation errors"],
        [{"lower_comparison": "<", "lower_scale_factor": 0}, ValidationError, "1 validation error"],
        [{"upper_value": 1}, ValidationError, "Both upper_comparison and upper_value must be provided together"],
        [{"upper_value": 1, "upper_comparison": "<"}, ValidationError, "Upper threshold should not be used without lower threshold"],
    ],
    ids=["missing-value", "missing-lower-value", "missing-upper-comparison", "upper-without-lower"]
)
def test_threshold_invalid(config: dict[str, Any], error: type[Exception], match: str):
    with pytest.raises(error, match=match):
        threshold = Threshold(**config)