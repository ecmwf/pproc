from typing import Optional

from earthkit.workflows.plugins.pproc.config.threshold import Threshold


def threshold_metadata(
    threshold: Threshold,
    metadata: Optional[dict] = None,
    clim_metadata: Optional[dict] = None,
) -> dict:
    metadata = metadata or {}
    edition = metadata.get("edition", 1)
    thr_metadata = {}
    if "paramId" in metadata:
        thr_metadata["paramId"] = metadata["paramId"]

    if threshold.upper_comparison is None:
        threshold_value = round(
            threshold.lower_value * 10**threshold.lower_scale_factor, 0
        )
        comparison = threshold.lower_comparison.strip("=")
        if edition == 1 and comparison == "<":
            thr_metadata.update(
                {
                    "localDefinitionNumber": 5,
                    "localDecimalScaleFactor": threshold.lower_scale_factor,
                    "thresholdIndicator": 2,
                    "upperThreshold": threshold_value,
                }
            )
        elif edition == 1 and comparison == ">":
            thr_metadata.update(
                {
                    "localDefinitionNumber": 5,
                    "localDecimalScaleFactor": threshold.lower_scale_factor,
                    "thresholdIndicator": 1,
                    "lowerThreshold": threshold_value,
                }
            )
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
            thr_metadata.update(
                {
                    f"scaleFactorOf{limit_type.capitalize()}Limit": threshold.lower_scale_factor,
                    f"scaledValueOf{limit_type.capitalize()}Limit": threshold_value,
                    "probabilityType": probability_type,
                    f"scaleFactorOf{missing}Limit": "MISSING",
                    f"scaledValueOf{missing}Limit": "MISSING",
                }
            )
            thr_metadata.update(clim_metadata or {})
        else:
            raise ValueError(
                f"Unsupported threshold comparison {comparison} for grib edition {edition}"
            )
        return thr_metadata
    if edition != 2:
        raise ValueError("Threshold ranges are only supported for GRIB edition 2")
    thr_metadata.update(
        {
            "scaleFactorOfLowerLimit": threshold.lower_scale_factor,
            "scaledValueOfLowerLimit": round(
                threshold.lower_value * 10**threshold.lower_scale_factor, 0
            ),
            "probabilityType": 2,
            "scaleFactorOfUpperLimit": threshold.upper_scale_factor,
            "scaledValueOfUpperLimit": round(
                threshold.upper_value * 10**threshold.upper_scale_factor, 0
            ),
        }
    )
    thr_metadata.update(clim_metadata or {})
    return thr_metadata
