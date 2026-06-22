# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator

from ppcore.configs.common.preprocessing import PreprocessingConfig
from ppcore.configs.common.input import create_input_model
from ppcore.configs.common.output import Output
from ppcore.configs.common.accumulation import Accumulation
from ppcore.configs.common.stats import (
    Mean,
    StandardDeviation,
    Quantiles,
    ThresholdProbability,
)


EnsembleInputModel = create_input_model("EnsembleInputsModel", inputs=["fc"])


class Ensemble(BaseModel):
    name: Literal["ensemble"] = Field("ensemble")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    accumulations: dict[str, Accumulation] = Field(default_factory=dict)
    statistics: Union[Mean, StandardDeviation, Quantiles, ThresholdProbability]
    inputs: Optional[EnsembleInputModel] = None
    output: Optional[Output] = None

    @model_validator(mode="before")
    def validate_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            print("Validating ensemble config...", data)
            if "requests" in data:
                requests = data.pop("requests")
                if sources := data.pop("sources", None):
                    if isinstance(sources, dict):
                        sources = sources["fc"]
                    if isinstance(sources, str):
                        sources = [sources]
                    data["inputs"] = {
                        "fc": {
                            "requests": requests.pop("inputs"),
                            "sources": sources,
                            "dtype": data.pop("dtype"),
                        }
                    }
            if targets := data.pop("targets", None):
                data["output"] = {
                    "targets": targets if isinstance(targets, list) else [targets],
                    "request": requests.pop("original"),
                }
            if metadata := data.pop("metadata", None):
                stat_metadata = data["statistics"].setdefault("metadata", {})
                stat_metadata.update(metadata)
        return data


Config = Ensemble
