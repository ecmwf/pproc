# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Union, Literal, Annotated, Optional
from pydantic import Field, model_validator

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from ppcore.configs.common.preprocessing import PreprocessingConfig
from ppcore.configs.common.input import create_input_model, InputsCollection
from ppcore.configs.common.output import Output
from ppcore.configs.common.accumulation import Accumulation
from ppcore.configs.common.stats import (
    Mean,
    StandardDeviation,
    Quantiles,
    ThresholdProbability,
)
from ppcore.utils.mars import extract_mars


EnsembleInputModel: type[InputsCollection] = create_input_model(
    "EnsembleInputsModel", inputs=["fc"]
)


class Ensemble(PProcBaseModel):
    name: Literal["ensemble"] = Field("ensemble")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    accumulations: dict[str, Accumulation] = Field(default_factory=dict)
    statistics: Optional[
        Annotated[
            Union[Mean, StandardDeviation, Quantiles, ThresholdProbability],
            Field(discriminator="operation"),
        ]
    ] = None
    inputs: EnsembleInputModel  # type: ignore
    output: Output

    @model_validator(mode="before")
    def validate_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "requests" in data:
            requests = data.pop("requests")
            input_config = data.pop("input", {})
            original = requests.pop("original")
            sources = data.pop("sources", [])
            if isinstance(sources, dict):
                sources = sources["fc"]
            if isinstance(sources, str):
                sources = [sources]
            data["inputs"] = {
                "fc": {
                    "requests": requests.pop("inputs"),
                    "sources": sources,
                    **input_config,
                }
            }
            targets = data.pop("targets", [])
            data["output"] = {
                "targets": targets if isinstance(targets, list) else [targets],
                "request": extract_mars(original),
            }
            if metadata := data.pop("metadata", None):
                stat_metadata = data.get("statistics", {}).setdefault("metadata", {})
                stat_metadata.update(metadata)
            if data.pop("quantiles", None) is not None:
                data["statistics"]["quantiles"] = original["quantile"]
        return data


Config = Ensemble
