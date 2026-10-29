# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Iterator, Optional, Union, Literal, Any

import numpy as np
import yaml
from typing_extensions import Self

from ppcore.schema.config import ConfigSchema
from ppcore.schema.input import InputSchema
from ppcore.schema.step import StepSchema
from ppcore.schema.forecast import (
    DatasetDefinitions,
    DatasetDefinition,
    ForecastDefinition,
    ReforecastDefinition,
    ClimatologyDefinition,
)
from ppcore.utils.mars import METADATA_KEYS
from ppcore.utils.requests import VALUE_TYPES, expand


class Schema:
    def __init__(
        self,
        config: dict,
        inputs: dict,
        windows: dict,
        datasets: dict,
        matching_cache_size: int = 0,
    ):
        self.config_schema = ConfigSchema(
            config, matching_cache_size=matching_cache_size
        )
        self.param_schema = InputSchema(inputs, matching_cache_size=matching_cache_size)
        self.step_schema = StepSchema(windows, matching_cache_size=matching_cache_size)
        self.datasets = DatasetDefinitions(definitions=datasets)

    @classmethod
    def from_file(
        cls, schema_path: Union[str, os.PathLike], matching_cache_size: int = 0
    ) -> Self:
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
        return cls(**schema, matching_cache_size=matching_cache_size)

    def config_from_output(
        self,
        output_request: dict[str, Any],
        forecast: Union[ForecastDefinition, ReforecastDefinition],
        climatology: Optional[ClimatologyDefinition] = None,
    ) -> dict:
        config = self.config_schema.config(output_request)
        inputs = list(self.param_schema.inputs(output_request, forecast, climatology))

        # Set metadata
        base_request = inputs[0]
        metadata = config.setdefault("metadata", {})
        if np.size(base_request["param"]) > 1:
            config["name"] = f"{output_request['param']}_{output_request['levtype']}"
        config.setdefault(
            "name", f"{base_request['param']}_{output_request['levtype']}"
        )
        for key in config.pop("metadata_from_output", []):
            if key in output_request and base_request.get(key, None) != VALUE_TYPES.get(
                key, str
            )(output_request[key]):
                metadata_key = METADATA_KEYS.get(key, key)
                metadata[metadata_key] = VALUE_TYPES.get(metadata_key, str)(
                    output_request[key]
                )
        return {**config, "inputs": inputs}

    def config_from_input(
        self,
        forecast: Union[ForecastDefinition, ReforecastDefinition],
        climatology: Optional[ClimatologyDefinition] = None,
        output_template: Optional[dict] = None,
    ) -> Iterator[dict]:
        for output_template in expand(output_template) if output_template else [None]:
            for output, inputs in self.outputs_from_inputs(
                forecast, climatology, output_template=output_template
            ):
                out_config = self.config_from_output(output, forecast, climatology)
                out_config["inputs"] = inputs
                yield out_config

    def outputs_from_inputs(
        self,
        forecast: Union[ForecastDefinition, ReforecastDefinition],
        climatology: Optional[ClimatologyDefinition] = None,
        output_template: Optional[dict[str, Any]] = None,
        method: Literal["dfs", "bfs"] = "bfs",
        enable_cache: bool = True,
    ) -> Iterator[tuple[dict, list[dict]]]:
        yield from self.param_schema.outputs(
            forecast,
            climatology,
            self.step_schema,
            output_template,
            method=method,
            enable_cache=enable_cache,
        )

    def definition(self, name: str) -> DatasetDefinition:
        return self.datasets.definition(name)
