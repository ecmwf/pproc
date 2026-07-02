# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from typing_extensions import Self

from ppcore.schema.config import ConfigSchema
from ppcore.schema.input import InputSchema
from ppcore.schema.step import StepSchema
from ppcore.utils.mars import METADATA_KEYS
from ppcore.utils.requests import VALUE_TYPES
from ppcore.utils.requests import validate_request


class Schema:
    def __init__(self, config: dict, inputs: dict, windows: dict):
        self.config_schema = ConfigSchema(config)
        self.param_schema = InputSchema(inputs)
        self.step_schema = StepSchema(windows)

    @classmethod
    def from_file(cls, schema_path: str) -> Self:
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
        return cls(**schema)

    def config_from_output(self, output_request: dict) -> dict:
        config = self.config_schema.config(output_request)
        inputs = list(self.param_schema.inputs(output_request, self.step_schema))

        # Set metadata
        base_request = inputs[0]
        metadata = config.setdefault("metadata", {})
        if np.size(base_request["param"]) > 1:
            config["name"] = f"{output_request['param']}_{output_request['levtype']}"
        config.setdefault(
            "name", f"{base_request['param']}_{output_request['levtype']}"
        )
        for key in config.pop("metadata_from_output", []):
            if base_request.get(key, None) != VALUE_TYPES.get(key, str)(
                output_request[key]
            ):
                metadata_key = METADATA_KEYS.get(key, key)
                metadata[metadata_key] = VALUE_TYPES.get(metadata_key, str)(
                    output_request[key]
                )
        return {**config, "inputs": inputs}

    def config_from_input(
        self,
        input_requests: list[dict],
        output_template: Optional[dict] = None,
        entrypoint: Optional[str] = None,
    ) -> Iterator[dict]:
        # If entrypoint is provided, find output templates provided by that entrypoint
        if entrypoint is not None:
            reconstructed = self.config_schema.reconstruct(
                output_template=(
                    None
                    if output_template is None
                    else validate_request(output_template)
                ),
                **({} if entrypoint is None else {"entrypoint": entrypoint}),
            )
            matching_types = pd.DataFrame([x for x, _ in reconstructed])
            output_keys = (
                [] if output_template is None else list(output_template.keys())
            )
            drop = [
                x
                for x in self.config_schema.all_filters.difference(
                    ["type"] + output_keys
                )
            ]
            matching_types.drop(columns=drop, inplace=True, errors="ignore")
            matching_types.drop_duplicates(inplace=True)
            output_templates = matching_types.to_dict(orient="records")
        else:
            output_templates = [output_template]

        for template in output_templates:
            for output, inputs in self.outputs_from_inputs(
                input_requests, output_template=template
            ):
                out_config = self.config_from_output(output)
                out_config["inputs"] = inputs
                yield out_config

    def outputs_from_inputs(
        self,
        inputs: list[dict],
        output_template: Optional[dict] = None,
    ) -> Iterator[tuple[dict, list[dict]]]:
        return self.param_schema.outputs(inputs, self.step_schema, output_template)
