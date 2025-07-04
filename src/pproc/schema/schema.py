# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator, Optional
from typing_extensions import Self
import copy
import numpy as np
import pandas as pd
import yaml

from pproc.schema.config import ConfigSchema
from pproc.schema.input import InputSchema
from pproc.schema.step import StepSchema

from pproc.config.utils import expand, METADATA_KEYS

VALUE_TYPES = {
    "param": str,
    "paramId": int,
    "levelist": int,
    "step": int,
    "fcmonth": int,
    "number": int,
    "dataDate": int,
}


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

    @classmethod
    def validate_request(cls, request: dict) -> dict:
        out = copy.deepcopy(request)
        # Map types
        for key, value_type in VALUE_TYPES.items():
            if key in out:
                value = out[key]
                try:
                    out[key] = (
                        value_type(value)
                        if np.ndim(value) == 0
                        else list(map(value_type, value))
                    )
                except ValueError:
                    pass
        # Format time
        if "time" in out:
            time = out["time"]
            if isinstance(time, list):
                assert len(time) == 1, "Only single value of time supported per request"
                time = time[0]
            if isinstance(time, int):
                time = f"{time:02d}"
            out["time"] = time.ljust(4, "0")
        return out

    def config_from_output(
        self, output_request: dict, inputs: Optional[list[dict]] = None
    ) -> dict:
        valid_out = self.validate_request(output_request)
        config = self.config_schema.config(valid_out)
        inputs = inputs or list(self.param_schema.inputs(valid_out, self.step_schema))

        # Set metadata
        base_request = inputs[0]
        metadata = config.setdefault("metadata", {})
        if base_request["param"] != valid_out["param"]:
            if (
                not isinstance(base_request["param"], str)
                and len(base_request["param"]) > 1
            ):
                config["name"] = f"{valid_out['param']}_{valid_out['levtype']}"
        config.setdefault("name", f"{base_request['param']}_{valid_out['levtype']}")
        for key in ["param", "stream", "date"]:
            if base_request[key] != valid_out[key]:
                metadata_key = METADATA_KEYS.get(key, key)
                metadata[metadata_key] = VALUE_TYPES.get(metadata_key, str)(
                    valid_out[key]
                )
        return {**config, "inputs": inputs}

    def config_from_input(
        self,
        input_requests: list[dict],
        output_template: Optional[dict] = None,
        entrypoint: Optional[str] = None,
    ) -> Iterator[dict]:
        input_requests = list(
            expand([self.validate_request(req) for req in input_requests])
        )
        reconstructed = self.config_schema.reconstruct(
            output_template=None
            if output_template is None
            else self.validate_request(output_template),
            **({} if entrypoint is None else {"entrypoint": entrypoint}),
        )
        matching_types = pd.DataFrame([x for x, _ in reconstructed])
        output_keys = [] if output_template is None else list(output_template.keys())
        drop = [
            x for x in self.config_schema.all_filters.difference(["type"] + output_keys)
        ]
        matching_types.drop(columns=drop, inplace=True, errors="ignore")
        matching_types.drop_duplicates(inplace=True)
        for template in matching_types.to_dict(orient="records"):
            for output, inputs in self.param_schema.outputs(
                input_requests, self.step_schema, output_template=template
            ):
                yield self.config_from_output(output, inputs)
