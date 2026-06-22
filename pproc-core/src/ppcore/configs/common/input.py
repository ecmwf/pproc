# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, ClassVar, Union

from annotated_types import Annotated
from pydantic import (
    BaseModel,
    Field,
    create_model,
    field_validator,
)
from ppcore.configs.common.source import (
    FDBSource,
    FilePatternSource,
    FileSource,
    MARSSource,
)
from ppcore.utils.io import split_location


class Input(BaseModel):
    sources: list[
        Annotated[
            Union[FileSource, FilePatternSource, FDBSource, MARSSource],
            Field(discriminator="name"),
        ]
    ] = [FDBSource()]
    requests: list[dict] = []
    dtype: str = "float32"

    @field_validator("sources", mode="before")
    @classmethod
    def validate_sources(cls, data: Any) -> Any:
        for index, source in enumerate(data):
            if isinstance(source, str):
                name, loc = split_location(source, default="file")
                config = {"name": name}
                if loc:
                    if name == "fdb":
                        config["config"] = loc
                    elif name == "file":
                        config["file"] = loc
                    elif name == "file-pattern":
                        config["pattern"] = loc
                    else:
                        raise ValueError(
                            f"Source type {name} does not support location specification"
                        )
                data[index] = config
        return data

    def base_request(self) -> dict:
        keys = set(self.requests[0].keys())
        for ireq in range(1, len(self.requests)):
            keys.intersection_update(self.requests[ireq].keys())
        return {
            k: self.requests[0][k]
            for k in keys
            if all(self.requests[0][k] == x[k] for x in self.requests)
        }


class InputsCollection(BaseModel):
    names: ClassVar[list[str]]


def create_input_model(
    name: str, inputs: list[str], optional: list[str] = [], **kwargs
) -> InputsCollection:
    field_definitions = {input: (Input, ...) for input in inputs}
    for input in optional:
        field_definitions[input] = (Input, Input(source={"type": "null"}))
    return create_model(
        f"{name}InputModel",
        names=(ClassVar[list[str]], inputs + optional),
        **field_definitions,
        __base__=InputsCollection,
        **kwargs,
    )
