# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, ClassVar, Union

from pydantic import (
    create_model,
    field_validator,
)
from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from ppcore.utils.io import split_location
from ppcore.configs.common.target import Target


class Output(PProcBaseModel):
    targets: list[Target] = []
    request: dict
    metadata: dict = {}

    @field_validator("targets", mode="before")
    @classmethod
    def validate_targets(cls, data: Any) -> Any:
        for index, target in enumerate(data):
            if isinstance(target, str):
                name, loc = split_location(target, default="file")
                config = {"name": name}
                if loc:
                    if name == "fdb":
                        config["config"] = loc
                    elif name in ["file", "file-pattern"]:
                        config["file"] = loc
                    elif name == "zarr":
                        config["xarray_to_zarr_kwargs"] = {"store": loc, "mode": "w"}
                    else:
                        raise ValueError(
                            f"Target type {name} does not support location specification"
                        )
                data[index] = config
        return data


class OutputsCollection(PProcBaseModel):
    names: ClassVar[list[str]]


def create_output_model(
    name: str, outputs: Union[list[str], dict[str, dict]], **kwargs
):
    field_definitions = {output: (Output, ...) for output in outputs}
    names = outputs if isinstance(outputs, list) else list(outputs.keys())
    return create_model(
        f"{name}OutputModel",
        names=(ClassVar[list[str]], names),
        **field_definitions,
        __base__=OutputsCollection,
        **kwargs,
    )
