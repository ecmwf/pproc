# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Literal, Annotated, Any

from pydantic import Discriminator, Field, Tag, field_validator, model_validator

from earthkit.workflows.fluent import Payload
from earthkit.workflows.plugins.pproc.config.accumulation import (
    Default,
    Monthly,
)
from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel

Coords = Union[list[str], list[int]]


class BaseAccumulation(PProcBaseModel):
    coords: list[Coords]
    payload: Optional[str] = None
    metadata: Optional[dict] = None
    deaccumulate: bool = False
    name: Annotated[
        Union[Default, Monthly],
        Field(discriminator="type_", default_factory=Default),
    ]

    @model_validator(mode="before")
    def validate_config(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # TODO Update schema to remove these keys
            data.pop("type", None)
        return data

    @field_validator("coords", mode="before")
    @classmethod
    def validate_coords(cls, values: Any) -> list[Union[list[int], list[str]]]:
        if isinstance(values, (str, int)):
            return [[values]]
        if isinstance(values, list) and all(isinstance(v, (str, int)) for v in values):
            return [values]
        return values

    def create_action(self) -> dict:
        return {
            "operation": self.payload,
            "coords": self.coords,
            "metadata": self.metadata,
            "deaccumulate": self.deaccumulate,
            "name": self.name,
        }


class NullAccumulation(BaseAccumulation):
    operation: Literal["aggregation"] = "aggregation"
    payload: None = None


class MinAccumulation(BaseAccumulation):
    operation: Literal["minimum"] = "minimum"
    payload: Literal["min"] = "min"


class MaxAccumulation(BaseAccumulation):
    operation: Literal["maximum"] = "maximum"
    payload: Literal["max"] = "max"


class MeanAccumulation(BaseAccumulation):
    operation: Literal["mean"] = "mean"
    payload: Literal["mean"] = "mean"


class StdAccumulation(BaseAccumulation):
    operation: Literal["standard_deviation"] = "standard_deviation"
    payload: Literal["std"] = "std"


class SumAccumulation(BaseAccumulation):
    operation: Literal["sum"] = "sum"
    payload: Literal["sum"] = "sum"


class DifferenceAccumulation(BaseAccumulation):
    operation: Literal["difference"] = "difference"
    payload: None = None
    deaccumulate: bool = Field(True, frozen=True)

    @field_validator("coords", mode="after")
    @classmethod
    def validate_num_coords(cls, coords: list[Coords]) -> list[Coords]:
        if any([len(coord) > 2 for coord in coords]):
            raise ValueError("difference accumulation accepts only 1 or 2 coordinates")
        return coords


class DifferenceRateAccumulation(BaseAccumulation):
    operation: Literal["difference_rate"] = "difference_rate"
    payload: Literal["ppruntime.accumulation.difference_rate"] = (
        "ppruntime.accumulation.difference_rate"
    )
    factor: float = 1.0

    @field_validator("coords", mode="after")
    @classmethod
    def validate_num_coords(cls, coords: list[Coords]) -> list[Coords]:
        if any([len(coord) > 2 for coord in coords]):
            raise ValueError(
                "difference_rate accumulation accepts only 1 or 2 coordinates"
            )
        return coords

    def create_action(self) -> dict:
        return {
            "operation": Payload(
                self.payload,
                kwargs={"factor": self.factor, "metadata": self.metadata},
                metadata={"environment": ["ppruntime"]},
            ),
            "coords": self.coords,
            "deaccumulate": self.deaccumulate,
            "name": self.name,
        }


def accum_discriminator(data: Any) -> str:
    if isinstance(data, dict):
        return data.get("operation", "aggregation")
    return data.operation


Accumulation = Annotated[
    Union[
        Annotated[NullAccumulation, Tag("aggregation")],
        Annotated[MinAccumulation, Tag("minimum")],
        Annotated[MaxAccumulation, Tag("maximum")],
        Annotated[MeanAccumulation, Tag("mean")],
        Annotated[StdAccumulation, Tag("standard_deviation")],
        Annotated[SumAccumulation, Tag("sum")],
        Annotated[DifferenceAccumulation, Tag("difference")],
        Annotated[DifferenceRateAccumulation, Tag("difference_rate")],
    ],
    Discriminator(accum_discriminator),
]
