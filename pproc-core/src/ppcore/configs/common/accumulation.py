from typing import Optional, Union, Literal, Annotated, Any

from pydantic import BaseModel, Field, field_validator

from earthkit.workflows.plugins.pproc.config.accumulation import (
    Default,
    Monthly,
)


class Accumulation(BaseModel):
    operation: Optional[Literal["min", "max", "mean", "std", "add", "diff"]] = None
    coords: list[Union[list[int], list[str]]]
    metadata: Optional[dict] = None
    deaccumulate: bool = False
    name: Optional[
        Annotated[
            Union[Default, Monthly],
            Field(discriminator="type_"),
        ]
    ] = None

    @field_validator("operation", mode="before")
    @classmethod
    def validate_operation(cls, operation: str) -> Optional[str]:
        OPS = {
            "aggregation": None,
            "difference": "diff",
            "maximum": "max",
            "minimum": "min",
            "mean": "mean",
            "standard_deviation": "std",
            "sum": "add",
        }
        return OPS.get(operation, operation)

    @field_validator("coords", mode="before")
    @classmethod
    def validate_coords(cls, values: Any) -> list[Union[list[int], list[str]]]:
        if isinstance(values, (str, int)):
            return [[values]]
        if isinstance(values, list) and all(isinstance(v, (str, int)) for v in values):
            return [values]
        return values
