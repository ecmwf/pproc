# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import Field
from pydantic import model_validator

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from earthkit.workflows.plugins.pproc.config.mask import MaskExpression


class Scale(PProcBaseModel):
    #   - operation: scale
    #     value: 3600
    operation: Literal["scale"] = "scale"
    value: Union[float, int]
    metadata: Optional[dict] = None


class Divide(PProcBaseModel):
    #   - operation: divide
    #     value: 3600
    operation: Literal["divide"] = "divide"
    value: Union[float, int]
    metadata: Optional[dict] = None


class Combination(PProcBaseModel):
    # - operation: norm
    operation: Literal["direction", "norm", "sum"]
    metadata: Optional[dict] = None


class Expand(PProcBaseModel):
    operation: Literal["expand"] = "expand"
    internal_dim: list[Union[str, list[str], list[int]]]
    backend_kwargs: Optional[dict] = None


class Select(PProcBaseModel):
    operation: Literal["select"] = "select"
    dim: str
    values: Union[int, str, list[int], list[str]]


class ThermalIndex(PProcBaseModel):
    operation: Literal["thermal_index"] = "thermal_index"
    function: str
    params: list[str]
    deaccumulate: Optional[list[str]] = None
    metadata: Optional[dict] = None
    join: bool = True


# class Expression(Preprocessing):
#     operation: Literal["expression"] = "expression"
#     expr: str
#     expr_data: dict
#     dtype: Optional[str] = None


class Masking(PProcBaseModel):
    #   - operation: mask
    #     select: {param: 228036}
    #     mask: [{param: 228035, level: 250}, ">=", 10]
    operation: Literal["mask"] = "mask"
    mask: MaskExpression
    select: dict
    replacement: float = 0.0
    metadata: Optional[dict] = None


class PreprocessingConfig(PProcBaseModel):
    # preprocessing:
    #   - operation: norm
    #   - operation: mask
    #     select: {param: 228036}
    #     mask: [{param: 228035, level: 250}, ">=", 10]
    #   - operation: scale
    #     value: 3600
    actions: List[
        Annotated[
            Union[Scale, Divide, Combination, Masking, Expand, Select, ThermalIndex],
            Field(discriminator="operation"),
        ]
    ] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"actions": data}
        return data
