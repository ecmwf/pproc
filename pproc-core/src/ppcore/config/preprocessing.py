# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from typing import Annotated
from typing import Any
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class Preprocessing(BaseModel, ABC):
    metadata: Optional[dict] = None


class Scaling(Preprocessing):
    #   - operation: scale
    #     value: 3600
    operation: Literal["scale"] = "scale"
    value: Union[float, int]


class Combination(Preprocessing):
    # - operation: norm
    operation: Literal["direction", "norm", "sum"]


class Reshape(Preprocessing):
    operation: Literal["reshape"] = "reshape"
    shape: Union[int, tuple[int, int]]
    order: Literal["F", "C"] = "F"


class Expression(Preprocessing):
    operation: Literal["expression"] = "expression"
    expr: str
    expr_data: dict
    dtype: Optional[str] = None


class MaskExpression(BaseModel):
    select: dict
    comparison: Literal["<", ">", ">=", "<=", "==", "!="]
    value: float

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if isinstance(data, list):
            assert (
                len(data) == 3
            ), "Mask expression should be a [select, comparison, value] list"
            return {"select": data[0], "comparison": data[1], "value": data[2]}
        return data


class Masking(Preprocessing):
    #   - operation: mask
    #     select: {param: 228036}
    #     mask: [{param: 228035, level: 250}, ">=", 10]
    operation: Literal["mask"] = "mask"
    mask: MaskExpression
    select: dict
    replacement: float = 0.0


class PreprocessingConfig(BaseModel):
    # preprocessing:
    #   - operation: norm
    #   - operation: mask
    #     select: {param: 228036}
    #     mask: [{param: 228035, level: 250}, ">=", 10]
    #   - operation: scale
    #     value: 3600
    actions: List[
        Annotated[
            Union[Scaling, Combination, Masking, Reshape, Expression],
            Field(discriminator="operation"),
        ]
    ] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"actions": data}
        return data
