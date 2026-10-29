# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import model_validator

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel


class MaskExpression(PProcBaseModel):
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
