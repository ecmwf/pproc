# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import bisect
from typing import Annotated
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
from pydantic import Field
from pydantic import RootModel

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from ppcore.schema.base import BaseSchema
from ppcore.schema.exceptions import PProcStepSchemaError
from ppcore.utils.stepseq import steprange_to_fcmonth, stepseq_monthly


class Instantaneous(PProcBaseModel):
    type_: Literal["instantaneous"] = Field("instantaneous", alias="type")
    deaccumulate: bool = False
    start: Optional[int] = None
    end: Optional[int] = None
    interval: Optional[int] = None
    dim: Literal["step"] = "step"

    def generate_steps(self, steps: list[int]) -> list[int]:
        start = self.start or steps[0]
        end = self.end or steps[-1]
        if self.interval:
            interval_steps = list(range(start, end + 1, self.interval))
            selected_steps = [x for x in interval_steps if x in steps]
        else:
            start_index = bisect.bisect_left(steps, start)
            end_index = bisect.bisect_right(steps, end)
            selected_steps = steps[start_index:end_index]
        return selected_steps[1:] if self.deaccumulate else selected_steps


class Range(PProcBaseModel):
    type_: Literal["range"] = Field("range", alias="type")
    start: Optional[int] = None
    end: Optional[int] = None
    interval: int
    width: int
    dim: Literal["step"] = "step"

    def generate_steps(self, steps: Union[list[int], list[str]]) -> list[str]:
        if all(isinstance(x, str) for x in steps):
            return steps  # type: ignore
        assert all(
            isinstance(x, int) for x in steps
        ), "Steps can not be a mix of strings and integers"
        start: int = self.start or steps[0]  # type: ignore
        end: int = min((self.end or steps[-1]), steps[-1]) - self.width  # type: ignore
        rstarts = set(range(start, end + 1, self.interval))
        rstarts.intersection_update(steps)
        return [f"{rstart}-{rstart + self.width}" for rstart in sorted(rstarts)]


class Monthly(PProcBaseModel):
    type_: Literal["monthly"] = Field("monthly", alias="type")
    date: str
    dim: Literal["fcmonth"] = "fcmonth"

    def generate_steps(self, steps: list[int]) -> list[int]:
        by = np.diff(steps)
        if len(by) == 0:
            return []
        if len(set(by)) != 1:
            raise ValueError("Monthly steps must be evenly spaced")
        return [
            steprange_to_fcmonth(self.date, f"{x[0]}-{x[-1]}")
            for x in stepseq_monthly(self.date, steps[0], steps[-1], by[0])
        ]


StepType = RootModel[
    Annotated[
        Union[Instantaneous, Range, Monthly],
        Field(discriminator="type_"),
    ]
]


class StepSchema(BaseSchema):
    exception = PProcStepSchemaError

    @classmethod
    def _create_steps(cls, step_config: list[dict]) -> list[int]:
        steps = set(
            sum(
                [
                    list(range(x["from"], x["to"] + 1, x.get("by", 1)))
                    for x in step_config
                ],
                [],
            )
        )
        return sorted(steps)

    def out_steps(
        self, request: dict, in_steps: list[int]
    ) -> tuple[str, list[Union[int, str]]]:
        for dim in ["step", "fcmonth"]:
            if dim in request:
                return dim, [request[dim]]

        config = self.traverse(request, {})
        step_configs = config.get("out_steps", None)
        if step_configs is None:
            raise self.exception(f"No output steps defined {request}")

        if isinstance(step_configs, dict):
            step_configs = [step_configs]

        out = []
        dim: str = ""
        for step_config in step_configs:
            if isinstance(step_config, str):
                if all(x in in_steps for x in map(int, step_config.split("-"))):
                    out.append(step_config)
                continue
            step_config = {
                k: v.format_map(request) if isinstance(v, str) else v
                for k, v in step_config.items()
            }
            step_type = StepType(**step_config).root
            if not dim:
                dim = step_type.dim
            assert dim == step_type.dim, "All steps must be of the same dimension"
            out.extend([x for x in step_type.generate_steps(in_steps) if x not in out])
        return dim, out
