from pydantic import BaseModel, Field, RootModel
from typing import Literal, Optional, Annotated, Union
import numpy as np
import bisect

from pproc.schema.base import Schema
from pproc.common.stepseq import stepseq_monthly, steprange_to_fcmonth


class Instantaneous(BaseModel):
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


class Range(BaseModel):
    type_: Literal["range"] = Field("range", alias="type")
    start: Optional[int] = None
    end: Optional[int] = None
    interval: int
    width: int
    dim: Literal["step"] = "step"

    def generate_steps(self, steps: list[int | str]) -> list[str]:
        if all(isinstance(x, str) for x in steps):
            return steps
        assert all(
            isinstance(x, int) for x in steps
        ), "Steps can not be a mix of strings and integers"
        start = self.start or steps[0]
        end = self.end or steps[-1]
        return [
            f"{rstart}-{rstart + self.width}"
            for rstart in range(start, end - self.width + 1, self.interval)
        ]


class Monthly(BaseModel):
    type_: Literal["monthly"] = Field("monthly", alias="type")
    date: str
    dim: Literal["fcmonth"] = "fcmonth"

    def generate_steps(self, steps: list[int]) -> list[str]:
        by = np.diff(steps)
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


class StepSchema(Schema):
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

    def in_steps(self, request: dict) -> list[int]:
        config = self.traverse(request)
        return self._create_steps(config.get("in_steps", []))

    def out_steps(
        self, request_or_name: dict | str, steps: Optional[list[int]] = None
    ) -> tuple[str, list[int | str]]:
        if isinstance(request_or_name, str):
            step_configs = self.schema.get("defs", {}).get(request_or_name, [])
        else:
            for dim in ["step", "fcmonth"]:
                if dim in request_or_name:
                    return dim, [request_or_name[dim]]

            config = self.traverse(request_or_name, {})
            steps = steps or self._create_steps(config.get("in_steps", []))
            step_configs = config.get("out_steps")

        if isinstance(step_configs, dict):
            step_configs = [step_configs]

        out = []
        dim = None
        for step_config in step_configs:
            if isinstance(step_config, str):
                if all(x in steps for x in map(int, step_config.split("-"))):
                    out.append(step_config)
                continue
            step_config = {
                k: v.format_map(request_or_name) if isinstance(v, str) else v
                for k, v in step_config.items()
            }
            step_type = StepType(**step_config).root
            if dim is None:
                dim = step_type.dim
            assert dim == step_type.dim, "All steps must be of the same dimension"
            out.extend([x for x in step_type.generate_steps(steps) if x not in out])
        return dim, out
