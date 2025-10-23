# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Literal, Any, Optional, Annotated, Union
import datetime
import bisect
from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np

from earthkit.time import Sequence
from earthkit.time.climatology import RelativeYear, date_range

from pproc.common.stepseq import fcmonth_to_steprange


class DefaultStepDeriver(BaseModel):
    type_: Literal["default"] = Field("default", alias="type")
    by: Optional[int] = None
    include_start: bool = False
    allow_missing_zero: bool = False

    def _inst_step(self, step: int, fc_steps: list[int]) -> list[int]:
        if step not in fc_steps:
            raise ValueError(f"Required step {start} not in forecast steps")
        return [step]

    def _range(self, start: int, end: int, fc_steps: list[int]) -> tuple[int]:
        if self.by:
            fc_steps = [
                x
                for x in fc_steps
                if x
                in range((fc_steps[0] // self.by) * self.by, fc_steps[-1] + 1, self.by)
            ]
        if self.include_start:
            if start not in fc_steps:
                if start == 0 and self.allow_missing_zero:
                    start_index = 0
                else:
                    raise ValueError(f"Required step {start} not in forecast steps")
            else:
                start_index = fc_steps.index(start)
        else:
            start_index = bisect.bisect_right(fc_steps, start)
        if end not in fc_steps:
            raise ValueError(f"Required step {end} not in forecast steps")
        return fc_steps[start_index : fc_steps.index(end) + 1]

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        steps = list(map(int, str(output_request["step"]).split("-")))
        if len(steps) == 1:
            return self._inst_step(int(steps[0]), fc_steps)
        return self._range(*steps, fc_steps)


class DeaccumulateStepDeriver(BaseModel):
    type_: Literal["deaccumulate"] = Field("deaccumulate", alias="type")
    by: int = 0
    allow_missing_zero: bool = False

    def _inst_step(self, step: int, fc_steps: list[int]) -> list[int]:
        if step == 0:
            raise ValueError(f"Cannot perform de-accumulation for step 0")
        if step == fc_steps[0]:
            return [step]
        start = fc_steps[fc_steps.index(step) - 1]
        return self._range(start, step, fc_steps)

    def _range(self, start: int, end: int, fc_steps: list[int]) -> list[int]:
        end = max(end, self.by)
        start = min(end - self.by, start)
        if end not in fc_steps:
            raise ValueError(f"Required step {end} not in forecast steps")
        if start not in fc_steps:
            if start == 0 and self.allow_missing_zero:
                return [end]
            raise ValueError(f"Required step {start} not in forecast steps")
        return [start, end]

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        steps = list(map(int, str(output_request["step"]).split("-")))
        if len(steps) == 1:
            return self._inst_step(int(steps[0]), fc_steps)
        return self._range(*steps, fc_steps)


class PrecomputedStepDeriver(BaseModel):
    type_: Literal["precomputed"] = Field("precomputed", alias="type")

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        if not isinstance(output_request["step"], str):
            raise ValueError(
                f"Step {output_request['step']} must be step range for pre-computed steps"
            )
        return [output_request["step"]]


class FcmonthStepDeriver(DefaultStepDeriver):
    type_: Literal["monthly"] = Field("monthly", alias="type")

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        fcmonth = int(output_request["fcmonth"])
        start, end = map(
            int,
            fcmonth_to_steprange(
                datetime.datetime.strptime(str(output_request["date"]), "%Y%m%d"),
                fcmonth,
            ).split("-"),
        )
        return self._range(start, end, fc_steps)


class SelectionStepDeriver(DefaultStepDeriver):
    type_: Literal["select"] = Field("select", alias="type")
    index: int

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        steps = list(map(int, str(output_request["step"]).split("-")))
        if len(steps) == 1:
            assert ValueError("SelectionStepDeriver can not be used for single steps")
        selection_range = self._range(*steps, fc_steps)
        return [selection_range[self.index]]


class StaticStepDeriver(BaseModel):
    type_: Literal["static"] = Field("static", alias="type")
    values: Union[int, list[int]]

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        steps = [self.values] if isinstance(self.values, int) else self.values
        if not all([x in fc_steps for x in steps]):
            raise ValueError(f"Values {steps}, must be contained in forecast steps")
        return steps


ForecastStepDeriver = Annotated[
    Union[
        DefaultStepDeriver,
        DeaccumulateStepDeriver,
        PrecomputedStepDeriver,
        FcmonthStepDeriver,
        SelectionStepDeriver,
        StaticStepDeriver,
    ],
    Field(default_factory=DefaultStepDeriver, discriminator="type_"),
]


class ClimStepDeriver(BaseModel):
    type_: Literal["range", "instantaneous"] = Field("range", alias="type")

    @model_validator(mode="before")
    @classmethod
    def set_type(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        return data

    @staticmethod
    def _range(fc_request: dict, clim_steps: list[int]) -> str:
        time = int(fc_request["time"]) // 100
        req_steps = fc_request["step"]

        if len(req_steps) == 1 and isinstance(req_steps[0], str):
            req_steps = list(map(int, req_steps[0].split("-")))
        start, end = req_steps[0], req_steps[-1]
        widths = [np.diff(list(map(int, x.split("-"))))[0] for x in clim_steps]
        width = end - start
        if width not in widths:
            if len(req_steps) == 1:
                # Only possible if start should be step 0 and it is missing
                start = 0
            else:
                step_intervals = np.diff(req_steps)
                if np.any(step_intervals != step_intervals[0]):
                    raise ValueError(
                        "Can not derive width not irregular step intervals"
                    )
                start = start - step_intervals[0]
            width = end - start
            if width not in widths:
                raise ValueError("Can not derive step range from input steps")
        filtered_clim_steps = [
            x for index, x in enumerate(clim_steps) if widths[index] == width
        ]

        # Find nearest clim window range to real forecast time
        relative_time = start + int(time)
        if time == 12:
            relative_time = start - int(time)

        if end < 240:
            clim_start_steps = [int(x.split("-")[0]) for x in filtered_clim_steps]
            nearest = bisect.bisect_right(clim_start_steps, relative_time)
            clim_start = clim_start_steps[nearest - 1]
            if (nearest <= (len(clim_start_steps) - 1)) and (
                (clim_start_steps[nearest] - relative_time)
                < (relative_time - clim_start)
            ):
                clim_start = clim_start_steps[nearest]
            return f"{clim_start}-{clim_start + width}"
        return f"{start}-{end}"

    @staticmethod
    def _instantaneous(fc_request: dict, clim_steps: list[int]) -> list[int]:
        time = int(fc_request["time"]) // 100
        steps = fc_request["step"]
        if time in [12, 18]:
            return [
                (step - 12) if step == clim_steps[-1] else step + 12 for step in steps
            ]
        return steps

    def derive(self, request: dict, clim_steps: list[str]) -> int | str:
        return getattr(ClimStepDeriver, f"_{self.type_}")(request, clim_steps)


class ClimDateDeriver(BaseModel):
    model_config = ConfigDict(extra="allow")

    option: str
    sequence: Optional[dict] = None

    def derive(self, fc_request: dict, scheme: str) -> str | list[str]:
        date = datetime.datetime.strptime(str(fc_request["date"]), "%Y%m%d")
        if self.sequence is not None:
            seq = Sequence.from_dict(self.sequence)
        else:
            seq = Sequence.from_resource(scheme)
        kwargs = self.model_dump(exclude={"option", "sequence"})
        clim_date = getattr(seq, self.option)(date.date(), **kwargs)
        if isinstance(clim_date, datetime.date):
            return datetime.datetime.strftime(clim_date, "%Y%m%d")
        return [datetime.datetime.strftime(x, "%Y%m%d") for x in clim_date]


class HindcastDatesDeriver(BaseModel):
    rstart: int
    rend: int
    recurrence: str = "yearly"
    include_endpoint: bool = False

    def derive(self, fc_request: dict) -> list[str]:
        date = datetime.datetime.strptime(str(fc_request["date"]), "%Y%m%d").date()
        kwargs = self.model_dump()
        start = RelativeYear(self.rstart).relative_to(date)
        end = RelativeYear(self.rend).relative_to(date)
        return [
            datetime.datetime.strftime(x, "%Y%m%d")
            for x in date_range(
                date, start, end, self.recurrence, self.include_endpoint
            )
        ]
