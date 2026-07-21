# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import bisect
import datetime
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
from earthkit.time import Sequence
from earthkit.time.climatology import RelativeYear
from earthkit.time.climatology import date_range
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel
from ppcore.utils.stepseq import fcmonth_to_steprange
from ppcore.schema.forecast import Dataset, Climatology


class BaseDeriver(BaseModel):
    model_config = ConfigDict(validate_by_name=True)


class DefaultStepDeriver(BaseDeriver):
    name: Literal["step_default"] = Field("step_default")
    by: Optional[int] = None
    include_start: bool = False
    allow_missing_zero: bool = False

    def _inst_step(self, step: int, fc_steps: list[int]) -> list[int]:
        if step not in fc_steps:
            raise ValueError(f"Required step {step} not in forecast steps")
        return [step]

    def _range(self, start: int, end: int, fc_steps: list[int]) -> list[int]:
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

    def derive(self, request: dict, forecast: Dataset) -> list[int]:
        steps = list(map(int, str(request["step"]).split("-")))
        fc_steps = list(map(int, forecast.steps(request)))
        if len(steps) == 1:
            return self._inst_step(int(steps[0]), fc_steps)
        return self._range(*steps, fc_steps=fc_steps)


class DeaccumulateStepDeriver(BaseDeriver):
    name: Literal["step_deaccumulate"] = Field("step_deaccumulate")
    by: int = 0
    allow_missing_zero: bool = False

    def _inst_step(self, step: int, fc_steps: list[int]) -> list[int]:
        if step == 0:
            raise ValueError("Cannot perform de-accumulation for step 0")
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

    def derive(self, request: dict, forecast: Dataset) -> list[int]:
        steps = list(map(int, str(request["step"]).split("-")))
        fc_steps = list(map(int, forecast.steps(request)))
        if len(steps) == 1:
            return self._inst_step(int(steps[0]), fc_steps)
        return self._range(*steps, fc_steps=fc_steps)


class PrecomputedStepDeriver(BaseDeriver):
    name: Literal["step_precomputed"] = Field("step_precomputed")

    def derive(self, request: dict, forecast: Dataset) -> list[int]:
        if not isinstance(request["step"], str):
            raise ValueError(
                f"Step {request['step']} must be step range for pre-computed steps"
            )
        return [request["step"]]


class FcmonthStepDeriver(DefaultStepDeriver):
    name: Literal["step_monthly"] = Field("step_monthly")

    def derive(self, request: dict, forecast: Dataset) -> list[int]:
        fcmonth = int(request["fcmonth"])
        fc_steps = list(map(int, forecast.steps(request)))
        start, end = map(
            int,
            fcmonth_to_steprange(
                datetime.datetime.strptime(str(request["date"]), "%Y%m%d"),
                fcmonth,
            ).split("-"),
        )
        return self._range(start, end, fc_steps)


class SelectionStepDeriver(DefaultStepDeriver):
    name: Literal["step_select"] = Field("step_select")
    index: int

    def derive(self, request: dict, forecast: Dataset) -> list[int]:
        steps = list(map(int, str(request["step"]).split("-")))
        if len(steps) == 1:
            raise ValueError("SelectionStepDeriver can not be used for single steps")
        fc_steps = list(map(int, forecast.steps(request)))
        selection_range = self._range(*steps, fc_steps=fc_steps)
        return [selection_range[self.index]]


class StaticStepDeriver(BaseDeriver):
    name: Literal["step_static"] = Field("step_static")
    values: Union[int, list[int]]

    def derive(self, request: dict, forecast: Dataset) -> list[int]:
        steps = [self.values] if isinstance(self.values, int) else self.values
        fc_steps = forecast.steps(request)
        if not all([x in fc_steps for x in steps]):
            raise ValueError(f"Values {steps}, must be contained in forecast steps")
        return steps


class ClimStepRangeDeriver(BaseDeriver):
    name: Literal["clim_step_range"] = Field("clim_step_range")

    def derive(self, request: dict, climatology: Climatology) -> str:
        time = int(request["time"]) // 100
        req_steps = request["step"]
        if isinstance(req_steps, (int, str)):
            req_steps = [req_steps]
        clim_steps = list(map(str, climatology.steps(request)))

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


class ClimStepInstantaneousDeriver(BaseDeriver):
    name: Literal["clim_step_inst"] = Field("clim_step_inst")

    def derive(self, request: dict, climatology: Climatology) -> list[int]:
        time = int(request["time"]) // 100
        steps = request["step"]
        if isinstance(steps, (int, str)):
            steps = [steps]
        clim_steps = climatology.steps(request)
        # Match on valid time. Reforecast assumed to run at 00UTC only
        matched_steps = []
        for step in steps:
            valid_time = step + time
            if valid_time > clim_steps[-1] and time == 12:
                # This rule only applies to 12UTC runs
                valid_time = step - time
            if valid_time not in clim_steps:
                raise ValueError(f"Required step {valid_time} not in climatology steps")
            matched_steps.append(valid_time)
        return matched_steps


class ClimDateDeriver(PProcBaseModel):
    model_config = ConfigDict(extra="allow")
    name: Literal["clim_date"] = Field("clim_date")
    option: str
    sequence: Optional[dict] = None

    def derive(self, request: dict, climatology: Climatology) -> str | list[str]:
        date = datetime.datetime.strptime(str(request["date"]), "%Y%m%d")
        if self.sequence is not None:
            seq = Sequence.from_dict(self.sequence)
        else:
            seq = Sequence.from_resource(climatology.scheme)
        kwargs = self.model_dump(exclude={"option", "sequence", "name"})
        clim_date = getattr(seq, self.option)(date.date(), **kwargs)
        if isinstance(clim_date, datetime.date):
            return datetime.datetime.strftime(clim_date, "%Y%m%d")
        return [datetime.datetime.strftime(x, "%Y%m%d") for x in clim_date]


class HindcastDatesDeriver(PProcBaseModel):
    name: Literal["hindcast_dates"] = Field("hindcast_dates")
    rstart: int
    rend: int
    recurrence: str = "yearly"
    include_endpoint: bool = False

    def derive(self, request: dict, forecast: Dataset) -> list[str]:
        date = datetime.datetime.strptime(str(request["date"]), "%Y%m%d").date()
        start = RelativeYear(self.rstart).relative_to(date)
        end = RelativeYear(self.rend).relative_to(date)
        return [
            datetime.datetime.strftime(x, "%Y%m%d")
            for x in date_range(
                date, start, end, self.recurrence, self.include_endpoint
            )
        ]


class OutputDeriver(PProcBaseModel):
    name: Literal["from_output"] = Field("from_output")
    key: str

    def derive(self, request: dict, forecast: Dataset) -> Any:
        return request[self.key]


ForecastDeriver = Annotated[
    Union[
        DefaultStepDeriver,
        DeaccumulateStepDeriver,
        PrecomputedStepDeriver,
        FcmonthStepDeriver,
        SelectionStepDeriver,
        StaticStepDeriver,
        HindcastDatesDeriver,
        OutputDeriver,
    ],
    Field(discriminator="name"),
]

ClimatologyDeriver = Annotated[
    Union[
        ClimStepRangeDeriver,
        ClimStepInstantaneousDeriver,
        ClimDateDeriver,
        OutputDeriver,
    ],
    Field(discriminator="name"),
]
