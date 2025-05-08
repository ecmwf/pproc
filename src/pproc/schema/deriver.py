from typing import Literal, Any, Optional, Annotated, Union
import datetime
import bisect
from pydantic import BaseModel, Field, model_validator, ConfigDict

from earthkit.time import Sequence

from pproc.common.stepseq import fcmonth_to_steprange


class DefaultStepDeriver(BaseModel):
    type_: Literal["default"] = Field("default", alias="type")
    by: Optional[int] = None
    precomputed: bool = False
    deaccumulation: bool = False

    def _inst_step(self, step: int, fc_steps: list[int]) -> list[int]:
        in_steps = []
        if self.deaccumulation:
            if step == fc_steps[0]:
                raise ValueError(f"Cannot perform de-accumulation for step {step}")
            in_steps.append(fc_steps[fc_steps.index(step) - 1])
        in_steps.append(step)
        return in_steps

    def _range(self, steps: list[int], fc_steps: list[int]) -> list[int]:
        start, end = steps
        if self.precomputed:
            in_steps = [f"{start}-{end}"]
        elif self.deaccumulation:
            in_steps = [start, end]
        else:
            if self.by:
                fc_steps = [
                    x
                    for x in fc_steps
                    if x in range(fc_steps[0], fc_steps[-1] + 1, self.by)
                ]
            in_steps = fc_steps[fc_steps.index(start) : fc_steps.index(end) + 1]
        return in_steps

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        steps = list(map(int, str(output_request["step"]).split("-")))
        if len(steps) == 1:
            return self._inst_step(int(steps[0]), fc_steps)
        return self._range(steps, fc_steps)


class FcmonthStepDeriver(BaseModel):
    type_: Literal["monthly"] = Field("monthly", alias="type")
    by: Optional[int] = None

    def derive(self, output_request: dict, fc_steps: list[int]) -> list[int]:
        fcmonth = int(output_request["fcmonth"])
        start, end = map(
            int,
            fcmonth_to_steprange(
                datetime.datetime.strptime(str(output_request["date"]), "%Y%m%d"),
                fcmonth,
            ).split("-"),
        )
        if self.by:
            fc_steps = [
                x
                for x in fc_steps
                if x in range(fc_steps[0], fc_steps[-1] + 1, self.by)
            ]
        return fc_steps[fc_steps.index(start) : fc_steps.index(end) + 1]


ForecastStepDeriver = Annotated[
    Union[DefaultStepDeriver, FcmonthStepDeriver],
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
        time = int(fc_request["time"])
        req_steps = fc_request["step"]

        if len(req_steps) == 1:
            assert isinstance(req_steps[0], str)
            req_steps = list(map(int, req_steps[0].split("-")))
        start, end = req_steps[0], req_steps[-1]

        # Find nearest clim window range to real forecast time
        relative_time = start + int(time)
        if time == 12:
            relative_time = start - int(time)

        if end < 240:
            clim_start_steps = [int(x.split("-")[0]) for x in clim_steps]
            nearest = bisect.bisect_right(clim_start_steps, relative_time)
            clim_start = clim_start_steps[nearest - 1]
            if (nearest <= (len(clim_start_steps) - 1)) and (
                (clim_start_steps[nearest] - relative_time)
                < (relative_time - clim_start)
            ):
                clim_start = clim_start_steps[nearest]
            return f"{clim_start}-{clim_start + (end - start)}"
        return f"{start}-{end}"

    @staticmethod
    def _instantaneous(fc_request: dict, clim_steps: list[int]) -> list[int]:
        time = int(fc_request["time"])
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

    def derive(self, fc_request: dict, scheme: str) -> str | list[str]:
        date = datetime.datetime.strptime(str(fc_request["date"]), "%Y%m%d")
        clim_seq = Sequence.from_resource(scheme)
        kwargs = self.model_dump(exclude={"option"})
        clim_date = getattr(clim_seq, self.option)(date.date(), **kwargs)
        if isinstance(clim_date, datetime.date):
            return datetime.datetime.strftime(clim_date, "%Y%m%d")
        return [datetime.datetime.strftime(x, "%Y%m%d") for x in clim_date]
