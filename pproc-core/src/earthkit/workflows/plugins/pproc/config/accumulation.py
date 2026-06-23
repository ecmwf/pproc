from typing import Literal, Optional, Union
import datetime

from earthkit.time.calendar import MonthInYear
from earthkit.time.sequence import MonthlySequence

from pydantic import Field, ConfigDict

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel

NumericCoord = int
NumericCoords = Union[list[int], range]

Coord = Union[str, NumericCoord]
Coords = Union[list[str], NumericCoords]


class Default(PProcBaseModel):
    model_config = ConfigDict(validate_by_name=True)

    type_: Literal["default"] = Field("default", alias="type")
    length: Optional[int] = None

    def name(self, coords: Coords) -> str:
        if len(coords) == 0:
            return ""
        end = coords[-1]
        if len(coords) == 1 and (self.length is None or isinstance(coords[0], str)):
            return f"{end}"
        start = coords[0] if self.length is None else int(end) - self.length
        return f"{start}-{end}"


class Monthly(PProcBaseModel):
    model_config = ConfigDict(validate_by_name=True)

    type_: Literal["monthly"] = Field("monthly", alias="type")
    date: str

    def name(self, coords: Coords) -> str:
        if len(coords) == 0:
            return ""
        end = coords[-1]
        fcdate = datetime.datetime.strptime(self.date, "%Y%m%d")
        end = int(end)
        seq = MonthlySequence(1)
        seq.next(fcdate, False)
        this_month = fcdate + datetime.timedelta(hours=end - 1)
        month_length = MonthInYear(this_month.year, this_month.month).length() * 24
        start = end - month_length
        return f"{start}-{end}"
