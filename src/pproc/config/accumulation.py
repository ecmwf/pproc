from pydantic import BaseModel, Field, BeforeValidator, ConfigDict
from typing import Literal, Union, Annotated, Tuple, Iterator, Optional, List, Any
import datetime

from earthkit.time.calendar import parse_date
from earthkit.time.sequence import Sequence

from pproc.common.stepseq import stepseq_ranges, stepseq_monthly
from pproc.common.accumulation import coords_extent
from pproc.common.window import legacy_window_factory


class LegacyStepConfig(BaseModel):
    start_step: int
    end_step: int
    interval: int
    range: Optional[int] = None

    def values(self) -> list:
        if self.range is None:
            return list(range(self.start_step, self.end_step + 1, self.interval))
        return [
            f"{start}-{start + self.range - 1}"
            for start in range(
                self.start_step, self.end_step - self.range + 1, self.interval
            )
        ]


class LegacyPeriodConfig(BaseModel):
    range: Union[list[int], str]

    def values(self) -> Iterator:
        by = self.range[2] if len(self.range) == 3 else 1
        return range(self.range[0], self.range[1] + 1, by)

    def name(self) -> str:
        return (
            f"{self.range[0]}-{self.range[1]}"
            if len(self.range) == 2
            else str(self.range[0])
        )


class StepRanges(BaseModel):
    type_: Literal["ranges"] = Field("ranges", alias="type")
    to: int
    from_: int = Field(default=0, alias="from")
    interval: int = 1
    by: int = 1
    width: int = 0

    def coords(self) -> list[list[int]]:
        return [
            x
            for x in stepseq_ranges(
                self.from_, self.to, self.width, self.interval, self.by
            )
        ]

    def periods(self) -> list[LegacyPeriodConfig]:
        return [
            LegacyPeriodConfig(range=[x[0], x[0]])
            if len(x) == 1
            else LegacyPeriodConfig(range=[x[0], x[-1], x[1] - x[0]])
            for x in stepseq_ranges(
                self.from_, self.to, self.width, self.interval, self.by
            )
        ]


class StepMonthly(BaseModel):
    type_: Literal["monthly"] = Field("monthly", alias="type")
    date: str
    from_: int = Field(default=0, alias="from")
    to: int
    by: int = 1

    def coords(self) -> list[list[int]]:
        return [x for x in stepseq_monthly(self.date, self.from_, self.to, self.by)]

    def periods(self) -> list[LegacyPeriodConfig]:
        return [
            LegacyPeriodConfig(range=[x[0], x[-1], x[1] - x[0]])
            for x in stepseq_monthly(self.date, self.from_, self.to, self.by)
        ]


def _to_periods(periods: Any) -> List[LegacyPeriodConfig]:
    if not isinstance(periods, dict):
        return periods
    if periods["type"] == "ranges":
        return StepRanges(**periods).periods()
    if periods["type"] == "monthly":
        return StepMonthly(**periods).periods()
    raise ValueError(f"Invalid period type {periods['type']!r}")


class LegacyWindowConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    window_operation: Optional[str] = None
    include_start_step: bool = False
    deaccumulate: bool = False
    grib_set: dict = {}
    periods: Annotated[list[LegacyPeriodConfig], BeforeValidator(_to_periods)] = []

    def unique_coords(self):
        coords = set()
        for period in self.periods:
            coords.update(
                period.name()
                if self.window_operation == "precomputed"
                else period.values()
            )
        coords = list(coords)
        coords.sort()
        return coords


def _to_date(
    arg: Union[str, Tuple[int, int, int], datetime.date, datetime.datetime]
) -> datetime.date:
    if isinstance(arg, datetime.datetime):
        return arg.date()
    if isinstance(arg, datetime.date):
        return arg
    return parse_date(arg)


def _dateseq_factory(seq_config: Union[str, dict]) -> Sequence:
    if isinstance(seq_config, dict):
        seq = Sequence.from_dict(seq_config)
    elif isinstance(seq_config, str):
        seq = Sequence.from_resource(seq_config)
    else:
        raise ValueError(f"Invalid sequence definition {seq_config!r}")
    return seq


class DateBracket(BaseModel):
    date: Annotated[datetime.date, BeforeValidator(_to_date)]
    before: int = 1
    after: int = 1
    strict: bool = True

    def coords(self, seq: Sequence) -> list[datetime.date]:
        return [
            d.strftime("%Y%m%d")
            for d in seq.bracket(self.date, (self.before, self.after), self.strict)
        ]


class DateRange(BaseModel):
    from_: Annotated[datetime.date, BeforeValidator(_to_date)] = Field(alias="from")
    to: Annotated[datetime.date, BeforeValidator(_to_date)]
    include_start: bool = True
    include_end: bool = True

    def coords(self, seq: Sequence) -> list[datetime.date]:
        return [
            d.strftime("%Y%m%d")
            for d in seq.range(
                self.from_, self.to, self.include_start, self.include_end
            )
        ]


def _to_date_range(coords: Any) -> Any:
    for i, coord in enumerate(coords):
        if not isinstance(coord, dict):
            continue

        if "bracket" in coord:
            coords[i] = DateBracket(**coord["bracket"])
        elif "range" in coord:
            coords[i] = DateRange(**coord["range"])
        else:
            raise ValueError(f"Invalid date sequence type {coord!r}")
    return coords


class BaseAccumulation(BaseModel):
    operation: Optional[str] = None
    grib_keys: dict = {}
    sequential: bool = False


class DefaultAccumulation(BaseAccumulation):
    model_config = ConfigDict(extra="allow")

    type_: Literal["default"] = Field("default", alias="type")
    coords: list[Union[List[int | str], dict]] = []

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        for coords in self.coords:
            acc_config = self.model_dump(by_alias=True, exclude_none=True)
            acc_config["coords"] = coords
            min_coord, max_coord = coords_extent(coords)
            name = (
                f"{min_coord}" if min_coord == max_coord else f"{min_coord}-{max_coord}"
            )
            acc_grib_keys = metadata.copy()
            acc_grib_keys.update(acc_config.get("grib_keys", {}))
            if acc_grib_keys:
                acc_config["grib_keys"] = acc_grib_keys
            yield name, acc_config

    def unique_coords(self):
        coords = set()
        for coord in self.coords:
            if isinstance(coord, list):
                coords.update(coord)
            elif isinstance(coord, dict):
                coords.update(
                    range(coords.get("from", 0), coords["to"] + 1), coords.get("by", 1)
                )
        coords = list(coords)
        coords.sort()
        return coords


class StepSeqAccumulation(BaseAccumulation):
    model_config = ConfigDict(extra="allow")

    type_: Literal["stepseq"] = Field("stepseq", alias="type")
    sequence: Annotated[
        Union[StepRanges, StepMonthly],
        Field(discriminator="type_"),
    ]

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        return DefaultAccumulation(
            **self.model_dump(by_alias=True, exclude={"type_", "sequence"}),
            coords=self.sequence.coords(),
        ).make_configs(metadata)

    def unique_coords(self):
        coords = list(set.union(*self.sequence.coords()))
        coords.sort()
        return coords


class DateSeqAccumulation(BaseAccumulation):
    type_: Literal["dateseq"] = Field("dateseq", alias="type")
    sequence: Union[dict, str]
    coords: Annotated[
        list[Union[DateBracket, DateRange]], BeforeValidator(_to_date_range)
    ] = []

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        seq = _dateseq_factory(self.sequence)
        return DefaultAccumulation(
            **self.model_dump(by_alias=True, exclude={"type_", "sequence", "coords"}),
            coords=[c.coords(seq) for c in self.coords],
        ).make_configs(metadata)

    def unique_coords(self):
        seq = _dateseq_factory(self.sequence)
        coords = list(set.union(*(coord.coords(seq) for coord in self.coords)))
        coords.sort()
        return coords


class LegacyStepAccumulation(BaseModel):
    type_: Literal["legacywindow"] = Field("legacywindow", alias="type")
    windows: list[LegacyWindowConfig] = []
    std_anomaly_windows: Optional[list[LegacyWindowConfig]] = None
    steps: Optional[list[LegacyStepConfig]] = None

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        return legacy_window_factory(
            self.model_dump(by_alias=True, exclude={"type"}, exclude_none=True),
            metadata,
        )

    def unique_coords(self):
        coords = set()
        for window in self.windows + (self.std_anomaly_windows or []):
            coords.update(window.unique_coords())

        if self.steps:
            steps = set.union(*(step.values() for step in self.steps))
            coords = coords.intersection(steps)

        coords = list(coords)
        coords.sort()
        return coords


AccumulationConfig = Union[
    LegacyStepAccumulation,
    DefaultAccumulation,
    StepSeqAccumulation,
    DateSeqAccumulation,
]


def accumulation_factory(config: dict) -> AccumulationConfig:
    type = config.get("type", "default")
    acc_cls = {
        "default": DefaultAccumulation,
        "stepseq": StepSeqAccumulation,
        "dateseq": DateSeqAccumulation,
        "legacywindow": LegacyStepAccumulation,
    }
    if type not in acc_cls:
        raise ValueError(f"Unknown accumulation type {type!r}")
    return acc_cls[type](**config)
