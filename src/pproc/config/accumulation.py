# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pydantic import BaseModel, Field, BeforeValidator, ConfigDict, Tag, Discriminator
from typing import Literal, Union, Annotated, Tuple, Iterator, Optional, List, Any
from typing_extensions import Self
import datetime

from earthkit.time.calendar import parse_date
from earthkit.time.sequence import Sequence

from pproc.common.stepseq import stepseq_ranges, stepseq_monthly
from pproc.common.accumulation import coords_extent
from pproc.common.window import legacy_window_factory
from pproc.config.utils import extract_mars, _get


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


class StepMonthly(BaseModel):
    type_: Literal["monthly"] = Field("monthly", alias="type")
    date: str
    from_: int = Field(default=0, alias="from")
    to: int
    by: int = 1

    def coords(self) -> list[list[int]]:
        return [x for x in stepseq_monthly(self.date, self.from_, self.to, self.by)]


def _to_coords(coords: Any) -> list[list[int]]:
    if not isinstance(coords, dict):
        return coords
    if coords["type"] == "ranges":
        return StepRanges(**coords).coords()
    if coords["type"] == "monthly":
        return StepMonthly(**coords).coords()
    raise ValueError(f"Invalid period type {coords['type']!r}")


class LegacyWindowConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    operation: Optional[str] = None
    include_start: bool = False
    deaccumulate: bool = False
    metadata: dict = {}
    coords: Annotated[
        list[Union[List[int | str], dict]], BeforeValidator(_to_coords)
    ] = []

    def unique_coords(self):
        coords = set()
        for coord in self.coords:
            if isinstance(coord, list):
                update = coord
            elif isinstance(coord, dict):
                update = list(
                    range(coord.get("from", 0), coord["to"] + 1, coord.get("by", 1))
                )
            if len(update) > 1 and not (self.include_start or self.deaccumulate):
                update = update[1:]
            coords.update(update)
        coords = list(coords)
        coords.sort()
        return coords

    def out_mars(self, dim: str) -> dict:
        base = extract_mars(self.metadata)
        if self.operation is not None and dim != "step":
            return base

        base[dim] = []
        for coord in self.coords:
            if isinstance(coord, list):
                base[dim].append(
                    coord[0] if len(coord) == 1 else f"{coord[0]}-{coord[-1]}"
                )
            elif isinstance(coord, dict):
                base[dim].append(f"{coord['from']}-{coord['to']}")
        if hasattr(self, "thresholds"):
            base["param"] = []
            for thr in self.thresholds:
                mars_keys = extract_mars(thr.get("metadata", {}))
                mars_keys.pop("param", None)
                assert (
                    len(mars_keys) == 0
                ), "Mars metadata keys can not be set in threshold metadata"
                base["param"].append(thr["out_paramid"])
        return base


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
    metadata: dict = {}
    sequential: bool = False


class DefaultAccumulation(BaseAccumulation):
    type_: Literal["default"] = Field("default", alias="type")
    coords: list[Union[List[int | str], dict]] = []
    deaccumulate: bool = False

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        for coords in self.coords:
            acc_config = self.model_dump(by_alias=True, exclude_none=True)
            acc_config["coords"] = coords
            min_coord, max_coord = coords_extent(coords)
            name = (
                f"{min_coord}" if min_coord == max_coord else f"{min_coord}-{max_coord}"
            )
            acc_grib_keys = metadata.copy()
            acc_grib_keys.update(acc_config.get("metadata", {}))
            if acc_grib_keys:
                acc_config["metadata"] = acc_grib_keys
            yield name, acc_config

    def unique_coords(self):
        coords = set()
        for coord in self.coords:
            if isinstance(coord, list):
                coords.update(coord)
            elif isinstance(coord, dict):
                coords.update(
                    range(coord.get("from", 0), coord["to"] + 1, coord.get("by", 1))
                )
        coords = list(coords)
        coords.sort()
        return coords

    def out_mars(self, dim: str) -> list[dict]:
        base = extract_mars(self.metadata)
        if self.operation is not None and dim != "step":
            return [base]

        base[dim] = []
        for coord in self.coords:
            if isinstance(coord, list):
                base[dim].append(
                    coord[0] if len(coord) == 1 else f"{coord[0]}-{coord[-1]}"
                )
            elif isinstance(coord, dict):
                base[dim].append(f"{coord['from']}-{coord['to']}")
        return [base]

    def merge(self, other: Self) -> Self:
        if not isinstance(other, DefaultAccumulation):
            raise ValueError("Merge only possible with other DefaultAccumulation")

        current_config = self.model_dump(by_alias=True, exclude={"coords"})
        other_config = other.model_dump(by_alias=True, exclude={"coords"})
        if current_config != other_config:
            raise ValueError(
                "Merging of two DefaultAccumulations requires them to be the same, except for coords"
            )
        new_coords = self.coords + [x for x in other.coords if x not in self.coords]
        return type(self)(**current_config, coords=new_coords)


class StepSeqAccumulation(BaseAccumulation):
    type_: Literal["stepseq"] = Field("stepseq", alias="type")
    sequence: Annotated[
        Union[StepRanges, StepMonthly],
        Field(discriminator="type_"),
    ]
    deaccumulate: bool = False

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        return DefaultAccumulation(
            **self.model_dump(by_alias=True, exclude={"type_", "sequence"}),
            coords=self.sequence.coords(),
        ).make_configs(metadata)

    def unique_coords(self):
        coords = list(set.union(*self.sequence.coords()))
        coords.sort()
        return coords

    def out_mars(self, dim: str) -> list[dict]:
        return [
            {
                dim: [
                    x if len(x) == 1 else f"{x[0]}-{x[-1]}"
                    for x in self.sequence.coords()
                ],
                **extract_mars(self.metadata),
            }
        ]

    def merge(self, other: Self) -> Self:
        raise NotImplementedError("Merging of StepSeqAccumulation not implemented")


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

    def out_mars(self, dim: str):
        base = extract_mars(self.metadata)
        if self.operation is not None:
            return [base]
        seq = _dateseq_factory(self.sequence)
        dates = base.setdefault(dim, [])
        for c in self.coords:
            date_range = c.coords(seq)
            dates.append(f"{date_range[0]}-{date_range[-1]}")
        return [base]

    def merge(self, other: Self) -> Self:
        raise NotImplementedError("Merging of DateSeqAccumulation not implemented")


class LegacyStepAccumulation(BaseModel):
    type_: Literal["legacywindow"] = Field("legacywindow", alias="type")
    windows: list[LegacyWindowConfig] = []
    std_anomaly_windows: Optional[list[LegacyWindowConfig]] = None

    def make_configs(self, metadata: dict) -> Iterator[Tuple[str, dict]]:
        return legacy_window_factory(
            self.model_dump(by_alias=True, exclude={"type"}, exclude_none=True),
            metadata,
        )

    def unique_coords(self):
        coords = set()
        for window in self.windows + (self.std_anomaly_windows or []):
            coords.update(window.unique_coords())

        coords = list(coords)
        coords.sort()
        return coords

    def out_mars(self, dim: str) -> list[dict]:
        return [window.out_mars(dim) for window in self.windows]

    def merge(self, other: Self) -> Self:
        if not isinstance(other, LegacyStepAccumulation):
            raise ValueError("Merge only possible with other LegacyStepAccumulation")
        current = self.model_copy(deep=True)

        for wtype in ["windows", "std_anomaly_windows"]:
            current_windows = getattr(current, wtype, None)
            other_windows = getattr(other, wtype, None)
            if current_windows is None:
                if other_windows is not None:
                    setattr(current, wtype, other_windows)
                continue

            for w2 in other_windows:
                matched = False
                for w1 in current_windows:
                    w1_config = w1.model_dump(by_alias=True, exclude={"coords"})
                    w2_config = w2.model_dump(by_alias=True, exclude={"coords"})
                    if w1_config == w2_config:
                        w1.coords = w1.coords + [
                            x for x in w2.coords if x not in w1.coords
                        ]
                        matched = True
                        break
                if not matched:
                    current_windows.append(w2)
        return current.model_validate(current)


def accumulation_discriminator(config: dict) -> str:
    return _get(config, "type", "default")


AccumulationConfig = Annotated[
    Union[
        Annotated[DefaultAccumulation, Tag("default")],
        Annotated[LegacyStepAccumulation, Tag("legacywindow")],
        Annotated[StepSeqAccumulation, Tag("stepseq")],
        Annotated[DateSeqAccumulation, Tag("dateseq")],
    ],
    Discriminator(accumulation_discriminator),
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
