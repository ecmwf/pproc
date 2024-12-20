import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from earthkit.time.calendar import parse_date, MonthInYear
from earthkit.time.sequence import Sequence, MonthlySequence
from pproc.common.accumulation import Accumulator, Coord, coords_extent
from pproc.common.utils import dict_product
from pproc.common.window import legacy_window_factory


def _default_accumulation_factory(
    config: dict, grib_keys: dict
) -> Iterator[Tuple[str, dict]]:
    for coords in config["coords"]:
        acc_config = config.copy()
        acc_config["coords"] = coords
        min_coord, max_coord = coords_extent(coords)
        name = f"{min_coord}" if min_coord == max_coord else f"{min_coord}-{max_coord}"
        acc_grib_keys = grib_keys.copy()
        acc_grib_keys.update(acc_config.get("grib_keys", {}))
        if acc_grib_keys:
            acc_config["grib_keys"] = acc_grib_keys
        yield name, acc_config


def _to_date(
    arg: Union[str, Tuple[int, int, int], datetime.date, datetime.datetime]
) -> datetime.date:
    if isinstance(arg, datetime.datetime):
        return arg.date()
    if isinstance(arg, datetime.date):
        return arg
    return parse_date(arg)


def _eval_sequence(seq: Sequence, config: dict) -> List[str]:
    if "bracket" in config:
        bracket = config["bracket"]
        ref = _to_date(bracket["date"])
        before = bracket.get("before", 1)
        after = bracket.get("after", before)
        strict = bracket.get("strict", True)
        return [d.strftime("%Y%m%d") for d in seq.bracket(ref, (before, after), strict)]
    elif "range" in config:
        range_ = config["range"]
        from_ = _to_date(range_["from"])
        to = _to_date(range_["to"])
        include_start = range_.get("include_start", True)
        include_end = range_.get("include_end", True)
        return [
            d.strftime("%Y%m%d")
            for d in seq.range(from_, to, include_start, include_end)
        ]
    else:
        raise ValueError(
            "No sequence action found. Currently supported options are 'bracket', 'range'"
        )


def _dateseq_accumulation_factory(
    config: dict, grib_keys: dict
) -> Iterator[Tuple[str, dict]]:
    seq_config = config["sequence"]
    if isinstance(seq_config, dict):
        seq = Sequence.from_dict(seq_config)
    elif isinstance(seq_config, str):
        seq = Sequence.from_resource(seq_config)
    else:
        raise ValueError(f"Invalid sequence definition {seq_config!r}")

    new_config = config.copy()
    new_config["coords"] = [_eval_sequence(seq, cfg) for cfg in config["coords"]]

    return _default_accumulation_factory(new_config, grib_keys)


def _stepseq_monthly(date: str, start: int, end: int, interval: int):
    dt = datetime.datetime.strptime(date, "%Y%m%d") + datetime.timedelta(hours=start)
    seq = MonthlySequence(1)
    start_month = seq.next(dt.date(), strict=False)
    step_start = (start_month - dt.date()).days * 24
    miny = MonthInYear(start_month.year, start_month.month)
    while step_start < end:
        delta = miny.length() * 24
        step_end = step_start + delta

        if step_end > end:
            break

        yield miny, [step_start, step_end, interval]
        miny = miny.next()
        step_start = step_end


def _monthly_config(date: str, config: dict) -> dict:
    config.pop("type", None)
    coords = config.pop("coords")
    start = int(coords["from"])
    end = int(coords["to"])
    interval = int(coords["by"])

    periods = []
    for _, steps in _stepseq_monthly(date, start, end, interval):
        periods.append({"range": steps})
    config = {
        "type": "legacywindow",
        "windows": [
            {
                "window_operation": config.pop("operation", "none"),
                **config,
                "periods": periods,
            }
        ],
    }
    return config


def _stepseq_accumulation_factory(
    config, grib_keys: dict
) -> Iterator[Tuple[str, dict]]:
    seq_config = config.pop("sequence")
    if seq_config["type"] == "monthly":
        new_config = _monthly_config(seq_config["date"], config)
        return legacy_window_factory(new_config, grib_keys)
    raise ValueError(f"Unknown sequence type {seq_config!r}")


def _make_accumulation_configs(
    config: dict, grib_keys: dict
) -> Iterator[Tuple[str, dict]]:
    tp = config.get("type", "default")
    known = {
        "default": _default_accumulation_factory,
        "dateseq": _dateseq_accumulation_factory,
        "stepseq": _stepseq_accumulation_factory,
        "legacywindow": legacy_window_factory,
    }
    factory = known.get(tp)
    if factory is None:
        raise ValueError(f"Unknown accumulation type {tp!r}")
    return factory(config, grib_keys)


class AccumulationManager:
    accumulations: Dict[str, Accumulator]  # accum name -> accumulator
    coords: Dict[
        str, Set[Coord]
    ]  # dimension key -> unique coords for all accumulations

    def __init__(self, accum_configs: Dict[str, Iterator[Tuple[str, dict]]]):
        self.coords = {key: set() for key in accum_configs.keys()}
        self.accumulations = {}
        for i, accum_config in enumerate(dict_product(accum_configs)):
            new_accum = Accumulator.create(accum_config)
            acc_name = f"accum{i}" if new_accum.name is None else new_accum.name
            if acc_name in self.accumulations:
                raise ValueError(f"Duplicate accumulator {acc_name!r}")
            self.accumulations[acc_name] = new_accum
            for dim in new_accum.dims:
                self.coords[dim.key].update(dim.accumulation.coords)

    def sorted_coords(
        self, sortkeys: Dict[str, Callable[[Coord], Any]] = {}
    ) -> Dict[str, List[Coord]]:
        sc = {}
        for key, coords in self.coords.items():
            sortkey = sortkeys.get(key, None)
            if sortkey is None:
                sc[key] = sorted(coords)
            else:
                sc[key] = sorted(coords, key=sortkey)
        return sc

    def feed(
        self, keys: Dict[str, Coord], values: np.ndarray
    ) -> Iterator[Tuple[str, Accumulator]]:
        for name, accum in list(self.accumulations.items()):
            processed = accum.feed(keys, values)
            if processed and accum.is_complete():
                yield name, self.accumulations.pop(name)

    def delete(self, accumulations: List[str]) -> None:
        for name in accumulations:
            del self.accumulations[name]

        for key, coords in self.coords.items():
            for coord in coords.copy():
                if not any(
                    coord in accum[key] for accum in self.accumulations.values()
                ):
                    coords.remove(coord)

    @classmethod
    def create(cls, config: Dict[str, dict], grib_keys: Optional[dict] = None):
        grib_keys = {} if grib_keys is None else grib_keys
        accum_configs = {}
        for key, acc_config in config.items():
            accum_configs[key] = _make_accumulation_configs(acc_config, grib_keys)
        return cls(accum_configs)
