from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np

from pproc.common.accumulation import Accumulator, Coord
from pproc.common.utils import dict_product
from pproc.common.window import legacy_window_factory


def _default_accumulation_factory(config: dict, grib_keys: dict) -> Iterator[Tuple[str, dict]]:
    for coords in config["coords"]:
        acc_config = config.copy()
        acc_config["coords"] = coords
        acc_grib_keys = grib_keys.copy()
        acc_grib_keys.update(acc_config.get("grib_keys", {}))
        if acc_grib_keys:
            acc_config["grib_keys"] = acc_grib_keys
        yield "", acc_config


def _make_accumulation_configs(config: dict, grib_keys: dict) -> Iterator[Tuple[str, dict]]:
    tp = config.get("type", "default")
    known = {
        "default": _default_accumulation_factory,
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

    def __init__(self, config: Dict[str, dict], grib_keys: Optional[dict] = None):
        self.coords = {}
        accum_configs = {}
        grib_keys = {} if grib_keys is None else grib_keys
        for key, acc_config in config.items():
            self.coords[key] = set()
            accum_configs[key] = _make_accumulation_configs(acc_config, grib_keys)
        self.accumulations = {}
        for i, accum_config in enumerate(dict_product(accum_configs)):
            new_accum = Accumulator.create(accum_config)
            acc_name = f"accum{i}" if new_accum.name is None else new_accum.name
            if acc_name in self.accumulations:
                raise ValueError(f"Duplicate accumulator {acc_name!r}")
            self.accumulations[acc_name] = new_accum
            for dim in new_accum.dims:
                self.coords[dim.key].update(dim.accumulation.coords)

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
