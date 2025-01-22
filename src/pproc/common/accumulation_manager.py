from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np

from pproc.common.accumulation import Accumulator, Coord
from pproc.common.utils import dict_product
from pproc.config.accumulation import accumulation_factory


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
            if isinstance(acc_config, dict):
                acc_config = accumulation_factory(acc_config)
            accum_configs[key] = acc_config.make_configs(grib_keys)
        return cls(accum_configs)
