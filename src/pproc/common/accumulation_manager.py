from typing import Dict, Iterator, List, Set, Tuple

import numpy as np

from pproc.common.accumulation import Accumulator, Coord
from pproc.common.utils import dict_product


class AccumulationManager:
    accumulations: Dict[str, Accumulator]  # accum name -> accumulator
    coords: Dict[
        str, Set[Coord]
    ]  # dimension key -> unique coords for all accumulations

    def __init__(self, config: Dict[str, dict]):
        self.coords = {}
        accum_coords = {}
        for key, acc_config in config.items():
            self.coords[key] = set()
            accum_coords[key] = acc_config["coords"]
        self.accumulations = {}
        for i, coords in enumerate(dict_product(accum_coords)):
            new_config = {}
            for key, acc_config in config.items():
                new_config[key] = acc_config.copy()
                new_config[key]["coords"] = coords[key]
            new_accum = Accumulator.create(new_config)
            # XXX: might be worth having a more explicit name to print
            self.accumulations[f"accum{i}"] = new_accum
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
