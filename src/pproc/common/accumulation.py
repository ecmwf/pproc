from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np


NumericCoord = int
NumericCoords = Union[List[int], range]

Coord = Union[str, NumericCoord]
Coords = Union[List[str], NumericCoords]


class Accumulation(metaclass=ABCMeta):
    values: Optional[np.ndarray]

    def __init__(self, coords: Coords):
        self.coords = coords
        self.reset(initial=True)

    def __contains__(self, coord: Coord):
        return coord in self.coords

    def reset(self, initial: bool = False) -> None:
        self.todo = set(self.coords)
        self.values = None

    def feed(self, coord: Coord, values: np.ndarray) -> bool:
        if coord not in self.todo:
            return False
        processed = True
        if self.values is None:
            self.values = values.copy()
        else:
            processed = self.combine(coord, values)
        if processed:
            self.todo.remove(coord)
        return processed

    def is_complete(self) -> bool:
        return not self.todo

    def get_values(self) -> Optional[np.ndarray]:
        return self.values

    def grib_keys(self) -> dict:
        pass  # FIXME

    @abstractmethod
    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, operation: str, coords: Coords, config: dict) -> "Accumulation":
        raise NotImplementedError


class SimpleAccumulation(Accumulation):
    def __init__(self, operation: str, coords: Coords):
        super().__init__(coords)
        self.operation = getattr(np, operation)

    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        assert self.values is not None
        self.operation(self.values, values, out=self.values)
        return True

    @classmethod
    def create(cls, operation: str, coords: Coords, config: dict) -> Accumulation:
        if operation == "sum":
            operation = "add"
        return cls(operation, coords)


class Integral(SimpleAccumulation):
    def __init__(self, init: int, coords: NumericCoords):
        self.init = init
        super().__init__("add", coords)

    def reset(self, initial: bool = False) -> None:
        super().reset(initial)
        self.previous = self.init

    def feed(self, coord: NumericCoord, values: np.ndarray) -> bool:
        weight = coord - self.previous
        processed = super().feed(coord, weight * values)
        if processed:
            self.previous = coord
        return processed

    @classmethod
    def create(
        cls, operation: str, coords: NumericCoords, config: dict
    ) -> Accumulation:
        if isinstance(coords, range):
            init = coords.start
            coords = range(coords.start + coords.step, coords.stop, coords.step)
        else:
            coords = coords.copy()
            init = coords.pop(0)
        return cls(init, coords)


class Difference(Accumulation):
    def __init__(self, coords: Coords):
        assert len(coords) == 2
        super().__init__(coords)

    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        assert self.values is not None
        np.subtract(values, self.values, out=self.values)
        return True

    @classmethod
    def create(cls, operation: str, coords: Coords, config: dict) -> Accumulation:
        return cls(coords)


class DifferenceRate(Difference):
    def __init__(self, coords: NumericCoords, factor: float = 1.0):
        super().__init__(coords)
        self.factor = factor

    def combine(self, coord: NumericCoord, values: np.ndarray) -> bool:
        assert coord == self.coords[1]
        length = self.coords[1] - self.coords[0]
        processed = super().combine(coord, values)
        if processed:
            self.values /= self.factor * length
        return processed

    @classmethod
    def create(
        cls, operation: str, coords: NumericCoords, config: dict
    ) -> Accumulation:
        return cls(coords, factor=config.get("factor", 1.0))


class Mean(SimpleAccumulation):
    def __init__(self, coords: Coords):
        super().__init__("add", coords)

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        return self.values / len(self.coords)

    @classmethod
    def create(cls, operation: str, coords: Coords, config: dict) -> Accumulation:
        return cls(coords)


class WeightedMean(Integral):
    def __init__(self, init: int, coords: NumericCoords):
        super().__init__(init, coords)
        self.length = coords[-1] - init

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        return self.values / self.length


class Aggregation(Accumulation):
    def __init__(self, coords: Coords):
        super().__init__(coords)
        self.lookup = {k: i for i, k in enumerate(coords)}

    def feed(self, coord: Coord, values: np.ndarray) -> bool:
        if coord not in self.todo:
            return False
        if self.values is None:
            self.values = np.zeros(
                (len(self.lookup),) + values.shape, dtype=values.dtype
            )
        return super().feed(coord, values)

    def combine(self, coord: Coord, values: np.ndarray) -> None:
        i = self.lookup.get(coord)
        if i is None:
            return False
        self.values[i, ...] = values
        return True

    @classmethod
    def create(cls, operation: str, coords: Coords, config: dict) -> Accumulation:
        return cls(coords)


def convert_range(config: dict) -> range:
    r_from = config.get("from", 0)
    r_to = config.get("to")
    r_by = config.get("by", 1)
    if r_to is None:
        raise ValueError("Ranges must set at least 'to'")
    return range(r_from, r_to + 1, r_by)


def convert_coords(config: Union[dict, list]) -> Coords:
    if isinstance(config, dict):
        return convert_range(config)
    coords = []
    for c in config:
        if isinstance(c, dict):
            coords.extend(convert_range(c))
        else:
            coords.append(c)
    return coords


def create_accumulation(config: dict) -> Accumulation:
    op = config.get("operation", "aggregation")
    coords = convert_coords(config["coords"])
    known = {
        "sum": SimpleAccumulation,
        "minimum": SimpleAccumulation,
        "maximum": SimpleAccumulation,
        "integral": Integral,
        "difference": Difference,
        "difference_rate": DifferenceRate,
        "mean": Mean,
        "weighted_mean": WeightedMean,
        "aggregation": Aggregation,
    }
    cls = known.get(op)
    if cls is None:
        raise ValueError(f"Unknown accumulation {op!r}")
    return cls.create(op, coords, config)


@dataclass
class Dimension:
    key: str
    accumulation: Accumulation


DimensionsLike = Union[
    Iterable[Union[Dimension, Tuple[str, Accumulation]]], Dict[str, Accumulation]
]


def convert_dim(dim: Union[Dimension, Tuple[str, Accumulation]]) -> Dimension:
    if isinstance(dim, Dimension):
        return dim
    key, accum = dim
    return Dimension(key, accum)


def convert_dims(dims: DimensionsLike) -> List[Dimension]:
    if isinstance(dims, dict):
        return [Dimension(key, accum) for key, accum in dims.items()]
    return [convert_dim(dim) for dim in dims]


class Accumulator:
    values: Optional[np.ndarray]

    def __init__(self, dims: DimensionsLike):
        self.dims = convert_dims(dims)
        self.values = None

    def __contains__(self, keys: Dict[str, Coord]) -> bool:
        return all(keys[dim.key] in dim.accumulation for dim in self.dims)

    def feed(self, keys: Dict[str, Coord], values: np.ndarray) -> bool:
        if keys not in self:
            return False

        for dim in self.dims[::-1]:
            if dim.accumulation.is_complete():
                dim.accumulation.reset()

            processed = dim.accumulation.feed(keys[dim.key], values)
            assert processed

            if dim.accumulation.is_complete():
                values = dim.accumulation.get_values()
                assert values is not None
            else:
                return True

        self.values = values
        return True

    def is_complete(self) -> bool:
        if not self.dims:
            return True
        return self.dims[0].accumulation.is_complete()

    def grib_keys(self) -> dict:
        pass  # FIXME

    @classmethod
    def create(cls, config: dict) -> "Accumulator":
        return cls(
            [(key, create_accumulation(acc_cfg)) for key, acc_cfg in config.items()]
        )
