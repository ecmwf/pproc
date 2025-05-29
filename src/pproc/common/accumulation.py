# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABCMeta, abstractmethod
import copy
from dataclasses import dataclass
from math import prod
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike


NumericCoord = int
NumericCoords = Union[List[int], range]

Coord = Union[str, NumericCoord]
Coords = Union[List[str], NumericCoords]


class Accumulation(metaclass=ABCMeta):
    values: Optional[np.ndarray]

    def __init__(
        self,
        coords: Coords,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        self.coords = coords
        self.sequential = sequential
        self._grib_keys = {} if metadata is None else metadata.copy()
        self.reset(initial=True)

    def __len__(self) -> int:
        return len(self.coords)

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
            if self.sequential:
                self.todo.difference_update([t for t in self.todo if t < coord])
        return processed

    def is_complete(self) -> bool:
        return not self.todo

    def get_values(self) -> Optional[np.ndarray]:
        return self.values

    def grib_keys(self) -> dict:
        return self._grib_keys

    @abstractmethod
    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> "Accumulation":
        raise NotImplementedError


class SimpleAccumulation(Accumulation):
    def __init__(
        self,
        operation: str,
        coords: Coords,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        super().__init__(coords, sequential, metadata)
        self.operation = getattr(np, operation)

    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        assert self.values is not None
        self.operation(self.values, values, out=self.values)
        return True

    @classmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        if operation == "sum":
            operation = "add"
        return cls(operation, coords, sequential=sequential, metadata=metadata)


class Integral(SimpleAccumulation):
    def __init__(
        self,
        init: int,
        coords: NumericCoords,
        metadata: Optional[dict] = None,
    ):
        self.init = init
        super().__init__("add", coords, sequential=True, metadata=metadata)

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
        cls,
        operation: str,
        coords: NumericCoords,
        config: dict,
        sequential: bool = True,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        if isinstance(coords, range):
            init = coords.start
            coords = range(coords.start + coords.step, coords.stop, coords.step)
        else:
            coords = coords.copy()
            init = coords.pop(0)
        return cls(init, coords, metadata=metadata)


class Difference(Accumulation):
    def __init__(
        self,
        coords: Coords,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        assert len(coords) in {1, 2}
        super().__init__(coords, sequential, metadata)

    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        assert self.values is not None
        assert coord == self.coords[-1]
        np.subtract(values, self.values, out=self.values)
        return True

    @classmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        return cls(coords, sequential=sequential, metadata=metadata)


class DifferenceRate(Difference):
    def __init__(
        self,
        coords: NumericCoords,
        factor: float = 1.0,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        super().__init__(coords, sequential, metadata)
        length = self.coords[-1] - (self.coords[0] if len(self.coords) == 2 else 0)
        self.factor = factor * length

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        return self.values / self.factor

    @classmethod
    def create(
        cls,
        operation: str,
        coords: NumericCoords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        return cls(
            coords,
            factor=config.get("factor", 1.0),
            sequential=sequential,
            metadata=metadata,
        )


class Mean(SimpleAccumulation):
    def __init__(
        self,
        coords: Coords,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        super().__init__("add", coords, sequential, metadata)

    def reset(self, initial: bool = False) -> None:
        super().reset(initial)
        self.count = 0 if self.sequential else len(self.coords)

    def feed(self, coord: Coord, values: np.ndarray) -> bool:
        processed = super().feed(coord, values)
        if self.sequential and processed:
            self.count += 1
        return processed

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        return self.values / self.count

    @classmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        return cls(coords, sequential=sequential, metadata=metadata)


class WeightedMean(Integral):
    def __init__(
        self,
        init: int,
        coords: NumericCoords,
        metadata: Optional[dict] = None,
    ):
        super().__init__(init, coords, metadata)
        self.length = coords[-1] - init

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        return self.values / self.length


class Histogram(SimpleAccumulation):
    def __init__(
        self,
        coords: Coords,
        bins: Union[np.ndarray, List[float]],
        mod: Optional[float] = None,
        normalise: bool = True,
        scale_out: Optional[float] = None,
        dtype: DTypeLike = np.float32,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        super().__init__("add", coords, sequential, metadata)
        self.bins = np.asarray(bins)
        self.mod = mod
        self.normalise = normalise
        self.scale_out = scale_out
        self.out_dtype = dtype

    def feed(self, coord: Coord, values: np.ndarray) -> bool:
        if coord not in self.todo:
            return False

        nbins = len(self.bins) - 1
        if self.mod is not None:
            values %= self.mod
        ind = np.digitize(values, self.bins) - 1
        if self.mod is not None:
            ind[ind < 0] = nbins - 1
            ind[ind >= nbins] = 0

        hist_values = np.zeros((nbins,) + values.shape, dtype=np.int64)
        for i in range(nbins):
            hist_values[i, ind == i] += 1
        return super().feed(coord, hist_values)

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None:
            return None
        values = self.values.astype(self.out_dtype)
        if self.normalise:
            values /= values.sum(axis=0)
        if self.scale_out is not None:
            values *= self.scale_out
        return values

    @classmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        return cls(
            coords,
            bins=np.asarray(config["bins"]),
            mod=config.get("mod"),
            normalise=config.get("normalise", True),
            scale_out=config.get("scale_out"),
            sequential=sequential,
            metadata=metadata,
        )


class Aggregation(Accumulation):
    def __init__(
        self,
        coords: Coords,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        super().__init__(coords, sequential, metadata)
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

    def get_values(self) -> Optional[np.ndarray]:
        if self.values is None or self.values.shape[0] != 1:
            return self.values
        return self.values[0, ...]

    @classmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> Accumulation:
        return cls(coords, sequential=sequential, metadata=metadata)


class StandardDeviation(Mean):
    def __init__(
        self,
        coords: Coords,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ):
        super().__init__(coords, sequential, metadata)
        self.sumsq = None

    def reset(self, initial: bool = False) -> None:
        super().reset(initial)
        self.sumsq = None

    def feed(self, coord: Coord, values: np.ndarray) -> bool:
        processed = super().feed(coord, values)
        if processed:
            if self.sumsq is None:
                self.sumsq = values**2
            else:
                np.add(self.sumsq, values**2, out=self.sumsq)
        return processed

    def get_values(self) -> Optional[np.ndarray]:
        mean = super().get_values()
        if mean is None:
            return None
        return np.sqrt(self.sumsq / self.count - mean**2)


class DeaccumulationWrapper(Accumulation):
    def __init__(self, accumulation: Accumulation):
        self.coords = copy.deepcopy(accumulation.coords)
        # Remove first coord from accumulation
        accumulation.coords = list(accumulation.coords)
        accumulation.coords.pop(0)
        self.acc = accumulation
        self.sequential = accumulation.sequential
        self.reset(initial=True)

    def reset(self, initial: bool = False) -> None:
        super().reset(initial)
        self.acc.reset(initial)

    def is_complete(self) -> bool:
        return self.acc.is_complete()

    def get_values(self) -> Optional[np.ndarray]:
        return self.acc.get_values()

    def grib_keys(self) -> dict:
        return self.acc.grib_keys()

    def combine(self, coord: Coord, values: np.ndarray) -> bool:
        processed = self.acc.feed(coord, values - self.values)
        if processed:
            self.values = values.copy()
        return processed

    @classmethod
    def create(
        cls,
        operation: str,
        coords: Coords,
        config: dict,
        sequential: bool = False,
        metadata: Optional[dict] = None,
    ) -> "Accumulation":
        raise NotImplementedError


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


def coords_extent(config: Union[dict, list]) -> Tuple[Coord, Coord]:
    if isinstance(config, dict):
        return config.get("from", 0), config["to"]
    return min(config), max(config)


def create_accumulation(config: dict) -> Accumulation:
    op = config.get("operation", "aggregation")
    coords = convert_coords(config["coords"])
    sequential = config.get("sequential", False)
    metadata = config.get("metadata", {})
    known = {
        "sum": SimpleAccumulation,
        "minimum": SimpleAccumulation,
        "maximum": SimpleAccumulation,
        "integral": Integral,
        "difference": Difference,
        "difference_rate": DifferenceRate,
        "mean": Mean,
        "weighted_mean": WeightedMean,
        "histogram": Histogram,
        "aggregation": Aggregation,
        "standard_deviation": StandardDeviation,
    }
    cls = known.get(op)
    if cls is None:
        raise ValueError(f"Unknown accumulation {op!r}")
    acc = cls.create(op, coords, config, sequential=sequential, metadata=metadata)
    if config.get("deaccumulate", False):
        return DeaccumulationWrapper(acc)
    return acc


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
    name: Optional[str]
    values: Optional[np.ndarray]

    def __init__(self, dims: DimensionsLike, name: Optional[str] = None):
        self.dims = convert_dims(dims)
        self.name = name
        self.values = None

    def __len__(self) -> int:
        return prod(len(dim.accumulation) for dim in self.dims)

    def __contains__(self, keys: Dict[str, Coord]) -> bool:
        return all(keys[dim.key] in dim.accumulation for dim in self.dims)

    def __getitem__(self, key: str) -> Accumulation:
        for dim in self.dims:
            if dim.key == key:
                return dim.accumulation
        raise KeyError(key)

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
        keys = {}
        for dim in self.dims:
            keys.update(dim.accumulation.grib_keys())
        return keys

    @classmethod
    def create(cls, config: dict) -> "Accumulator":
        names = {}
        dims = []
        for key, acc_cfg in config.items():
            if isinstance(acc_cfg, tuple):
                name = acc_cfg[0]
                if name:
                    names[key] = name
                acc_cfg = acc_cfg[1]
            dims.append((key, create_accumulation(acc_cfg)))
        if not names:
            name = None
        elif len(dims) == 1:
            name = next(str(v) for v in names.values())
        else:
            name = ":".join(f"{k}_{v}" for k, v in names.items())
        return cls(dims, name)
