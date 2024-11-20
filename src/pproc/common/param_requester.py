from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import eccodes
import numexpr
import numpy as np
from earthkit.meteo.wind import direction
from pydantic import BaseModel

from pproc.common.dataset import open_multi_dataset
from pproc.common.io import missing_to_nan
from pproc.common.steps import AnyStep
from pproc.config.base import Members, SourceConfig, SourceModel

IndexFunc = Callable[[eccodes.GRIBMessage], int]


def expand(request: dict, dim: str):
    coords = request.pop(dim, [])
    if not isinstance(coords, list):
        coords = [coords]
    for coord in coords:
        yield {**request, dim: coord}


def read_ensemble(
    source: SourceConfig,
    total: int,
    dtype=np.float32,
    index_func: Optional[IndexFunc] = None,
    **kwargs,
) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
    """Read GRIB data as a single array, in arbitrary order

    Parameters
    ----------
    sources: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    total: int
        Number of fields to expect
    dtype: numpy data type
        Data type for the result array (default float32)
    index_func: callable (GRIBMessage -> int) or None
        If set, use this to index fields (return value must be in range(0, total))
    kwargs: any
        Extra arguments for backends that support them

    Returns
    -------
    eccodes.GRIBMessage
        GRIB template (first message read)
    numpy array (total, npoints)
        Read data
    """

    readers = open_multi_dataset(source, **kwargs)
    template = None
    data = None
    n_read = 0
    for reader in readers:
        with reader:
            message = reader.peek()
            if message is None:
                raise EOFError(f"No data in {source!r}")
            if template is None:
                template = message
                data = np.empty(
                    (total, template.get("numberOfDataPoints")), dtype=dtype
                )
            for message in reader:
                i = n_read if index_func is None else index_func(message)
                data[i, :] = missing_to_nan(message)
                n_read += 1
    if n_read != total:
        raise EOFError(f"Expected {total} fields in {source!r}, got {n_read}")
    return template, data


def parse_paramids(pid):
    if isinstance(pid, int):
        return [pid]
    if isinstance(pid, str):
        return pid.split("/")
    if isinstance(pid, list):
        if not all(isinstance(p, (int, str)) for p in pid):
            raise TypeError("Lists of paramids can contain only ints or strings")
        return pid
    raise TypeError(f"Invalid paramid type {type(pid)}")


@dataclass
class ParamFilter:
    comparison: str
    threshold: float
    param: Optional[str]
    replacement: float

    @classmethod
    def from_config(cls, config: Optional[dict]) -> Optional["ParamFilter"]:
        if config is None:
            return None
        return cls(
            config["comparison"],
            config["threshold"],
            config.get("param", None),
            config.get("replacement", 0.0),
        )


class ParamConfig(BaseModel):
    name: str
    sources: dict
    preprocessing: Optional[list] = []
    accumulations: dict = {}
    overrides: Dict[str, Any] = {}
    dtype: type = np.float32
    metadata: Dict[str, Any] = {}

    def in_keys(self, name: str, base: Optional[Dict[str, Any]] = None, **kwargs):
        keys = base.copy() if base is not None else {}
        keys.update(self.sources[name]["request"])
        keys.update(kwargs)
        keys.update(self.overrides)
        return keys


class ParamRequester:
    def __init__(
        self,
        param: ParamConfig,
        sources: SourceModel,
        members: int | Members,
        total: int,
        src_names: Optional[List[str]] = None,
        index_func: Optional[IndexFunc] = None,
    ):
        self.param = param
        self.sources = sources
        self.src_names = self.sources.names if src_names is None else src_names
        self.members = members
        self.total = total
        self.index_func = index_func

    def _set_number(self, keys):
        number = None
        if isinstance(self.members, Members):
            number = range(self.members.start, self.members.end + 1)
        if keys.get("type") == "pf":
            keys["number"] = number or range(1, self.members + 1)
        elif keys.get("type") == "fcmean":
            keys["number"] = number or range(self.members + 1)

    def filter_data(self, data: np.ndarray, step: AnyStep, **kwargs) -> np.ndarray:
        filt = self.param.filter
        if filt is None:
            return data
        fdata = data
        if filt.param is not None:
            filt_keys = self.param.in_keys(step=str(step), **kwargs)[0]
            filt_keys["param"] = filt.param
            _, fdata = read_ensemble(
                self.source,
                self.total,
                dtype=self.param.dtype,
                update=self._set_number,
                index_func=self.index_func,
                **filt_keys,
            )
        comp = numexpr.evaluate(
            "data " + filt.comparison + " threshold",
            local_dict={
                "data": fdata,
                "threshold": np.asarray(filt.threshold, dtype=fdata.dtype),
            },
        )
        return np.where(comp, filt.replacement, data)

    def combine_data(self, data_list: List[np.ndarray]) -> np.ndarray:
        if self.param.combine is None:
            assert (
                len(data_list) == 1
            ), "Multiple input fields require a combine operation"
            return data_list[0]
        if self.param.combine == "norm":
            return np.linalg.norm(data_list, axis=0)
        if self.param.combine == "direction":
            assert len(data_list) == 2, "'direction' requires exactly 2 input fields"
            return direction(
                data_list[0], data_list[1], convention="meteo", to_positive=True
            ).astype(self.param.dtype)
        return getattr(np, self.param.combine)(data_list, axis=0)

    def retrieve_data(
        self, step: AnyStep, **kwargs
    ) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
        data_list = []
        for src in self.src_names:
            src_config = getattr(self.sources, src)
            in_keys = self.param.in_keys(
                src, src_config.request, step=str(step), **kwargs
            )
            for param_req in expand(in_keys, "param"):
                requests = list(expand(param_req, "type"))
                config = src_config.model_copy(update={"request": requests})
                template, data = read_ensemble(
                    config,
                    self.total,
                    dtype=self.param.dtype,
                    update=self._set_number,
                    index_func=self.index_func,
                )
                data_list.append(data)
        return (
            template,
            np.asarray(data_list)
            # self.filter_data(self.combine_data(data_list), step, **kwargs) * self.param.scale,
        )

    @property
    def name(self):
        return self.param.name
