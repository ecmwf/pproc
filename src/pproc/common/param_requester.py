from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numexpr
import numpy as np

import eccodes
from meteokit.wind import direction

from pproc.common.dataset import open_multi_dataset
from pproc.common.io import missing_to_nan
from pproc.common.steps import AnyStep
from pproc.common.window import Window


IndexFunc = Callable[[eccodes.GRIBMessage], int]


def read_ensemble(
    sources: dict,
    loc: str,
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

    readers = open_multi_dataset(sources, loc, **kwargs)
    template = None
    data = None
    n_read = 0
    for reader in readers:
        with reader:
            message = reader.peek()
            if message is None:
                raise EOFError(f"No data in {loc!r}")
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
        raise EOFError(f"Expected {total} fields in {loc!r}, got {n_read}")
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
        return cls(config["comparison"], config["threshold"], config.get("param", None), config.get("replacement", 0.))


class ParamConfig:
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        self.name = name
        self.in_paramids = parse_paramids(options["in"])
        self.combine = options.get("combine_operation", None)
        self.filter = ParamFilter.from_config(options.get("input_filter_operation", None))
        self.scale = options.get("scale", 1.0)
        self.out_paramid = options.get("out", None)
        self._in_keys = options.get("in_keys", {})
        self._out_keys = options.get("out_keys", {})
        self._steps = options.get("steps", None)
        self._windows = options.get("windows", None)
        self._in_overrides = overrides

    def in_keys(self, base: Optional[Dict[str, Any]] = None, **kwargs):
        keys = base.copy() if base is not None else {}
        keys.update(self._in_keys)
        keys.update(kwargs)
        keys.update(self._in_overrides)
        keys_list = []
        for pid in self.in_paramids:
            keys["param"] = pid
            keys_list.append(keys.copy())
        return keys_list

    def out_keys(self, base: Optional[Dict[str, Any]] = None, **kwargs):
        keys = base.copy() if base is not None else {}
        keys.update(self._out_keys)
        keys.update(kwargs)
        return keys

    def window_config(self, base: List[dict], base_steps: Optional[List[dict]] = None):
        if self._windows is not None:
            config = {"windows": self._windows}
            if self._steps is not None:
                config["steps"] = self._steps
            return config

        windows = []
        for coarse_cfg in base:
            coarse_window = Window(coarse_cfg)
            periods = [{"range": [step, step]} for step in coarse_window.steps]
            windows.append(
                {
                    "window_operation": "none",
                    "periods": periods,
                }
            )
        config = {"windows": windows}
        if base_steps:
            config["steps"] = base_steps

        return config


class ParamRequester:
    def __init__(
        self,
        param: ParamConfig,
        sources: dict,
        loc: str,
        members: int,
        total: Optional[int] = None,
        index_func: Optional[IndexFunc] = None,
    ):
        self.param = param
        self.sources = sources
        self.loc = loc
        self.members = members
        self.total = total if total is not None else members
        self.index_func = index_func

    def _set_number(self, keys):
        if keys.get("type") == "pf":
            keys["number"] = range(1, self.members)

    def filter_data(self, data: np.ndarray, step: AnyStep) -> np.ndarray:
        filt = self.param.filter
        if filt is None:
            return data
        fdata = data
        if filt.param is not None:
            filt_keys = self.param.in_keys(step=str(step))[0]
            filt_keys["param"] = filt.param
            _, fdata = read_ensemble(
                self.sources,
                self.loc,
                self.total,
                update=self._set_number,
                index_func=self.index_func,
                **filt_keys,
            )
        comp = numexpr.evaluate(
            "data " + filt.comparison + str(filt.threshold),
            local_dict={"data": fdata}
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
            return direction(data_list[0], data_list[1], convention="meteo", to_positive=True)
        return getattr(np, self.param.combine)(data_list, axis=0)

    def retrieve_data(
        self, fdb, step: AnyStep, **kwargs
    ) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
        data_list = []
        for in_keys in self.param.in_keys(step=str(step), **kwargs):
            template, data = read_ensemble(
                self.sources,
                self.loc,
                self.total,
                update=self._set_number,
                index_func=self.index_func,
                **in_keys,
            )
            data_list.append(data)
        return template, self.filter_data(self.combine_data(data_list), step) * self.param.scale

    @property
    def name(self):
        return self.param.name
