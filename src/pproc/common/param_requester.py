from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import eccodes

from pproc.common.dataset import open_multi_dataset
from pproc.common.io import missing_to_nan
from pproc.common.steps import AnyStep
from pproc.common.window import parse_window_config
from pproc.config.preprocessing import (
    Combination,
    Masking,
    MaskExpression,
    PreprocessingConfig,
    Scaling,
)


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


def parse_paramids(pid: Any) -> List[str]:
    if isinstance(pid, int):
        return [str(pid)]
    if isinstance(pid, str):
        return pid.split("/")
    if isinstance(pid, list):
        if not all(isinstance(p, (int, str)) for p in pid):
            raise TypeError("Lists of paramids can contain only ints or strings")
        return pid
    raise TypeError(f"Invalid paramid type {type(pid)}")


def _compat_preprocessing(in_paramids: List[str], options: dict) -> PreprocessingConfig:
    combine_op = options.get("combine_operation", None)
    combine = None
    if combine_op is None:
        assert (
            len(in_paramids) == 1
        ), "Multiple input fields require a combine operation"
    else:
        combine = Combination(operation=combine_op, dim="param")
    filter_op = options.get("input_filter_operation", None)
    filter = None
    if filter_op is not None:
        assert (
            combine is None
        ), "Combining and filtering are not supported at the same time"
        filter_param = str(filter_op.get("param", None))
        if filter_param is None:
            filter_param = in_paramids[0]
        else:
            in_paramids.append(filter_param)
        filter = Masking(
            operation="mask",
            mask=MaskExpression(
                lhs={"param": filter_param},
                cmp=filter_op["comparison"],
                rhs=filter_op["threshold"],
            ),
            select={"param": in_paramids[0]},
            replacement=filter_op.get("replacement", 0.0),
        )
    scale_val = options.get("scale", None)
    scale = None if scale_val is None else Scaling(operation="scale", value=scale_val)

    pp_actions = [act for act in [combine, filter, scale] if act is not None]
    return PreprocessingConfig(actions=pp_actions)


class ParamConfig:
    def __init__(self, name, options: Dict[str, Any], overrides: Dict[str, Any] = {}):
        self.name = name
        self.in_paramids = parse_paramids(options["in"])
        self.out_paramid = options.get("out", None)
        self._in_keys = options.get("in_keys", {})
        self._out_keys = options.get("out_keys", {})
        self._steps = options.get("steps", None)
        self._windows = options.get("windows", None)
        self._accumulations = options.get("accumulations", None)
        self._in_overrides = overrides
        self.dtype = np.dtype(options.get("dtype", "float32")).type

        if any(
            key in options
            for key in ["combine_operation", "input_filter_operation", "scale"]
        ):
            self.preprocessing = _compat_preprocessing(self.in_paramids, options)
        else:
            self.preprocessing = PreprocessingConfig.model_validate(
                options.get("preprocessing", [])
            )

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
        if self._accumulations is not None:
            return {"accumulations": self._accumulations}

        if self._windows is not None:
            config = {"windows": self._windows}
            if self._steps is not None:
                config["steps"] = self._steps
            return config

        windows = []
        for coarse_cfg in base:
            coarse_window = parse_window_config(coarse_cfg)
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
        elif keys.get("type") == "fcmean":
            keys["number"] = range(self.members)

    def retrieve_data(
        self, fdb, step: AnyStep, **kwargs
    ) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
        metadata = self.param.in_keys(step=str(step), **kwargs)
        data_list = []
        template = None
        for in_keys in metadata:
            new_template, data = read_ensemble(
                self.sources,
                self.loc,
                self.total,
                dtype=self.param.dtype,
                update=self._set_number,
                index_func=self.index_func,
                **in_keys,
            )
            data_list.append(data)
            if template is None:
                template = new_template

        assert template is not None, "No data fetched"

        metadata, data_list = self.param.preprocessing.apply(metadata, data_list)
        return (template, data_list[0])

    @property
    def name(self):
        return self.param.name
