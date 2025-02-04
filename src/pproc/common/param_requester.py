from typing import Any, Callable, List, Optional, Tuple

import eccodes
import numpy as np

from pproc.common.dataset import open_multi_dataset
from pproc.common.io import missing_to_nan
from pproc.common.steps import AnyStep
from pproc.config.base import Members
from pproc.config.io import Source, SourceCollection
from pproc.config.utils import expand
from pproc.config.param import ParamConfig

IndexFunc = Callable[[eccodes.GRIBMessage], int]


def read_ensemble(
    source: Source,
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
    readers = open_multi_dataset(source.legacy_config(), source.location(), **kwargs)
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


class ParamRequester:
    def __init__(
        self,
        param: ParamConfig,
        sources: SourceCollection,
        members: int | Members,
        total: int,
        src_name: Optional[str] = None,
        index_func: Optional[IndexFunc] = None,
    ):
        self.param = param
        self.sources = sources
        self.src_name = src_name
        if self.src_name is None:
            assert len(sources.names) == 1, "Multiple sources, must specify src_name"
            self.src_name = sources.names[0]
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

    def retrieve_data(
        self, step: AnyStep, **kwargs
    ) -> Tuple[eccodes.GRIBMessage, np.ndarray]:
        metadata = []
        data_list = []
        template = None
        in_keys = self.param.in_keys(
            self.src_name,
            step=str(step),
            **kwargs,
            **self.sources.overrides,
        )
        src_config: Source = getattr(self.sources, self.src_name)
        for param_req in expand(in_keys, "param"):
            new_template, data = read_ensemble(
                Source(
                    type=self.param.sources[self.src_name].get("type", src_config.type),
                    path=self.param.sources[self.src_name].get("path", src_config.path),
                    request=src_config.request,
                ),
                self.total,
                dtype=self.param.dtype,
                update=self._set_number,
                index_func=self.index_func,
                **param_req,
            )
            metadata.append(param_req)
            data_list.append(data)
            if template is None:
                template = new_template

        assert template is not None, "No data fetched"

        new_metadata, data_list = self.param.preprocessing.apply(metadata, data_list)
        assert len(data_list) == 1, "More than one output of preprocessing"
        metadata_set = {k: v for k, v in new_metadata[0].items() if v != metadata[0][k]}
        template.set(metadata_set)
        return (template, data_list[0])

    @property
    def name(self):
        return self.param.name
