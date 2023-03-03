
from contextlib import ExitStack
import copy
from io import BytesIO
import pprint
from typing import Any, Callable, Iterable, Iterator, List, Union

import eccodes
import mir

from pproc.common.io import FileTarget, NullTarget, fdb, fdb_retrieve, split_location
from pproc.common.mars import mars_retrieve


def _open_dataset_marslike(name: str, retrieve_func: Callable[[dict, dict], eccodes.reader.ReaderBase], reqs: Union[dict, Iterable[dict]], **kwargs: Any) -> Iterator[eccodes.reader.ReaderBase]:
    if not isinstance(reqs, list):
        reqs = [reqs]
    update_func = kwargs.pop('update', None)
    interp_extra = kwargs.pop('interpolate', {})
    for req in reqs:
        req = copy.deepcopy(req)
        req.update(kwargs)
        if update_func is not None:
            update_func(req)
        interp = req.pop('interpolate', None)
        if interp_extra:
            if interp is None:
                interp = {}
            interp.update(interp_extra)
        print(f"{name} request:")
        pprint.pprint(req)
        if interp is not None:
            print("Interpolation:")
            pprint.pprint(interp)
        yield retrieve_func(req, interp)


def _fdb_retrieve_interp(request: dict, mir_options: dict) -> eccodes.reader.ReaderBase:
    fdb_reader = fdb_retrieve(fdb(), request, mir_options)
    return eccodes.StreamReader(fdb_reader)


def _open_dataset_fdb(reqs: Union[dict, Iterable[dict]], **kwargs: Any) -> Iterator[eccodes.reader.ReaderBase]:
    return _open_dataset_marslike("FDB", _fdb_retrieve_interp, reqs, **kwargs)


class MARSDecoder(eccodes.StreamReader):
    def __init__(self, stream, cache=None):
        super().__init__(stream)
        self.stack = ExitStack()
        self.cache = cache if cache is not None else NullTarget

    def __enter__(self):
        self.stack.enter_context(self.stream)
        self.stack.enter_context(self.cache)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.stack.__exit__(exc_type, exc_value, traceback)

    def __next__(self):
        msg = super().__next__()
        self.cache.write(msg)
        return msg


def _mars_retrieve_interp(request: dict, mir_options: dict, mars_cmd: Union[str, List[str]] = "mars", tmpdir=None) -> eccodes.reader.ReaderBase:
    cache_path = request.pop('cache', None)
    cache = None if cache_path is None else FileTarget(cache_path.format_map(request))
    mars_reader = mars_retrieve(request, mars_cmd=mars_cmd, tmpdir=tmpdir)
    if mir_options:
        with mars_reader:
            job = mir.Job(**mir_options)
            stream = BytesIO()
            job.execute(mars_reader, stream)
        stream.seek(0)
        mars_reader = stream
    return MARSDecoder(mars_reader, cache=cache)


def _open_dataset_mars(reqs: Union[dict, Iterable[dict]], **kwargs: Any) -> Iterator[eccodes.reader.ReaderBase]:
    return _open_dataset_marslike("MARS", _mars_retrieve_interp, reqs, **kwargs)


def _open_dataset_fileset(reqs: Union[dict, Iterable[dict]], **kwargs: Any) -> Iterator[eccodes.reader.ReaderBase]:
    if not isinstance(reqs, list):
        reqs = [reqs]
    update_func = kwargs.pop('update', None)
    for req in reqs:
        req = copy.deepcopy(req)
        req.update(kwargs)
        if update_func is not None:
            update_func(req)
        template = req.pop('location')
        path = template.format(**req)
        print(f"File path: {path!r}")
        yield eccodes.FileReader(path)


def open_dataset(config: dict, loc: str, **kwargs) -> eccodes.reader.ReaderBase:
    """Open a GRIB dataset

    Parameters
    ----------
    config: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    kwags: any
        Exta arguments for backends that support them

    Returns
    -------
    eccodes.reader.ReaderBase
        GRIB reader
    """
    readers = list(open_multi_dataset(config, loc, **kwargs))
    if len(readers) != 1:
        raise ValueError(f"Multiple readers found but not expected for {loc!r}")
    return readers[0]


_DATASET_BACKENDS = {
    'fdb': _open_dataset_fdb,
    'mars': _open_dataset_mars,
    'fileset': _open_dataset_fileset,
}


def open_multi_dataset(config: dict, loc: str, **kwargs) -> Iterable[eccodes.reader.ReaderBase]:
    """Open a multi-part GRIB dataset

    Parameters
    ----------
    config: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    kwargs: any
        Exta arguments for backends that support them

    Returns
    -------
    list[eccodes.reader.ReaderBase]
        GRIB readers
    """
    type_, ident = split_location(loc, default='file')
    if type_ == 'file':
        return [eccodes.FileReader(ident)]
    reqs = config.get(type_, {}).get(ident, None)
    open_func = _DATASET_BACKENDS.get(type_, None)
    if reqs is not None and open_func is not None:
        return open_func(reqs, **kwargs)
    raise ValueError(f"Unknown location {loc!r}")