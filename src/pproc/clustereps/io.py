
import copy
import pprint
import re
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np

import eccodes
import mir
import pyfdb

from pproc.clustereps.utils import normalise_angles
from pproc.common.io import fdb_retrieve


_LOCATION_RE = re.compile('^([a-z](?:[a-z0-9+-.])*):(.*)$', re.I)


def _split_location(loc: str) -> Tuple[Optional[str], str]:
    m = _LOCATION_RE.fullmatch(loc)
    if m is None:
        return (None, loc)
    return m.groups()


def _open_dataset_fdb(reqs: Union[dict, Iterable[dict]], **kwargs: Any) -> Iterator[eccodes.reader.ReaderBase]:
    fdb = pyfdb.FDB()
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
        print("FDB request:")
        pprint.pprint(req)
        if interp is not None:
            print("Interpolation:")
            pprint.pprint(interp)
        stream = fdb_retrieve(fdb, req, interp)
        yield eccodes.StreamReader(stream)


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
    type_, ident = _split_location(loc)
    if type_ is None or type_ == 'file':
        return [eccodes.FileReader(ident)]
    if type_ == 'fdb':
        reqs = config.get(type_, {}).get(ident, None)
        if reqs is not None:
            return _open_dataset_fdb(reqs, **kwargs)
    raise ValueError(f"Unknown location {loc!r}")


def read_ensemble_grib(sources: dict, loc: str, steps: List[int], nexp: int) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, eccodes.Message]:
    """Read ensemble data from a GRIB file

    Parameters
    ----------
    sources: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    steps: list[int]
        List of steps
    nexp: int
        Number of ensemble members

    Returns
    -------
    numpy array (npoints)
        Latitudes in deg
    numpy array (npoints)
        Longitudes in [0, 360) deg
    numpy array (nexp, nstep, npoints)
        Ensemble data
    eccodes.Message
        Template message
    """
    def set_number(keys):
        if keys['type'] == 'pf':
            keys['number'] = range(1, nexp)
    inv_steps = {s: i for i, s in enumerate(steps)}
    nstep = len(steps)
    ens = None
    template = None
    readers = open_multi_dataset(sources, loc, step=steps, update=set_number)
    first = True
    for reader in readers:
        with reader:
            if first:
                message = reader.peek()
                if message is None:
                    raise EOFError(f"No data in {loc!r} for steps [{', '.join(str(step) for step in steps)}] (expected {nexp} members)")
                template = message
                npoints = message.get('numberOfDataPoints')
                lat = message.get_array('latitudes')
                lon = normalise_angles(message.get_array('longitudes'))
                ens = np.empty((nexp, nstep, npoints))
                first = False
            for message in reader:
                iexp = message.get('perturbationNumber')
                step = message.get('step:int')
                # TODO: check param and level
                istep = inv_steps.get(step, None)
                if istep is not None:
                    ens[iexp, istep, :] = message.get_array('values')
    return lat, lon, ens, template


def read_steps_grib(sources: dict, loc: str, steps: List[int], **kwargs) -> np.ndarray:
    """Read multi-step data from a GRIB file

    Parameters
    ----------
    sources: dict
        Sources configuration
    loc: str
        Location of the data (file path, named fdb request, ...)
    steps: list[int]
        List of steps
    kwargs: any
        Exta arguments for source backends

    Returns
    -------
    numpy array (nstep, npoints)
        Read data
    """
    inv_steps = {s: i for i, s in enumerate(steps)}
    nstep = len(steps)
    with open_dataset(sources, loc, step=steps, **kwargs) as reader:
        message = reader.peek()
        if message is None:
            raise EOFError(f"No data in {loc!r} for steps [{', '.join(str(step) for step in steps)}]")
        npoints = message.get('numberOfDataPoints')
        data = np.empty((nstep, npoints))
        for message in reader:
            step = message.get('step:int')
            # TODO: check param and level
            istep = inv_steps.get(step, None)
            if istep is not None:
                data[istep, :] = message.get_array('values')
    return data