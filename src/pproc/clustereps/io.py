# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import List, Tuple

import numpy as np

import eccodes

from pproc.clustereps.utils import normalise_angles
from pproc.common.dataset import open_dataset, open_multi_dataset


def read_ensemble_grib(sources: dict, loc: str, steps: List[int], nexp: int, **kwargs) -> \
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
    kwargs: any
        Exta arguments for source backends

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
        if keys.get('type') == 'pf':
            keys['number'] = range(1, nexp)
    inv_steps = {s: i for i, s in enumerate(steps)}
    nstep = len(steps)
    ens = None
    template = None
    readers = open_multi_dataset(sources, loc, step=steps, update=set_number, **kwargs)
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