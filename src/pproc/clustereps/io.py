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
from pproc.common.param_requester import ParamRequester
from pproc.config.io import InputsCollection
from pproc.config.param import ParamConfig


def read_ensemble_grib(
    inputs: InputsCollection, src_name: str, steps: List[int], nexp: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, eccodes.Message]:
    """Read ensemble data from a GRIB file

    Parameters
    ----------
    inputs: InputsCollection
        Inputs configuration
    src_name: str
        Source name
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
    nstep = len(steps)

    def index_func(message: eccodes.GRIBMessage) -> int:
        return message.get("perturbationNumber")

    param = ParamConfig(name="ens", dtype="float64")
    requester = ParamRequester(
        param, inputs, nexp, src_name=src_name, index_func=index_func
    )
    ens = None
    template = None
    for i, step in enumerate(steps):
        templates, data = requester.retrieve_data(step, **kwargs)
        if ens is None:
            template = templates[0]
            npoints = template.get("numberOfDataPoints")
            ens = np.empty((nexp, nstep, npoints))
        ens[:, i, :] = data

    lat = template.get_array("latitudes")
    lon = normalise_angles(template.get_array("longitudes"))
    return lat, lon, ens, template


def read_steps_grib(
    inputs: InputsCollection, src_name: str, steps: List[int], **kwargs
) -> np.ndarray:
    """Read multi-step data from a GRIB file

    Parameters
    ----------
    inputs: InputsCollection
        Inputs configuration
    src_name: str
        Source name
    steps: list[int]
        List of steps
    kwargs: any
        Exta arguments for source backends

    Returns
    -------
    numpy array (nstep, npoints)
        Read data
    """
    nstep = len(steps)
    param = ParamConfig(name="input", dtype="float64")
    requester = ParamRequester(param, inputs, 1, src_name=src_name)

    data = None
    for i, step in enumerate(steps):
        templates, step_data = requester.retrieve_data(step, **kwargs)
        if data is None:
            data = np.empty((nstep, templates[0].get("numberOfDataPoints")))
        data[i, :] = step_data[0, :]

    return data


def read_grib_cluster(
    inputs: InputsCollection,
    name: str,
    steps: list[int],
    nexp: int,
    max_clusters: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read clustering data from a GRIB file

    Parameters
    ----------
    inputs: InputsCollection
        Inputs configuration
    name: str
        Source name
    steps: list[int]
        List of steps
    nexp: int
        Total number of ensemble members
    max_clusters: int
        Maximum number of clusters

    Returns
    -------
    numpy.array(nsteps, nclusters, npoints) [float64]
        Cluster data
    numpy.array (nclusters, nens) [int16]
        Ensemble numbers in each cluster
    numpy.array (npoints) [float64]
        Latitude in deg
    numpy.array (npoints) [float64]
        Longitudes in [0, 360) deg
    """
    nsteps = len(steps)

    def index_func(message: eccodes.GRIBMessage) -> int:
        return message.get("clusterNumber") - 1

    param = ParamConfig(name="clusters", dtype="float64")
    requester = ParamRequester(
        param, inputs, max_clusters, src_name=name, index_func=index_func
    )

    nclusters = None
    vals = None
    refs = None
    for jstep, step in enumerate(steps):
        templates, data = requester.retrieve_data(step)

        if vals is None:
            nclusters = templates[0].get("totalNumberOfClusters")
            refs = templates[:nclusters]
            npoints = refs[0].get("numberOfDataPoints")
            vals = np.empty((nsteps, nclusters, npoints), dtype=np.float64)
        vals[jstep, :nclusters, :] = data

    ens_numbers = np.full((nclusters, nexp), -1, dtype=np.int16)
    for i in range(nclusters):
        nFcsts = refs[i].get("numberOfForecastsInCluster")
        ens_numbers[i, :nFcsts] = refs[i].get_array("ensembleForecastNumbers")

    lat = refs[0].get("latitudes")
    lon = normalise_angles(refs[0].get("longitudes"))

    return vals, ens_numbers, lat, lon
