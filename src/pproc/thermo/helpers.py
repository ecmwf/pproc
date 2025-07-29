# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data
from earthkit.data.encoders import grib
import logging
import numpy as np
import thermofeel
from meters import metered
from typing import Optional

from pproc import common
from pproc.config.targets import Target

logger = logging.getLogger(__name__)

# Constants
UTCI_MIN_VALUE = thermofeel.celsius_to_kelvin(-80)
UTCI_MAX_VALUE = thermofeel.celsius_to_kelvin(90)

MRT_DIFF_HIGH = 150
MRT_DIFF_LOW = -30

# parameter to units
units = {
    "cossza": "",
    "2t": "K",
    "2d": "K",
    "wcf": "K",
    "aptmp": "K",
    "hmdx": "K",
    "nefft": "K",
    "wbgt": "K",
    "wbpt": "K",
    "gt": "K",
    "2r": "%",
    "10si": "m/s",
    "mrt": "K",
    "utci": "K",
    "heatx": "K",
    "dsrp": "W/m2",
    "ssrd": "W/m2",
    "ssr": "W/m2",
    "fdir": "W/m2",
    "strd": "W/m2",
    "str": "W/m2",
}


def compute_ehPa_(rh_pc, svp):
    return svp * rh_pc * 0.01  # / 100.0


@metered("ehPa", out=logger.debug)
def compute_ehPa(t2m, t2d):
    rh_pc = thermofeel.calculate_relative_humidity_percent(t2m, t2d)
    svp = thermofeel.calculate_saturation_vapour_pressure(t2m)
    ehPa = compute_ehPa_(rh_pc, svp)
    return ehPa


def find_utci_missing_values(t2m, va, mrt, ehPa, utci, print_misses=True):
    e_mrt = np.subtract(mrt, t2m)

    misses = np.where(t2m >= thermofeel.celsius_to_kelvin(70))
    nt2high = len(misses[0])
    t = np.where(t2m <= thermofeel.celsius_to_kelvin(-70))
    nt2low = len(t[0])
    misses = np.union1d(t, misses)

    t = np.where(va >= 25.0)  # 90kph
    nhighwind = len(t[0])
    misses = np.union1d(t, misses)

    t = np.where(ehPa > 50.0)
    nehpa = len(t[0])
    misses = np.union1d(t, misses)

    t = np.where(e_mrt >= MRT_DIFF_HIGH)
    ndiffmrt = len(t[0])
    misses = np.union1d(t, misses)

    t = np.where(e_mrt <= MRT_DIFF_LOW)
    ndiffmrtneg = len(t[0])
    misses = np.union1d(t, misses)

    t = np.where(np.isnan(utci))
    nnan = len(t[0])
    misses = np.union1d(t, misses)

    nmisses = len(misses)

    if print_misses:
        print(
            f"UTCI nmisses {nmisses} NANs {nnan} T2>70C {nt2high} T2<-70 {nt2low} highwind {nhighwind}"
            + f"nehpa {nehpa} MRT-T2>{MRT_DIFF_HIGH} {ndiffmrt} MRT-T2<{MRT_DIFF_LOW} {ndiffmrtneg}"
        )

    return misses


def validate_utci(utci, misses, lats, lons):

    out_of_bounds = 0
    nans = 0
    for i in range(len(utci)):
        v = utci[i]
        if v < UTCI_MIN_VALUE or v > UTCI_MAX_VALUE:
            out_of_bounds += 1
            logger.info("UTCI [", i, "] = ", utci[i], " : lat/lon ", lats[i], lons[i])
        if np.isnan(v):
            nans += 1
            logger.info("UTCI [", i, "] = ", utci[i], " : lat/lon ", lats[i], lons[i])

    nmisses = len(misses)
    if nmisses > 0 or out_of_bounds > 0 or nans > 0:
        logger.info(
            f"UTCI => nmisses {nmisses} out_of_bounds {out_of_bounds} NANs {nans}"
        )


def get_datetime(fields: earthkit.data.FieldList):
    dt = fields[0].datetime()
    base_time = dt["base_time"]
    valid_time = dt["valid_time"]
    assert all(
        x == valid_time for x in fields.datetime()["valid_time"]
    ), f"Obtained different valid times {[x for x in fields.datetime()['valid_time']]}"  # verify valid time all same
    return base_time, valid_time


def latlon(fields: earthkit.data.FieldList):
    latlon = fields[0].to_latlon(flatten=True)
    lat = latlon["lat"]
    lon = latlon["lon"]
    assert lat.size == lon.size
    assert fields[0].values.size == lat.size
    return lat, lon


def field_values(fields: earthkit.data.FieldList, param: str) -> np.ndarray:
    sel = fields.sel(param=param)
    if len(sel) == 0:
        raise ValueError(
            f"Field {param} not found in fields {fields.ls(namespace='mars')}"
        )
    return sel.to_array()


def check_field_sizes(fields: earthkit.data.FieldList):
    all(f.values.shape == fields[0].values.shape for f in fields)


def step_interval(fields) -> int:
    # Derive step interval from de-accumulated fields
    accum_field = fields.sel(stepType="diff")
    if len(accum_field) == 0:
        raise ValueError("No accumulation fields found, can not derive step interval")
    delta = (
        accum_field[0].metadata()["endStep"] - accum_field[0].metadata()["startStep"]
    )
    check = [
        delta == (f.metadata()["endStep"] - f.metadata()["startStep"])
        for f in accum_field
    ]
    if not all(check):
        raise ValueError(
            f"Step intervals are not consistent for accumulated fields {accum_field.ls()}"
        )
    return delta


def write(
    target: Target,
    ds: "earthkit.data.FieldList | earthkit.data.core.fieldlist.Field",
    metadata: Optional[dict] = None,
):
    if isinstance(ds, earthkit.data.core.fieldlist.Field):
        ds = [ds]
    metadata = metadata or {}
    for f in ds:
        field_metadata = f.metadata()
        updates = metadata.copy()
        # Handle wrapped metadata
        if hasattr(field_metadata, "extra"):
            updates.update(field_metadata.extra)
        message = f.metadata()._handle.copy()
        message.set(updates)
        common.io.write_grib(target, message, f.values)
