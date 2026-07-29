# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from earthkit.data import FieldList
from earthkit.data.core.metadata import Metadata as ekMetadata
import earthkit.meteo.solar
import thermofeel
from meters import metered
import numpy as np
import logging
from datetime import timedelta
from typing import Optional

from ppruntime.thermo.helpers import (
    compute_ehPa,
    find_utci_missing_values,
    get_datetime,
    latlon,
    step_interval,
    validate_utci,
    create_output,
)
from ppruntime.metadata import resolve_metadata

logger = logging.getLogger(__name__)


def metadata_intensity(fields: FieldList) -> list[ekMetadata]:
    return fields.sel(paramId=167).metadata()


def metadata_accumulation(fields: FieldList) -> list[ekMetadata]:
    return fields.sel(paramId=228021).metadata()


def field_values(fields: FieldList, params: list[int]):
    shape = None
    out_values = []
    for param in params:
        selected = fields.sel(paramId=param)
        if len(selected) == 0:
            raise ValueError(
                f"Field {param} not found in fields: \n {fields.ls(namespace='mars')}"
            )
        if shape is None:
            shape = selected.values.shape
        assert (
            shape == selected.values.shape
        ), f"Shape mismatch for {param} {shape} != {selected.values.shape}"
        out_values.append(selected.to_array())
    return out_values


@metered("cossza", out=logger.debug)
def calc_cossza(*fields: FieldList, metadata: Optional[dict] = None) -> FieldList:
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_cossza: {summed_fields.ls(namespace='mars')}")
    lats, lons = latlon(summed_fields)

    basetime, validtime = get_datetime(summed_fields)

    delta = step_interval(summed_fields)

    dtbegin = validtime - timedelta(hours=delta)
    dtend = validtime

    cossza = earthkit.meteo.solar.cos_solar_zenith_angle_integrated(
        latitudes=lats,
        longitudes=lons,
        begin_date=dtbegin,
        end_date=dtend,
        integration_order=2,
    )

    return create_output(
        cossza,
        metadata_intensity(summed_fields)[:1],
        {**resolve_metadata(metadata)},
    )


@metered("hmdx", out=logger.debug)
def calc_hmdx(*fields: FieldList, metadata: Optional[dict] = None) -> FieldList:
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_hmdx: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 168])
    hmdx = thermofeel.calculate_humidex(*inputs)
    return create_output(
        hmdx,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("rhp", out=logger.debug)
def calc_rhp(*fields: FieldList, metadata: Optional[dict] = None) -> FieldList:
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_rhp: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 168])
    rhp = thermofeel.calculate_relative_humidity_percent(*inputs)
    return create_output(
        rhp,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("heatx", out=logger.debug)
def calc_heatx(*fields: FieldList, metadata: Optional[dict] = None) -> FieldList:
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_heatx: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 168])
    heatx = thermofeel.calculate_heat_index_adjusted(*inputs)
    return create_output(
        heatx,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("dsrp", out=logger.debug)
def calc_dsrp(*fields: FieldList, metadata: Optional[dict] = None):
    """
    In the absence of dsrp, approximate it with fdir and cossza.
    Note this introduces some amount of error as cossza approaches zero
    """
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_dsrp: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [228021, 214001])
    dsrp = thermofeel.approximate_dsrp(*inputs)
    return create_output(
        dsrp,
        metadata_accumulation(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("utci", out=logger.debug)
def calc_utci(*fields: FieldList, metadata: Optional[dict] = None, validate=True):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_utci: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 168, 207, 261002])

    ehPa = compute_ehPa(inputs[0], inputs[1])
    utci = thermofeel.calculate_utci(
        t2_k=inputs[0], va=inputs[2], mrt=inputs[3], ehPa=ehPa
    )  # Kelvin

    for index in range(len(utci)):
        missing = find_utci_missing_values(
            inputs[0][index],
            inputs[2][index],
            inputs[3][index],
            ehPa[index],
            utci[index],
        )

        if validate:
            lats, lons = latlon(summed_fields)
            validate_utci(utci[index], missing, lats, lons)
        utci[index][missing] = np.nan

    return create_output(
        utci,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("wbgt", out=logger.debug)
def calc_wbgt(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_wbgt: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 261002, 207, 168])
    wbgt = thermofeel.calculate_wbgt(*inputs)
    return create_output(
        wbgt,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("gt", out=logger.debug)
def calc_gt(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_gt: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 261002, 207])

    gt = thermofeel.calculate_bgt(*inputs)

    return create_output(
        gt,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("wbt", out=logger.debug)
def calc_wbt(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_wbt: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 260242])
    wbt = thermofeel.calculate_wbt(*inputs)
    return create_output(
        wbt,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("nefft", out=logger.debug)
def calc_nefft(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_nefft: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 207, 260242])
    nefft = thermofeel.calculate_normal_effective_temperature(*inputs)
    return create_output(
        nefft,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("wcf", out=logger.debug)
def calc_wcf(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_wcf: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 207])
    wcf = thermofeel.calculate_wind_chill(*inputs)
    return create_output(
        wcf,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("aptmp", out=logger.debug)
def calc_aptmp(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_aptmp: {summed_fields.ls(namespace='mars')}")
    inputs = field_values(summed_fields, [167, 207, 260242])
    aptmp = thermofeel.calculate_apparent_temperature(*inputs)
    return create_output(
        aptmp,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )


@metered("mrt", out=logger.debug)
def calc_mrt(*fields: FieldList, metadata: Optional[dict] = None):
    summed_fields: FieldList = sum(fields[1:], fields[0])
    logger.debug(f"calc_mrt: {summed_fields.ls(namespace='mars')}")
    ssrd, ssr, dsrp, strd, fdir, strr, cossza = field_values(
        summed_fields, [169, 176, 47, 175, 228021, 177, 214001]
    )

    delta = step_interval(summed_fields)
    seconds_in_time_step = delta * 3600  # steps are in hours

    f = 1.0 / float(seconds_in_time_step)

    # remove negative values from deaccumulated solar fields
    for v in ssrd, fdir, strd, ssr:
        v[v < 0] = 0

    mrt = thermofeel.calculate_mean_radiant_temperature(
        ssrd * f, ssr * f, dsrp * f, strd * f, fdir * f, strr * f, cossza
    )  # Kelvin

    return create_output(
        mrt,
        metadata_intensity(summed_fields),
        {**resolve_metadata(metadata)},
    )
