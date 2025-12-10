# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
import logging
import numpy as np
import copy
from typing import Optional
import functools

import earthkit.data
import earthkit.meteo.solar
import thermofeel
from meters import metered

from pproc.thermo.helpers import (
    compute_ehPa,
    field_values,
    find_utci_missing_values,
    get_datetime,
    latlon,
    step_interval,
    validate_utci,
)

logger = logging.getLogger(__name__)

def fieldlist_wrapper(func, template_func, metadata):
    @functools.wraps(func)
    def wrapped(fields: earthkit.data.FieldList):
        values = func(*fields.to_array())
        template = [
            x.override(
                **metadata,
                typeOfFirstFixedSurface=1,
                scaleFactorOfFirstFixedSurface="MISSING",
                scaledValueOfFirstFixedSurface="MISSING",
            )
            for x in template_func(fields)
        ]
        return earthkit.data.FieldList.from_array(values, template)
    return wrapped


def metadata_intensity(fields):
    return fields.sel(param="2t").metadata()


def metadata_accumulation(fields):
    return fields.sel(param="fdir").metadata()


def calc_cossza_int(fields, metadata) -> earthkit.data.FieldList:
    lats, lons = latlon(fields)

    basetime, validtime = get_datetime(fields)

    delta = step_interval(fields)

    dtbegin = validtime - timedelta(hours=delta)
    dtend = validtime

    cossza = earthkit.meteo.solar.cos_solar_zenith_angle_integrated(
        latitudes=lats,
        longitudes=lons,
        begin_date=dtbegin,
        end_date=dtend,
        integration_order=2,
    )

    template = [
        x.override(
            **metadata,
            typeOfFirstFixedSurface=1,
            scaleFactorOfFirstFixedSurface="MISSING",
            scaledValueOfFirstFixedSurface="MISSING",
        ) for x in metadata_intensity(fields)[:1]
    ]
    return earthkit.data.FieldList.from_array(cossza, template)

def calc_utci(t2_k, td_k, ws, mrt):
    ehPa = compute_ehPa(t2_k, td_k)
    utci = thermofeel.calculate_utci(t2_k=t2_k, va=ws, mrt=mrt, ehPa=ehPa)  # Kelvin
    for index in range(len(utci)):
        missing = find_utci_missing_values(
            t2_k[index],
            ws[index],
            mrt[index],
            ehPa[index],
            utci[index],
        )
        utci[index][missing] = np.nan
    return utci

def calc_mrt():
        delta = step_interval(fields)
        seconds_in_time_step = delta * 3600  # steps are in hours

        f = 1.0 / float(seconds_in_time_step)

        ssrd = field_values(fields, "ssrd")  # W/m2
        fdir = field_values(fields, "fdir")  # W/m2
        strd = field_values(fields, "strd")  # W/m2
        strr = field_values(fields, "str")  # W/m2
        ssr = field_values(fields, "ssr")  # W/m2

        # remove negative values from deaccumulated solar fields
        for v in ssrd, fdir, strd, ssr:
            v[v < 0] = 0

        return thermofeel.calculate_mean_radiant_temperature(
            ssrd * f, ssr * f, dsrp * f, strd * f, fdir * f, strr * f, cossza
        )  # Kelvin


class ThermalIndices(fluent.Action):
    def cossza(self, dim: str, metadata: Optional[dict] = None):
        return self.reduce(functools.partial(calc_cossza_int, metadata=metadata), dim=dim)
        
    def wind_speed(self, dim: str, metadata: Optional[dict] = None):
        try:
            return self.sel(**{dim: ("10si",)})
        except KeyError:
            return self.sel(**{dim: ("10u", "10v")}).reduce(
                fieldlist_wrapper(
                    np.norm, 
                    lambda ds: ds.sel(param="10u").metadata(), 
                    {**metadata, "paramId": "207"}
                ),
                dim=dim
            )

    def hmdx(self, dim: str, metadata: Optional[dict] = None):
        selection = self.sel(**{dim: ("2t", "2d")})        
        return selection._wrapped_reduction(
            fieldlist_wrapper(
                thermofeel.calculate_humidex, 
                metadata_intensity,
                {**metadata, "paramId": "261016"},
                
            ), dim=dim
        )

    def rhp(self, dim: str, metadata: Optional[dict] = None):
        selection = self.sel(**{dim: ("2t", "2d")})        
        return selection._wrapped_reduction(
            fieldlist_wrapper(
                thermofeel.calculate_relative_humidity_percent, 
                metadata_intensity,
                {**metadata, "paramId": "260242"}
                
            ), dim=dim
        )

    def heatx(self, dim: str, metadata: Optional[dict] = None):
        selection = self.sel(**{dim: ("2t", "2d")})        
        return selection._wrapped_reduction(
            fieldlist_wrapper(
                thermofeel.calculate_heat_index_adjusted, 
                metadata_intensity,
                {**metadata, "paramId": "260004"}
                
            ), dim=dim
        )

    # def field_stats(self, name, values):
    #     logger.debug(
    #         f"{name:<8} min {np.nanmin(values):>16.6f} max {np.nanmax(values):>16.6f} "
    #         f"avg {np.nanmean(values):>16.6f} stddev {np.nanstd(values, dtype=np.float64):>16.6f} "
    #         f"missing {np.count_nonzero(np.isnan(values)):>8}"
    #     )

    def dsrp(self, dim: str, metadata: Optional[dict] = None):
        """
        In the absence of dsrp, approximate it with fdir and cossza.
        Note this introduces some amount of error as cossza approaches zero
        """
        try:
            return self.sel(**{dim: ("dsrp",)})
        except:

        fdir = self.sel(**{dim: ("fdir",)})
        cossza = self.cossza(dim, metadata)
        return fdir.join(cossza)._wrapped_reduction(
            fieldlist_wrapper(
                thermofeel.approximate_dsrp,
                metadata_accumulation,
                {**metadata, "paramId": "47"}
            ), 
            dim=dim,
        )

    def utci(self, dim: str, metadata):
        selection = self.sel(**{dim: ("2t", "2d")}) 
        ws = self.wind_speed(dim=dim)
        mrt = self.mrt(dim=dim, metadata=metadata)  # Kelvin
        return selection.join(ws, dim=dim).join(mrt, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calc_utci,
                metadata_intensity,
                {**metadata, "paramId": "261001"}
            ), dim=dim
        )

    def wbgt(self, dim: str, metadata):
        t2m = self.sel(**{dim: ("2t",)}) 
        ws = self.wind_speed(dim=dim)
        mrt = self.mrt(dim=dim, metadata=metadata)  # Kelvin
        t2d = self.sel(**{dim: ("2d",)}) 
        return t2m.join(mrt, dim=dim).join(ws, dim=dim).join(t2d, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_wbgt,
                metadata_intensity,
                {**metadata, "paramId": "261014"}
            ), dim=dim
        )

    def gt(self, dim: str, metadata):
        t2m = self.sel(**{dim: ("2t",)}) 
        ws = self.wind_speed(dim=dim)
        mrt = self.mrt(dim=dim, metadata=metadata)  # Kelvin
        return t2m.join(mrt, dim=dim).join(ws, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_bgt,
                metadata_intensity,
                {**metadata, "paramId": "261015"}
            ), dim=dim
        )

    def wbt(self, dim: str, metadata):
        t2m = self.sel(**{dim: ("2t",)}) 
        rhp = self.rhp(dim=dim, metadata=metadata)  # Kelvin
        return t2m.join(rhp, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_wbt,
                metadata_intensity,
                {**metadata, "paramId": "261023"}
            ), dim=dim
        )

    def nefft(self, dim: str, metadata):
        t2m = self.sel(**{dim: ("2t",)}) 
        ws = self.wind_speed(dim=dim)
        rhp = self.rhp(dim=dim, metadata=metadata)  # Kelvin
        return t2m.join(ws, dim=dim).join(rhp, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_normal_effective_temperature,
                metadata_intensity,
                {**metadata, "paramId": "261018"}
            ), dim=dim
        )

    def wcf(self, dim: str, metadata):
        t2m = self.sel(**{dim: ("2t",)}) 
        ws = self.wind_speed(dim=dim)
        return t2m.join(ws, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_wind_chill,
                metadata_intensity,
                {**metadata, "paramId": "260005"}
            ), dim=dim
        )

    def aptmp(self, dim: str, metadata):
        t2m = self.sel(**{dim: ("2t",)}) 
        ws = self.wind_speed(dim=dim)
        rhp = self.rhp(dim=dim, metadata=metadata)
        return t2m.join(ws, dim=dim).join(rhp, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_apparent_temperature,
                metadata_intensity,
                {**metadata, "paramId": "260255"}
            ), dim=dim
        )

    def mrt(self, dim: str, metadata):
        selection = self.sel(**{dim: ("ssrd", "fdir", "strd", "str", "ssr")})
        cossza = self.cossza("cossza", dim=dim, metadata)
        dsrp = self.dsrp("cossza", dim=dim, metadata)
        return selection.join(dsrp, dim=dim).join(cossza, dim=dim)._wrapped_reduction(
            fieldlist_wrapper(
                calculate_apparent_temperature,
                metadata_intensity,
                {**metadata, "paramId": "261002"}
            ), dim=dim
        )
