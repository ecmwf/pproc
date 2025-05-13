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
    units,
    validate_utci,
)

logger = logging.getLogger(__name__)


def metadata_intensity(fields):
    return fields.sel(param="2t").metadata()


def metadata_accumulation(fields):
    return fields.sel(param="fdir").metadata()


class ComputeIndices:
    def __init__(self, out_keys={}):
        self.out_keys = out_keys
        self.results = None

    def create_output(self, values, template, **overrides):
        grib_set = copy.deepcopy(self.out_keys)
        grib_set.update(overrides)
        template = [x.override(**grib_set) for x in template]
        return earthkit.data.FieldList.from_array(values, template)

    def create_surface_output(self, values, template, **overrides):
        grib_set = copy.deepcopy(self.out_keys)
        grib_set.update(overrides)
        template = [
            x.override(
                **grib_set,
                typeOfFirstFixedSurface=1,
                scaleFactorOfFirstFixedSurface="MISSING",
                scaledValueOfFirstFixedSurface="MISSING",
            )
            for x in template
        ]
        return earthkit.data.FieldList.from_array(values, template)

    @metered("cossza", out=logger.debug)
    def calc_cossza_int(self, fields):

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

        return self.create_output(
            cossza, metadata_intensity(fields)[:1], paramId="214001"
        )

    @metered("ws", out=logger.debug)
    def calc_ws(self, fields):
        if "10si" in fields.indices()["param"]:
            return fields.sel(param="10si")

        u10 = field_values(fields, "10u")  # m/s
        v10 = field_values(fields, "10v")  # m/s

        ws = np.sqrt(u10**2 + v10**2)  # m/s
        template = fields.sel(param="10u").metadata()
        return self.create_output(ws, template, paramId="207")

    @metered("hmdx", out=logger.debug)
    def calc_hmdx(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        td = field_values(fields, "2d")  # Kelvin

        hmdx = thermofeel.calculate_humidex(t2_k=t2m, td_k=td)  # Kelvin

        return self.create_surface_output(
            hmdx, metadata_intensity(fields), paramId="261016"
        )

    @metered("rhp", out=logger.debug)
    def calc_rhp(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        td = field_values(fields, "2d")  # Kelvin

        rhp = thermofeel.calculate_relative_humidity_percent(t2_k=t2m, td_k=td)  # %

        return self.create_output(rhp, metadata_intensity(fields), paramId="260242")

    @metered("heatx", out=logger.debug)
    def calc_heatx(self, fields):

        t2m = field_values(fields, "2t")  # Kelvin
        td = field_values(fields, "2d")  # Kelvin

        heatx = thermofeel.calculate_heat_index_adjusted(t2_k=t2m, td_k=td)  # Kelvin

        return self.create_surface_output(
            heatx, metadata_intensity(fields), paramId="260004"
        )

    def field_stats(self, name, values):
        try:
            unit = units[name]
        except:
            raise ValueError(f"No units specified for parameter {name}")

        logger.debug(
            f"{name:<8} {unit:<6} min {np.nanmin(values):>16.6f} max {np.nanmax(values):>16.6f} "
            f"avg {np.nanmean(values):>16.6f} stddev {np.nanstd(values, dtype=np.float64):>16.6f} "
            f"missing {np.count_nonzero(np.isnan(values)):>8}"
        )

    @metered("dsrp", out=logger.debug)
    def calc_dsrp(self, fields):
        """
        In the absence of dsrp, approximate it with fdir and cossza.
        Note this introduces some amount of error as cossza approaches zero
        """
        # Will use dsrp if available, otherwise approximate it
        if "dsrp" in fields.indices()["param"]:
            return fields.sel(param="dsrp")

        fdir = field_values(fields, "fdir")  # W/m2
        cossza = self.calc_field("uvcossza", self.calc_cossza_int, fields).to_array()

        dsrp = thermofeel.approximate_dsrp(fdir, cossza)

        return self.create_output(dsrp, metadata_accumulation(fields), paramId="47")

    def calc_field(self, name, func, fields, **kwargs):
        if self.results is not None:
            sel = self.results.sel(param=name)
            if len(sel) != 0:
                return sel

        res = func(fields, **kwargs)

        self.field_stats(name, res.to_array())
        if self.results is None:
            self.results = res
        else:
            self.results += res
        field_values(self.results, name)

        return res

    @metered("utci", out=logger.debug)
    def calc_utci(self, fields, *, print_misses=True, validate=True):

        lats, lons = latlon(fields)

        t2m = field_values(fields, "2t")  # Kelvin
        t2d = field_values(fields, "2d")  # Kelvin

        ws = self.calc_field("10si", self.calc_ws, fields).to_array()  # m/s
        mrt = self.calc_field("mrt", self.calc_mrt, fields).to_array()  # Kelvin

        ehPa = compute_ehPa(t2m, t2d)

        utci = thermofeel.calculate_utci(t2_k=t2m, va=ws, mrt=mrt, ehPa=ehPa)  # Kelvin

        for index in range(len(utci)):
            missing = find_utci_missing_values(
                t2m[index],
                ws[index],
                mrt[index],
                ehPa[index],
                utci[index],
                print_misses,
            )

            if validate:
                validate_utci(utci[index], missing, lats, lons)

            utci[index][missing] = np.nan

        return self.create_surface_output(
            utci, metadata_intensity(fields), paramId="261001"
        )

    @metered("wbgt", out=logger.debug)
    def calc_wbgt(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        t2d = field_values(fields, "2d")  # Kelvin

        ws = self.calc_field("10si", self.calc_ws, fields).to_array()  # m/s
        mrt = self.calc_field("mrt", self.calc_mrt, fields).to_array()  # Kelvin

        wbgt = thermofeel.calculate_wbgt(t2m, mrt, ws, t2d)  # Kelvin

        return self.create_surface_output(
            wbgt, metadata_intensity(fields), paramId="261014"
        )

    @metered("gt", out=logger.debug)
    def calc_gt(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin

        ws = self.calc_field("10si", self.calc_ws, fields).to_array()  # m/s
        mrt = self.calc_field("mrt", self.calc_mrt, fields).to_array()  # Kelvin

        gt = thermofeel.calculate_bgt(t2m, mrt, ws)  # Kelvin

        return self.create_surface_output(
            gt, metadata_intensity(fields), paramId="261015"
        )

    @metered("wbpt", out=logger.debug)
    def calc_wbpt(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin

        rhp = self.calc_field("2r", self.calc_rhp, fields).to_array()  # %

        wbpt = thermofeel.calculate_wbt(t2_k=t2m, rh=rhp)  # Kelvin

        return self.create_surface_output(
            wbpt, metadata_intensity(fields), paramId="261022"
        )

    @metered("nefft", out=logger.debug)
    def calc_nefft(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        ws = self.calc_field("10si", self.calc_ws, fields).to_array()  # m/s
        rhp = self.calc_field("2r", self.calc_rhp, fields).to_array()  # %

        nefft = thermofeel.calculate_normal_effective_temperature(
            t2m, ws, rhp
        )  # Kelvin

        return self.create_surface_output(
            nefft, metadata_intensity(fields), paramId="261018"
        )

    @metered("wcf", out=logger.debug)
    def calc_wcf(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin

        ws = self.calc_field("10si", self.calc_ws, fields).to_array()  # m/s

        wcf = thermofeel.calculate_wind_chill(t2m, ws)  # Kelvin

        return self.create_surface_output(
            wcf, metadata_intensity(fields), paramId="260005"
        )

    @metered("aptmp", out=logger.debug)
    def calc_aptmp(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        rhp = self.calc_field("2r", self.calc_rhp, fields).to_array()  # %
        ws = self.calc_field("10si", self.calc_ws, fields).to_array()  # m/s

        aptmp = thermofeel.calculate_apparent_temperature(
            t2_k=t2m, va=ws, rh=rhp
        )  # Kelvin

        return self.create_surface_output(
            aptmp, metadata_intensity(fields), paramId="260255"
        )

    @metered("mrt", out=logger.debug)
    def calc_mrt(self, fields):

        cossza = self.calc_field("uvcossza", self.calc_cossza_int, fields).to_array()
        dsrp = self.calc_field("dsrp", self.calc_dsrp, fields).to_array()

        delta = step_interval(fields)
        seconds_in_time_step = delta * 3600  # steps are in hours

        f = 1.0 / float(seconds_in_time_step)

        ssrd = field_values(fields, "ssrd")  # W/m2
        fdir = field_values(fields, "fdir")  # W/m2
        strd = field_values(fields, "strd")  # W/m2
        strr = field_values(fields, "str")  # W/m2
        ssr = field_values(fields, "ssr")  # W/m2

        self.field_stats("ssrd", ssrd)

        # remove negative values from deaccumulated solar fields
        for v in ssrd, fdir, strd, ssr:
            v[v < 0] = 0

        self.field_stats("dsrp", dsrp)
        self.field_stats("ssrd", ssrd)
        self.field_stats("fdir", fdir)
        self.field_stats("strd", strd)
        self.field_stats("str", strr)
        self.field_stats("ssr", ssr)

        mrt = thermofeel.calculate_mean_radiant_temperature(
            ssrd * f, ssr * f, dsrp * f, strd * f, fdir * f, strr * f, cossza
        )  # Kelvin

        return self.create_surface_output(
            mrt, metadata_intensity(fields), paramId="261002"
        )
