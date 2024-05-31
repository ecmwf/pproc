from datetime import timedelta

import earthkit.data
import earthkit.meteo.solar
import numpy as np
import thermofeel
from codetiming import Timer

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


def metadata_intensity(fields):
    return fields.sel(param="2t").metadata()[0]


def metadata_wind(fields):
    return fields.sel(param="10u").metadata()[0]


def metadata_accumulation(fields):
    return fields.sel(param="fdir").metadata()[0]


class ComputeIndices:
    def __init__(self, out_keys={}):
        self.out_keys = out_keys
        self.results = earthkit.data.FieldList()
        self.misses = {}

    def create_output(self, values, template, paramId):
        return earthkit.data.FieldList.from_numpy(
            values, template.override(**self.out_keys, paramId=paramId)
        )

    @Timer(name="cossza", logger=None)
    def calc_cossza_int(self, fields):

        lats, lons = latlon(fields)

        basetime, validtime = get_datetime(fields)

        delta = step_interval(fields)

        dtbegin = validtime - timedelta(hours=delta)
        dtend = validtime

        # print(f"computing cossza @ {validtime} tbegin {tbegin} tend {tend}")

        cossza = earthkit.meteo.solar.cos_solar_zenith_angle_integrated(
            latitudes=lats,
            longitudes=lons,
            begin_date=dtbegin,
            end_date=dtend,
            integration_order=2,
        )

        return self.create_output(cossza, metadata_intensity(fields), "214001")

    @Timer(name="ws", logger=None)
    def calc_ws(self, fields):
        u10 = field_values(fields, "10u")  # m/s
        v10 = field_values(fields, "10v")  # m/s

        ws = np.sqrt(u10**2 + v10**2)  # m/s
        return self.create_output(ws, metadata_wind(fields), "10")

    @Timer(name="hmdx", logger=None)
    def calc_hmdx(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        td = field_values(fields, "2d")  # Kelvin

        hmdx = thermofeel.calculate_humidex(t2_k=t2m, td_k=td)  # Kelvin

        return self.create_output(hmdx, metadata_intensity(fields), "261016")

    @Timer(name="rhp", logger=None)
    def calc_rhp(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        td = field_values(fields, "2d")  # Kelvin

        rhp = thermofeel.calculate_relative_humidity_percent(t2_k=t2m, td_k=td)  # %

        return self.create_output(rhp, metadata_intensity(fields), "260242")

    @Timer(name="heatx", logger=None)
    def calc_heatx(self, fields):

        t2m = field_values(fields, "2t")  # Kelvin
        td = field_values(fields, "2d")  # Kelvin

        heatx = thermofeel.calculate_heat_index_adjusted(t2_k=t2m, td_k=td)  # Kelvin

        return self.create_output(heatx, metadata_intensity(fields), "260004")

    def field_stats(self, name, values):
        if name in self.misses:
            values[self.misses[name]] = np.nan

        if name in units:
            unit = units[name]
        else:
            print(f"unknown unit for parameter {name}")
            raise ValueError

        print(
            f"{name:<8} {unit:<6} min {np.nanmin(values):>16.6f} max {np.nanmax(values):>16.6f} "
            f"avg {np.nanmean(values):>16.6f} stddev {np.nanstd(values, dtype=np.float64):>16.6f} "
            f"missing {np.count_nonzero(np.isnan(values)):>8}"
        )

    @Timer(name="dsrp", logger=None)
    def approximate_dsrp(self, fields):
        """
        In the absence of dsrp, approximate it with fdir and cossza.
        Note this introduces some amount of error as cossza approaches zero
        """
        fdir = field_values(fields, "fdir")  # W/m2
        cossza = self.calc_field("cossza", self.calc_cossza_int, fields)[0].values

        dsrp = thermofeel.approximate_dsrp(fdir, cossza)

        return self.create_output(dsrp, metadata_accumulation(fields), "47")

    def calc_field(self, name, func, fields, **kwargs):
        sel = fields.sel(param=name)
        if len(sel) != 0:
            return sel

        res = func(fields, **kwargs)

        self.field_stats(name, res[0].values)
        self.results += res

        return res

    @Timer(name="utci", logger=None)
    def calc_utci(self, fields, *, print_misses=True, validate=True):

        lats, lons = latlon(fields)

        t2m = field_values(fields, "2t")  # Kelvin
        t2d = field_values(fields, "2d")  # Kelvin

        ws = self.calc_field("ws", self.calc_ws, fields)[0].values  # m/s
        mrt = self.calc_field("mrt", self.calc_mrt, fields)[0].values  # Kelvin

        ehPa = compute_ehPa(t2m, t2d)

        utci = thermofeel.calculate_utci(t2_k=t2m, va=ws, mrt=mrt, ehPa=ehPa)  # Kelvin

        missing = find_utci_missing_values(t2m, ws, mrt, ehPa, utci, print_misses)

        if validate:
            validate_utci(utci, missing, lats, lons)

        utci[missing] = np.nan
        self.misses["utci"] = missing

        return self.create_output(utci, metadata_intensity(fields), "261001")

    @Timer(name="wbgt", logger=None)
    def calc_wbgt(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        t2d = field_values(fields, "2d")  # Kelvin

        ws = self.calc_field("ws", self.calc_ws, fields)[0].values  # m/s
        mrt = self.calc_field("mrt", self.calc_mrt, fields)[0].values  # Kelvin

        wbgt = thermofeel.calculate_wbgt(t2m, mrt, ws, t2d)  # Kelvin

        return self.create_output(wbgt, metadata_intensity(fields), "261014")

    @Timer(name="gt", logger=None)
    def calc_gt(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin

        ws = self.calc_field("ws", self.calc_ws, fields)[0].values  # m/s
        mrt = self.calc_field("mrt", self.calc_mrt, fields)[0].values  # Kelvin

        gt = thermofeel.calculate_bgt(t2m, mrt, ws)  # Kelvin

        return self.create_output(gt, metadata_intensity(fields), "261015")

    @Timer(name="wbpt", logger=None)
    def calc_wbpt(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin

        rhp = self.calc_field("rhp", self.calc_rhp, fields)[0].values  # %

        wbpt = thermofeel.calculate_wbt(t2_k=t2m, rh=rhp)  # Kelvin

        return self.create_output(wbpt, metadata_intensity(fields), "261022")

    @Timer(name="nefft", logger=None)
    def calc_nefft(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        ws = self.calc_field("ws", self.calc_ws, fields)[0].values  # m/s
        rhp = self.calc_field("rhp", self.calc_rhp, fields)[0].values  # %

        nefft = thermofeel.calculate_normal_effective_temperature(
            t2m, ws, rhp
        )  # Kelvin

        return self.create_output(nefft, metadata_intensity(fields), "261018")

    @Timer(name="wcf", logger=None)
    def calc_wcf(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin

        ws = self.calc_field("ws", self.calc_ws, fields)[0].values  # m/s

        wcf = thermofeel.calculate_wind_chill(t2m, ws)  # Kelvin

        return self.create_output(wcf, metadata_intensity(fields), "260005")

    @Timer(name="aptmp", logger=None)
    def calc_aptmp(self, fields):
        t2m = field_values(fields, "2t")  # Kelvin
        rhp = self.calc_field("rhp", self.calc_rhp, fields)[0].values  # %
        ws = self.calc_field("ws", self.calc_ws, fields)[0].values  # m/s

        aptmp = thermofeel.calculate_apparent_temperature(
            t2_k=t2m, va=ws, rh=rhp
        )  # Kelvin

        return self.create_output(aptmp, metadata_intensity(fields), "260255")

    @Timer(name="mrt", logger=None)
    def calc_mrt(self, fields):

        cossza = self.calc_field("cossza", self.calc_cossza_int, fields)[0].values

        # will use dsrp if available, otherwise approximate it
        # print(fields.indices()['param'])
        if "dsrp" in fields.indices()["param"]:
            dsrp = field_values(fields, "dsrp")
        else:
            dsrp = self.calc_field("dsrp", self.approximate_dsrp, fields)[0].values

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

        return self.create_output(mrt, metadata_intensity(fields), "261002")
