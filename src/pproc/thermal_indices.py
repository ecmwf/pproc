#!/usr/bin/env python3

# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Note: This script is intended only as example usage of thermofeel library.
#       It is designed to be used with ECMWF forecast data.
#       The function ifs_step_intervals() is used to calculate the time interval based on the forecast step.
#       This is particular to the IFS model and ECMWF's NWP operational system.

import sys
import os
import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path
from codetiming import Timer
import psutil

import numpy as np
import thermofeel as thermofeel

import earthkit.data
import earthkit.meteo.solar

__version__ = "2.0.0"

###########################################################################################################

# Constants

UTCI_MIN_VALUE = thermofeel.celsius_to_kelvin(-80)
UTCI_MAX_VALUE = thermofeel.celsius_to_kelvin(90)

MRT_DIFF_HIGH = 150
MRT_DIFF_LOW = -30

###########################################################################################################

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
    "rhp": "%",
    "ws": "m/s",
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


def field_stats(name, values):

    if name in misses:
        values[misses[name]] = np.nan

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

def field_values(fields, param):
    assert param in fields.indices()['param']
    return fields.sel(param=param)[0].values


###########################################################################################################

@Timer(name="cossza", logger=None)
def calc_cossza_int(fields):

    lats, lons = latlon(fields)

    assert lats.size == lons.size
    assert fields[0].values.size == lats.size

    basetime, validtime = get_datetime(fields)

    integration_start, step, delta = integration_interval(fields) # in hours

    # print(f"integration_start {integration_start} step {step} delta {delta}")

    tbegin = integration_start
    tend   = step

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

    return cossza


@Timer(name="dsrp", logger=None)
def approximate_dsrp(fields):
    """
    In the absence of dsrp, approximate it with fdir and cossza.
    Note this introduces some amount of error as cossza approaches zero
    """
    fdir = field_values(fields, "fdir")  # W/m2
    cossza = calc_field("cossza", calc_cossza_int, fields)

    dsrp = thermofeel.approximate_dsrp(fdir, cossza)

    return dsrp


@Timer(name="heatx", logger=None)
def calc_heatx(fields):

    t2m = field_values(fields,"2t")  # Kelvin
    td  = field_values(fields,"2d")  # Kelvin

    heatx = thermofeel.calculate_heat_index_adjusted(t2_k=t2m, td_k=td)  # Kelvin

    return heatx


@Timer(name="aptmp", logger=None)
def calc_aptmp(fields):
    t2m = field_values(fields,"2t")  # Kelvin
    rhp = calc_field("rhp", calc_rhp, fields)  # %
    ws = calc_field("ws", calc_ws, fields)  # m/s

    aptmp = thermofeel.calculate_apparent_temperature(t2_k=t2m, va=ws, rh=rhp)  # Kelvin

    return aptmp


@Timer(name="hmdx", logger=None)
def calc_hmdx(fields):
    t2m = field_values(fields,"2t")  # Kelvin
    td = field_values(fields,"2d")  # Kelvin

    hmdx = thermofeel.calculate_humidex(t2_k=t2m, td_k=td)  # Kelvin

    return hmdx


@Timer(name="rhp", logger=None)
def calc_rhp(fields):
    t2m = field_values(fields,"2t")  # Kelvin
    td = field_values(fields,"2d")  # Kelvin

    rhp = thermofeel.calculate_relative_humidity_percent(t2_k=t2m, td_k=td)

    return rhp  # %


@Timer(name="mrt", logger=None)
def calc_mrt(fields):
    
    _, _, delta = integration_interval(fields)

    cossza = calc_field("cossza", calc_cossza_int, fields)

    # will use dsrp if available, otherwise approximate it
    # print(fields.indices()['param'])
    if 'dsrp' in fields.indices()['param']:
        dsrp = field_values(fields,"dsrp")
    else:
        dsrp = calc_field("dsrp", approximate_dsrp, fields)

    seconds_in_time_step = delta * 3600  # steps are in hours

    f = 1.0 / float(seconds_in_time_step)

    ssrd = field_values(fields,"ssrd")   # W/m2
    fdir = field_values(fields, "fdir")  # W/m2
    strd = field_values(fields, "strd")  # W/m2
    strr = field_values(fields, "str")   # W/m2
    ssr  = field_values(fields, "ssr")   # W/m2

    field_stats("ssrd", ssrd)

    # remove negative values from deaccumulated solar fields
    for v in ssrd, fdir, strd, ssr:
        v[v < 0] = 0

    field_stats("dsrp", dsrp)
    field_stats("ssrd", ssrd)
    field_stats("fdir", fdir)
    field_stats("strd", strd)
    field_stats("str", strr)
    field_stats("ssr", ssr)

    mrt = thermofeel.calculate_mean_radiant_temperature(
        ssrd * f, ssr * f, dsrp * f, strd * f, fdir * f, strr * f, cossza
    )  # Kelvin

    return mrt


def calc_field(name, func, fields):
    if name in results:
        return results[name]

    values = func(fields)

    field_stats(name, values)
    results[name] = values  # cache results -- this should be a FieldList when append works properly

    return values


@Timer(name="ws", logger=None)
def calc_ws(fields):
    u10 = field_values(fields,"10u")  # m/s
    v10 = field_values(fields,"10v")  # m/s

    return np.sqrt(u10**2 + v10**2)  # m/s


def compute_ehPa_(rh_pc, svp):
    return svp * rh_pc * 0.01  # / 100.0


@Timer(name="ehPa", logger=None)
def compute_ehPa(t2m, t2d):
    rh_pc = thermofeel.calculate_relative_humidity_percent(t2m, t2d)
    svp = thermofeel.calculate_saturation_vapour_pressure(t2m)
    ehPa = compute_ehPa_(rh_pc, svp)
    return ehPa


def find_utci_missing_values(t2m, va, mrt, ehPa, utci):
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

    if args.utci_misses:
        print(
            f"UTCI nmisses {nmisses} NANs {nnan} T2>70C {nt2high} T2<-70 {nt2low} highwind {nhighwind} nehpa {nehpa} MRT-T2>{MRT_DIFF_HIGH} {ndiffmrt} MRT-T2<{MRT_DIFF_LOW} {ndiffmrtneg}"
        )

    return misses


def validate_utci(utci, misses, lats, lons):

    out_of_bounds = 0
    nans = 0
    for i in range(len(utci)):
        v = utci[i]
        if v < UTCI_MIN_VALUE or v > UTCI_MAX_VALUE:
            out_of_bounds += 1
            print("UTCI [", i, "] = ", utci[i], " : lat/lon ", lats[i], lons[i])
        if np.isnan(v):
            nans += 1
            print("UTCI [", i, "] = ", utci[i], " : lat/lon ", lats[i], lons[i])

    nmisses = len(misses)
    if nmisses > 0 or out_of_bounds > 0 or nans > 0:
        print(f"UTCI => nmisses {nmisses} out_of_bounds {out_of_bounds} NANs {nans}")


@Timer(name="utci", logger=None)
def calc_utci(fields):

    lats, lons = latlon(fields)

    t2m = field_values(fields,"2t")  # Kelvin
    t2d = field_values(fields,"2d")  # Kelvin

    ws = calc_field("ws", calc_ws, fields)  # m/s
    mrt = calc_field("mrt", calc_mrt, fields)  # Kelvin

    ehPa = compute_ehPa(t2m, t2d)

    utci = thermofeel.calculate_utci(
        t2_k=t2m, va=ws, mrt=mrt, ehPa=ehPa
    )  #  Kelvin

    missing = find_utci_missing_values(t2m, ws, mrt, ehPa, utci)

    if args.validateutci:
        validate_utci(utci, missing, lats, lons)

    utci[missing] = np.nan
    misses["utci"] = missing

    return utci


@Timer(name="wbgt", logger=None)
def calc_wbgt(fields):
    t2m = field_values(fields,"2t")  # Kelvin
    t2d = field_values(fields,"2d")  # Kelvin

    ws = calc_field("ws", calc_ws, fields)  # m/s
    mrt = calc_field("mrt", calc_mrt, fields)  # Kelvin

    wbgt = thermofeel.calculate_wbgt(t2m, mrt, ws, t2d)  # Kelvin

    return wbgt


@Timer(name="gt", logger=None)
def calc_gt(fields):
    t2m = field_values(fields,"2t")  # Kelvin

    ws = calc_field("ws", calc_ws, fields)  # m/s
    mrt = calc_field("mrt", calc_mrt, fields)  # Kelvin

    gt = thermofeel.calculate_bgt(t2m, mrt, ws)  # Kelvin

    return gt


@Timer(name="wbpt", logger=None)
def calc_wbpt(fields):
    t2m = field_values(fields,"2t")  # Kelvin

    rhp = calc_field("rhp", calc_rhp, fields)  # %

    wbpt = thermofeel.calculate_wbt(t2_k=t2m, rh=rhp)  # Kelvin

    return wbpt


@Timer(name="nefft", logger=None)
def calc_nefft(fields):
    t2m = field_values(fields,"2t")  # Kelvin
    ws = calc_field("ws", calc_ws, fields)  # m/s
    rhp = calc_field("rhp", calc_rhp, fields)  # %

    nefft = thermofeel.calculate_normal_effective_temperature(t2m, ws, rhp)  # Kelvin
    
    return nefft


@Timer(name="wcf", logger=None)
def calc_wcf(fields):
    t2m = field_values(fields,"2t")  # Kelvin

    ws = calc_field("ws", calc_ws, fields)  # m/s

    wcf = thermofeel.calculate_wind_chill(t2m, ws)  # Kelvin

    return wcf


###############################################################################

def check_field_sizes(fields):
    all(f.values.shape == fields[0].values.shape for f in fields)


def get_datetime(fields):
    dt = fields.sel(param='2t').datetime()
    base_time = dt["base_time"][0]
    valid_time = dt["valid_time"][0]
    assert all(x == valid_time for x in fields.datetime()["valid_time"]) # verify valid time all same
    return base_time, valid_time


def append_newfield(output, metadata, paramid, values, missing=None):
    
    newmd = metadata.override(edition=2, paramId=paramid)
    newfield = earthkit.data.FieldList.from_numpy(values, newmd)

    # missing values are set by having a NaN in the values array

    output += newfield

    # print(output.ls(namespace="mars"))
    return output

def ifs_step_intervals(step, mclass, mstream):
    """Computes the time integration interval for the IFS forecasting system given a forecast output step"""

    if mclass == "od" and mstream == "oper":
        assert step <= 240
        if step > 0:
            if step <= 90:
                return step - 1
            else:
                if step <= 144:
                    return step - 3
                else:
                    return step - 6
        else:
            return step

    if mclass == "od" and mstream == "enfo":
        # currently setup for 3-hourly post-processing
        assert step <= 360
        if step > 0:
            if step <= 144:
                return step - 3
            else:
                return step - 6
        else:
            return step

    if mclass == "od" and mstream == "mmsf":
        assert step <= 5160
        if step > 0:
            return step - 24
        else:
            return step
    
    if mclass == "ea" and mstream == "oper":
        # currently setup for 1-hourly post-processing
        assert step <= 18
        if step > 0:
            return step - 1
        else:
            return step

    if mclass == "ea" and mstream == "enda":
        # currently setup for 3-hourly post-processing
        assert step <= 18
        if step > 0:
            return step - 3
        else:
            return step

    raise NotImplemented(f"Combination of MARS class {mclass} and stream {mstream} not recognised")

def integration_interval(fields):
    '''Returns the time interval for the integration of the forecast step'''
    
    md = fields.sel(param='2t').metadata(namespace="mars")[0]
    
    step = md["step"]  # end of the forecast integration
    mclass = md["class"]
    mstream = md["stream"]
    
    if args.override_class:
        mclass = args.override_class
    if args.override_stream:
        mstream = args.override_stream

    integration_start = ifs_step_intervals(step, mclass, mstream)  # start of forecast integration step
    delta = step - integration_start

    return integration_start, step, delta


def metadata_intensity(fields):
    md = fields.sel(param='2t').metadata()[0]
    return md

def metadata_wind(fields):
    md = fields.sel(param='10u').metadata()[0]
    return md

def metadata_accumulation(fields):
    md = fields.sel(param='fdir').metadata()[0]
    return md


@Timer(name="proc_step", logger=None)
def process_step(args, fields, output):

    check_field_sizes(fields)
    basetime, validtime = get_datetime(fields)
    integration_start, step, delta = integration_interval(fields)
    
    time = basetime.hour
    print(
        f"validtime {validtime.isoformat()} - basetime {basetime.date().isoformat()} : time {time} step {step}"
    )

    global results
    global misses

    results = {}
    misses = {}

    # Windspeed - shortName ws
    if args.ws:
        ws = calc_field("ws", calc_ws, fields)
        output = append_newfield(output, metadata_wind(fields), "10", ws)

    # Cosine of Solar Zenith Angle - shortName uvcossza - ECMWF product
    # TODO: 214001 only exists for GRIB1 -- but here we use it for GRIB2 (waiting for WMO)
    if args.cossza:
        cossza = calc_field("cossza", calc_cossza_int, fields)
        output = append_newfield(output, metadata_intensity(fields), "214001", cossza)

    # direct solar radiation - shortName dsrp - ECMWF product
    if args.dsrp:
        dsrp = calc_field("dsrp", approximate_dsrp, fields)
        output = append_newfield(output, metadata_accumulation(fields), "47", dsrp)

    # Mean Radiant Temperature - shortName mrt - ECMWF product
    if args.mrt or args.all:
        mrt = calc_field("mrt", calc_mrt, fields)
        output = append_newfield(output, metadata_intensity(fields), "261002", mrt)

    # Univeral Thermal Climate Index - shortName utci - ECMWF product
    if args.utci or args.all:
        utci = calc_field("utci", calc_utci, fields)
        output = append_newfield(output, metadata_intensity(fields), "261001", utci)

    # Heat Index (adjusted) - shortName heatx - ECMWF product
    if args.heatx or args.all:
        heatx = calc_field("heatx", calc_heatx, fields)
        output = append_newfield(output, metadata_intensity(fields), "260004", heatx)

    # Wind Chill factor - shortName wcf - ECMWF product
    if args.wcf or args.all:
        wcf = calc_field("wcf", calc_wcf, fields)
        output = append_newfield(output, metadata_intensity(fields), "260005", wcf)

    # Apparent Temperature - shortName aptmp - ECMWF product
    if args.aptmp or args.all:
        aptmp = calc_field("aptmp", calc_aptmp, fields)
        output = append_newfield(output, metadata_intensity(fields), "260255", aptmp)

    # Relative humidity percent at 2m - shortName 2r - ECMWF product
    if args.rhp or args.all:
        rhp = calc_field("rhp", calc_rhp, fields)
        output = append_newfield(output, metadata_intensity(fields), "260242", rhp)

    # Humidex - shortName hmdx 
    if args.hmdx or args.all:
        hmdx = calc_field("hmdx", calc_hmdx, fields)
        output = append_newfield(output, metadata_intensity(fields), "261016", hmdx)

    # Normal Effective Temperature - shortName nefft 
    if args.nefft or args.all:
        nefft = calc_field("nefft", calc_nefft, fields)
        output = append_newfield(output, metadata_intensity(fields), "261018", nefft)

    # Globe Temperature - shortName gt
    if args.gt or args.all:
        gt = calc_field("gt", calc_gt, fields)
        output = append_newfield(output, metadata_intensity(fields), "261015", gt)

    # Wet-bulb potential temperature - shortName wbpt 
    if args.wbpt or args.all:
        wbpt = calc_field("wbpt", calc_wbpt, fields)
        output = append_newfield(output, metadata_intensity(fields), "261022", wbpt)

    # Wet Bulb Globe Temperature - shortName wbgt 
    if args.wbgt or args.all:  #
        wbgt = calc_field("wbgt", calc_wbgt, fields)
        output = append_newfield(output, metadata_intensity(fields), "261014", wbgt)

    # effective temperature 261017
    # standard effective temperature 261019

    return output, step


def load_input():
    
    if args.input_file:
        f = open(args.input_file, "rb")
        ds = earthkit.data.from_source("stream", f, group_by=["step", "level"]) 
        return ds
    
    if args.input_fdb:
        req = {k:v.split('/') for k,v in [y.split('=') for y in args.input_fdb.split(',')]}
        # print(f"Parsed request {args.request} into: {req}")
        ds = earthkit.data.from_source("fdb", req, stream=True, group_by=["step", "level"])
        return ds
    
    raise ValueError("No input specified")


def save_grib_file(path, output):
    output.save(path, append=True)


def command_line_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--accelerate", help="accelerate computations using JAX JIT", action="store_true")

    parser.add_argument("output", help="output file with GRIB messages")

    parser.add_argument(
        "--all", help="compute all available indices", action="store_true"
    )

    parser.add_argument(
        "--validateutci",
        help="validate utci by detecting nans and out of bounds values. NOT to use in production. Very verbose option.",
        action="store_true",
    )

    parser.add_argument(
        "--ws", help="compute wind speed from components", action="store_true"
    )
    parser.add_argument(
        "--cossza",
        help="compute Cosine of Solar Zenith Angle (cossza)",
        action="store_true",
    )
    parser.add_argument(
        "--dsrp",
        help="compute dsrp (approximated)",
        action="store_true",
    )
    parser.add_argument("--mrt", help="compute mrt", action="store_true")
    parser.add_argument(
        "--utci",
        help="compute UTCI Universal Thermal Climate Index",
        action="store_true",
    )
    parser.add_argument(
        "--heatx", help="compute Heat Index (adjusted)", action="store_true"
    )
    parser.add_argument("--wcf", help="compute wcf factor", action="store_true")
    parser.add_argument(
        "--aptmp", help="compute Apparent Temperature", action="store_true"
    )
    parser.add_argument(
        "--rhp", help="compute relative humidity percent", action="store_true"
    )

    parser.add_argument("--hmdx", help="compute humidex", action="store_true")
    parser.add_argument(
        "--nefft", help="compute net effective temperature", action="store_true"
    )

    # TODO: these outputs are not yet in WMO GRIB2 recognised parameters
    parser.add_argument(
        "--wbgt", help="compute Wet Bulb Globe Temperature", action="store_true"
    )
    parser.add_argument("--gt", help="compute  Globe Temperature", action="store_true")
    parser.add_argument(
        "--wbpt", help="compute Wet Bulb Temperature", action="store_true"
    )

    parser.add_argument("--override-class", help="override MARS class", type=str)
    parser.add_argument("--override-stream", help="override MARS stream", type=str)

    parser.add_argument("--input-file", help="input file with GRIB messages", type=Path)
    parser.add_argument("--input-fdb", help="input with FDB MARS request, eg. class=od,stream=oper,param=2t/2d", type=str)

    parser.add_argument("--timers", help="print function performance timers at the end", action="store_true")
    parser.add_argument("--usage", help="print cpu and memory usage during run", action="store_true")
    parser.add_argument("--utci-misses", help="print missing values for UTCI", action="store_true")
    
    return parser.parse_args()


def latlon(fields):
    latlon = fields[0].to_latlon(flatten=True)
    lat = latlon["lat"]
    lon = latlon["lon"]
    assert lat.size == lon.size
    return lat, lon

def print_timers():
    print("Performance summary:")
    print("--------------------")
    for t in Timer.timers.items():
        func, stats = t
        count = Timer.timers.count(func)
        if count > 0:
            mean = Timer.timers.mean(func)
            tmin = Timer.timers.min(func)
            tmax = Timer.timers.max(func)
            stdev = Timer.timers.stdev(func) if count > 1 else 0.0
            print(f"{func:<10} calls {count:>4}  --  avg + stdev [{mean:>8.4f} , {stdev:>8.4f}]s  --  min + max [{tmin:>8.4f} , {tmax:>8.4f}] s")


def print_usage():
    load1, load5, load15 = psutil.getloadavg()
    cpu_usage = (load5/os.cpu_count()) * 100
    sysmem = psutil.virtual_memory().used / 1024**3 # in GiB
    sysmemperc = psutil.virtual_memory().percent
    procmem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3 # in GiB
    procmemperc = psutil.Process(os.getpid()).memory_percent()
    print(f"[INFO] usage: cpu load {cpu_usage:5.1f}% -- proc mem {procmem:3.1f}GiB {procmemperc:3.1f}% -- sys mem {sysmem:3.1f}GiB {sysmemperc}%")


def main():

    global args
    args = command_line_options()

    print(f"Compute Thermal Indices: {__version__}")
    print(f"thermofeel: {thermofeel.__version__}")
    print(f"earthkit.data: {earthkit.data.__version__}")
    print(f"Numpy: {np.version.version}")
    print(f"Python: {sys.version}")
    # np.show_config()

    input = load_input()

    if os.path.exists(args.output):
        os.unlink(args.output)

    print("----------------------------------------")


    steps = []
    for fields in input:

        print('Input:')
        print(fields.ls(namespace="mars"))

        output = earthkit.data.FieldList()

        output, step = process_step(args, fields, output)
        
        steps.append(step)

        if args.usage: print_usage()
            
        print(f"[INFO] appending {len(output)} fields to {args.output}")
        save_grib_file(args.output, output)
        
        print("----------------------------------------")


    print(f"[INFO] Processed steps {steps}\n")

    if args.timers:
        print_timers()

    return 0


if __name__ == "__main__":
    sys.exit(main())
