#!/usr/bin/env python3
#
# (C) Copyright 1996- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation nor
# does it submit to any jurisdiction.


import argparse
import pickle
import re
from contextlib import ExitStack, nullcontext
from datetime import datetime
from importlib import resources
from itertools import chain, tee
from os import environ, makedirs, path

import eccodes
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.spatial import KDTree

pts_cache_dir = "PTS_CACHE_DIR"
pts_home_dir = "PTS_HOME_DIR"


def previous_and_current(some_iterable):
    prevs, currs = tee(some_iterable, 2)
    prevs = chain([None], prevs)
    return zip(prevs, currs)


def ll_to_ecef(lat, lon, height=0.0, radius=6371229.0):
    lonr = np.radians(lon)
    latr = np.radians(lat)

    x = (radius + height) * np.cos(latr) * np.cos(lonr)
    y = (radius + height) * np.cos(latr) * np.sin(lonr)
    z = (radius + height) * np.sin(latr)
    return x, y, z


def distance_from_overlap(radius, overlap):
    assert 0.0 < radius
    assert 0.0 <= overlap < 1.0
    if overlap <= 0.0:
        return np.inf

    def overlap_unit_circles(d_over_r):
        assert 0.0 <= d_over_r
        if 2.0 <= d_over_r:
            return 0.0

        hd = d_over_r / 2.0
        ha_inter = np.arccos(hd) - hd * np.sqrt(1.0 - hd * hd)
        ha_union = np.pi - ha_inter
        return ha_inter / ha_union

    d = root_scalar(lambda d: overlap_unit_circles(d) - overlap, bracket=[0, 2], x0=1)
    return radius * d.root


def parse_range(rstr):
    s = set()
    for part in rstr.split(","):
        x = part.split("-")
        s.update(range(int(x[0]), int(x[-1]) + 1))
    return sorted(s)


def delta_hours(a: datetime, b: datetime):
    delta = a - b
    return delta.days * 24 + delta.seconds // 3600


def main(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--input-tc-tracks", help="Input TC tracks files", nargs="+")
    group.add_argument("--input-points", help="Input points file", nargs=1)

    parser.add_argument(
        "--input-tc-tracks-no-flip",
        help="TC tracks latitude flip based on file name",
        action="store_false",
        dest="flip",
    )

    parser.add_argument(
        "--input-points-columns",
        help="Point column names to use",
        default=["lat", "lon", "number", "date", "step", "wind", "msl"],
        nargs="+",
    )

    parser.add_argument("--filter-number", help="Filter number range", default="1-50")

    parser.add_argument(
        "--filter-wind", help="Filter minimum wind speed", default=0.0, type=float
    )

    parser.add_argument(
        "--filter-time",
        help="Filter time (date/step) range [h]",
        default=[0.0, float("inf")],
        type=float,
        nargs=2,
    )

    parser.add_argument(
        "--basetime", help="date/time of the model run", default=None, type=str
    )

    parser.add_argument(
        "--distance", help="Search radius [m]", default=300.0e3, type=float
    )

    parser.add_argument(
        "--overlap", help="Search overlap [0, 1[", default=0.7, type=float
    )

    parser.add_argument("--grib-accuracy", help="GRIB accuracy", default=8, type=int)
    parser.add_argument("--grib-date", help="GRIB dataDate", default=None, type=int)
    parser.add_argument("--grib-time", help="GRIB dataTime", default=None, type=int)
    parser.add_argument("--grib-step", help="GRIB stepRange", default=None)
    parser.add_argument("--grib-paramid", help="GRIB paramId", default=None, type=int)

    parser.add_argument(
        "--grib-template",
        help="GRIB template (env. variable '" + pts_home_dir + "')",
        default="O640.grib1",
    )

    parser.add_argument("-v", "--verbosity", action="count", default=0)

    parser.add_argument(
        "--no-caching",
        help="Caching (env. variable '" + pts_cache_dir + "')",
        action="store_false",
        dest="caching",
    )

    parser.add_argument("out", help="Output GRIB file", metavar="OUTPUT_GRIB")

    args = parser.parse_args(args)
    if args.verbosity >= 1:
        print(args)

    dist_circle = distance_from_overlap(args.distance, args.overlap)
    basetime = datetime.fromisoformat(args.basetime) if args.basetime else None
    numbers = parse_range(args.filter_number)

    with ExitStack() as stack:
        if pts_home_dir in environ:
            tpl_dir = environ[pts_home_dir]
            tpl_path = nullcontext(
                path.realpath(path.join(tpl_dir, args.grib_template))
            )
        else:
            tpl_path = stack.enter_context(
                resources.path("pproc.data.pts", args.grib_template)
            )

        print("Loading template: '{}'".format(tpl_path))
        f = stack.enter_context(open(tpl_path, "rb"))
        h = eccodes.codes_grib_new_from_file(f)
        assert h is not None

        N = eccodes.codes_get(h, "numberOfDataPoints")

        # k-d tree
        tree_path = eccodes.codes_get(h, "md5GridSection") + ".tree"
        if pts_cache_dir in environ:
            tree_path = path.join(environ[pts_cache_dir], tree_path)

        if args.caching and path.exists(tree_path):
            print("Loading cache file: '{}'".format(tree_path))
            with open(tree_path, "rb") as f:
                tree = pickle.load(f)
        else:
            it = eccodes.codes_grib_iterator_new(h, 0)

            P = np.empty([N, 3])
            i = 0
            while True:
                result = eccodes.codes_grib_iterator_next(it)
                if not result:
                    break
                [lat, lon, value] = result

                assert i < N
                P[i, :] = ll_to_ecef(lat, lon)

                i += 1

            eccodes.codes_grib_iterator_delete(it)
            tree = KDTree(P)

        if args.caching and not path.exists(tree_path):
            tpl_dir = path.dirname(tree_path)
            if tpl_dir:
                makedirs(tpl_dir, mode=888, exist_ok=True)
                assert path.isdir(tpl_dir)
            with open(tree_path, "wb") as f:
                pickle.dump(tree, f)
            print("Created cache file: '{}'".format(tree_path))

        # input (apply filter_number)
        assert bool(args.input_points) != bool(args.input_tc_tracks)
        if args.input_tc_tracks:
            d = {
                col: []
                for col in ["lat", "lon", "number", "id", "date", "step", "wind", "msl"]
            }

            re_filename = re.compile(r"^\d{4}(\d{10})_(\d{3})_\d{6}_\l{3}$")
            re_split = re.compile(r"^.....  (TD|TS|HR\d)$")
            re_data = re.compile(
                r"^..... (\d{4}/\d{2}/\d{2})/(\d{2})\*(\d{3})(\d{4}) (\d{3}) (\d{4})\*"
                r"(\d{3})(\d{4})\*"
                r"\d{5}\d{5}\d{5}\d{5}\*\d{5}\d{5}\d{5}\d{5}\*\d{5}\d{5}\d{5}\d{5}\*$"
            )

            id = 0
            for fn in args.input_tc_tracks:
                filename = re_filename.search(path.basename(fn))
                if filename and not basetime:
                    basetime = datetime.strptime(filename.group(1), "%Y%m%d%H")
                number = int(filename.group(2)) if filename else 1
                if number not in numbers:
                    continue

                flip = args.flip and fn.endswith(("_aus", "_sin", "_spc"))

                with open(fn, "r") as file:
                    for line in file:
                        if re_split.search(line):
                            id += 1
                            continue
                        data = re_data.search(line)
                        if data:
                            d["lat"].append((0.1, -0.1)[flip] * float(data.group(3)))
                            d["lon"].append(0.1 * float(data.group(4)))
                            d["number"].append(number)
                            d["id"].append(id)
                            d["date"].append(data.group(1).replace("/", ""))
                            d["step"].append(data.group(2))
                            d["wind"].append(float(data.group(5)))
                            d["msl"].append(float(data.group(6)))
            df = pd.DataFrame(d)

        elif args.input_points:
            df = pd.read_csv(
                args.input_points,
                sep=r"\s+",
                header=None,
                comment="#",
                names=args.input_points_columns,
                usecols=["lat", "lon", "number", "date", "step", "wind", "msl"],
            )
            df = df[df.number.isin(numbers)]
            df["id"] = df.number

        # pre-process (apply filter_time and calculate/drop columns)
        datestep = [
            datetime.strptime(k, "%Y%m%d %H")
            for k in (df.date.astype(str) + " " + df.step.astype(str))
        ]
        if not basetime:
            basetime = min(datestep)
        df["t"] = [delta_hours(ds, basetime) for ds in datestep]
        df = df[(args.filter_time[0] <= df.t) & (df.t <= args.filter_time[1])]

        df["x"], df["y"], df["z"] = ll_to_ecef(df.lat, df.lon)
        df.drop(["lat", "lon", "date", "step"], axis=1, inplace=True)

        # probability field
        val = np.zeros(N)
        for id in set(df.id.tolist()):
            track = df[df.id == id].sort_values("t")

            # super-sampled time and position (apply filter_wind)
            ti = np.array([])
            tend = None
            npoints = 1
            for a, b in previous_and_current(track.itertuples()):
                if a is not None and args.filter_wind <= max(a.wind, b.wind):
                    tend = b.t
                    npoints += 1
                    dist_ab = np.linalg.norm(
                        np.array([b.x - a.x, b.y - a.y, b.z - a.z])
                    )
                    num = max(1, int(np.ceil(dist_ab / dist_circle)))
                    ti = np.append(ti, np.linspace(a.t, b.t, num=num, endpoint=False))
            if not tend:
                continue
            ti = np.append(ti, tend)

            print(
                f"segments={npoints-1}/{track.shape[0]-1} "
                f"len={len(ti)} "
                f"ss={round(float(len(ti)) / npoints, 1)}x"
            )

            xi = np.interp(ti, track.t, track.x)
            yi = np.interp(ti, track.t, track.y)
            zi = np.interp(ti, track.t, track.z)

            # track points
            pts = set()
            for p in zip(xi, yi, zi):
                pts.update(tree.query_ball_point(p, r=args.distance))
            for i in pts:
                assert i < N
                val[i] = val[i] + 1.0

        if numbers:
            val = np.minimum(val / len(numbers), 1.0) * 100.0  # %

        # write results
        if args.grib_accuracy:
            eccodes.codes_set(h, "accuracy", args.grib_accuracy)
        if args.grib_date:
            eccodes.codes_set(h, "dataDate", args.grib_date)
        if args.grib_time:
            eccodes.codes_set(h, "dataTime", args.grib_time)
        if args.grib_step:
            eccodes.codes_set(h, "stepRange", args.grib_step)
        if args.grib_paramid:
            eccodes.codes_set(h, "paramId", args.grib_paramid)

        eccodes.codes_set_values(h, val)

        with open(args.out, "wb") as f:
            eccodes.codes_write(h, f)

        eccodes.codes_release(h)


if __name__ == "__main__":
    main()
