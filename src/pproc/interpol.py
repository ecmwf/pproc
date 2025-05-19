#!/usr/bin/env python3
# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

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
import re
import yaml
from os import path
import mir


class _Regex(object):
    def __init__(self, pattern):
        self._pattern = re.compile(pattern)

    def __call__(self, value):
        if not self._pattern.fullmatch(value):
            raise argparse.ArgumentTypeError(
                "must match '{}'".format(self._pattern.pattern)
            )
        return value


def main(args=None):
    g = r"[ONF][1-9][0-9]*"
    f = r"([0-9]*[.])?[0-9]+"

    _grid = f + r"/" + f + r"|" + g
    _area = r"-?" + f + r"/-?" + f + r"/-?" + f + r"/-?" + f
    _accuracy = r"\d+"
    _edition = r"1|2"
    _interpolation = r"linear|nn|grid-box-average|grid-box-statistics|fail"
    _packing = r"ccsds|complex|ieee|second-order|simple"
    _statistics = r"maximum|minimum|count"
    _intgrid = g + r"|none|source"
    _truncation = r"[1-9][0-9]*|none"

    grids = path.join(mir.home(), "etc", "mir", "grids.yaml")
    if path.exists(grids):
        with open(grids) as file:
            for key in yaml.safe_load(file).keys():
                _grid = _grid + r"|" + key

    arg = argparse.ArgumentParser()

    arg.add_argument(
        "--area",
        type=_Regex(_area),
        help="sub-area to be extracted (" + _area + ")",
    )

    arg.add_argument(
        "--grid",
        type=_Regex(_grid),
        help="Regular latitude/longitude grids (<west-east>/<south-north> increments) or Gaussian octahedral/quasi-regular/regular grids ("
        + _grid
        + ")",
    )

    arg.add_argument(
        "--interpolation",
        type=_Regex(_interpolation),
        help="interpolation method (" + _interpolation + ")",
    )

    arg.add_argument(
        "--packing",
        type=_Regex(_packing),
        help="packing method (GRIB packingType, " + _packing + ")",
    )

    arg.add_argument(
        "--accuracy",
        type=_Regex(_accuracy),
        help="accuracy (GRIB bitsPerValue, " + _accuracy + ")",
    )

    arg.add_argument(
        "--edition",
        type=_Regex(_edition),
        help="edition (GRIB edition, " + _edition + ")",
    )

    arg.add_argument(
        "--interpolation-statistics",
        type=_Regex(_statistics),
        help="interpolation statistics method (" + _statistics + ")",
    )

    arg.add_argument(
        "--intgrid",
        type=_Regex(_intgrid),
        help="spectral transforms intermediate Gaussian grid (" + _intgrid + ")",
    )

    arg.add_argument(
        "--truncation",
        type=_Regex(_truncation),
        help="spectral transforms intermediate truncation (" + _truncation + ")",
    )

    g = arg.add_mutually_exclusive_group()

    g.add_argument(
        "--vod2uv",
        help="Input is vorticity and divergence (vo/d), convert to vector Cartesian components (gridded u/v or spectral U/V)",
        action="store_true",
    )

    g.add_argument(
        "--uv2uv",
        help="Input is vector Cartesian components spectral U/V or gridded u/v",
        action="store_true",
    )

    # arg.add_argument("-v", "--verbose", action="count", default=0, help="Verbosity level")

    arg.add_argument("grib_in", type=str, help="Input GRIB file")
    arg.add_argument("grib_out", type=str, help="Output GRIB file")

    args = arg.parse_args(args)
    print(args)

    options = {}
    for k in [
        "area",
        "grid",
        "interpolation",
        "interpolation_statistics",
        "intgrid",
        "packing",
        "accuracy",
        "edition",
        "truncation",
    ]:
        if hasattr(args, k):
            v = getattr(args, k)
            if v is not None:
                options[k.replace("_", "-")] = v

    job = mir.Job(**options)
    print("Running", job)

    grib_out = mir.GribFileOutput(args.grib_out)
    with open(args.grib_in, "rb") as grib_in:
        job.execute(grib_in, grib_out)


if __name__ == "__main__":
    main()
