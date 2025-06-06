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
import sys
import numpy as np

from eccodes import GribFile, GribMessage


def spectrum(T, sh):
    rds = 1.0
    ra = rds * 1000.0
    zlam = 0.5  # only need 0.5 due to eq in Lambert, is eq(3)*2 in IFS

    def norm(m, n, r, i):
        zmet = zlam * (1 if m == 0 else 2)
        zfact = 1.0 if n == 0 else ra ** 2 / (n * (n + 1))
        return zmet * zfact * (r ** 2 if m == 0 else (r ** 2 + i ** 2))

    sp = np.zeros(T + 1)
    i = 0
    for m in range(T + 1):
        for n in range(m, T + 1):
            sp[n] += norm(m, n, sh[i], sh[i + 1])
            i += 2

    return sp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files", metavar="file", type=str, nargs="+", help="input GRIB file(s)"
    )
    parser.add_argument(
        "--vod", action="store_true", help="extract kinetic energy from vo/d"
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="spectrum in linear scale (instead of log10)",
    )

    args = parser.parse_args()
    for f in args.files:
        with GribFile(f) as gribFile:
            if args.vod:
                for i in range(0, len(gribFile), 2):
                    msg1 = GribMessage(gribFile)
                    msg2 = GribMessage(gribFile)

                    T = msg1["pentagonalResolutionParameterJ"]
                    assert T == msg2["pentagonalResolutionParameterJ"]

                    sh_vo = msg1["values"]
                    sh_d = msg2["values"]
                    assert (T + 1) * (T + 2) == len(sh_vo)
                    assert (T + 1) * (T + 2) == len(sh_d)

                    rot = spectrum(T, sh_vo)
                    div = spectrum(T, sh_d)
                    ke = (rot + div) * np.array([n ** 1.6666 for n in range(T + 1)])
                    if not args.linear:
                        rot = np.log10(rot)
                        div = np.log10(div)
                        ke = np.log10(ke)

                    print(
                        "#{}: paramId={}/{}, T={}".format(
                            i + 1, msg1["paramId"], msg2["paramId"], T
                        )
                    )
                    print("T, rotKE, divKE, KE")
                    for n in range(1, T + 1):
                        print(
                            "{}, {}, {}, {}".format(np.log10(n), rot[n], div[n], ke[n])
                        )

            else:
                for i in range(len(gribFile)):
                    msg = GribMessage(gribFile)
                    T = msg["pentagonalResolutionParameterJ"]

                    sh = msg["values"]
                    assert (T + 1) * (T + 2) == len(sh)

                    scalar = spectrum(T, sh)
                    if not args.linear:
                        scalar = np.log10(scalar)

                    print("#{}: paramId={}, T={}".format(i + 1, msg["paramId"], T))
                    print("T, lev")
                    for n in range(1, T + 1):
                        print("{}, {}".format(np.log10(n), scalar[n]))


if __name__ == "__main__":
    sys.exit(main())
