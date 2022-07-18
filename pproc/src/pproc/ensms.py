#!/usr/bin/env python3

import numpy as np
from datetime import datetime

import eccodes


class Grib:
    def __init__(self, fn: str):
        self._file = open(fn)
        self.ids = list()

        while True:
            id = eccodes.codes_grib_new_from_file(self._file)
            if id:
                self.ids.append(id)
            else:
                break

    def __del__(self):
        for id in self.ids:
            eccodes.codes_release(id)
        self._file.close()


def parse_range(r: str):
    if not r:
        return None

    l = r.lower().split("/")
    if len(l) == 5 and l[1] == "to" and l[3] == "by":
        return range(int(l[0]), int(l[2]) + 1, int(l[4]))

    if len(l) == 3 and l[1] == "to":
        return range(int(l[0]), int(l[2]) + 1, 1)

    return sorted(set(map(int, l)))


def mars(input: str = ""):
    from subprocess import run
    from os import environ

    p = run(
        "/Users/mapm/.local/bin/mars",
        capture_output=True,
        text=True,
        input=input,
        env=environ,
    )
    if p.returncode:
        print(f"stdout:\n{p.stdout}\n")
        print(f"stderr:\n{p.stderr}\n")

    return p.returncode == 0


def calculate_mean(ids):
    """
    Ensemble mean:
    Calculates the mean across forecast ensemble members (perturbed and control).
    """
    data = np.asarray([eccodes.codes_get_values(id) for id in ids])
    return np.mean(data, axis=0)  # nanmean?


def calculate_stddev(ids):
    """
    Ensemble stddev:
    Calculates the standard deviation across forecast ensemble members (perturbed and control).
    """
    data = np.asarray([eccodes.codes_get_values(id) for id in ids])
    return np.std(data, axis=0)  # nanstd?


def write_fdb(fdb, template_grib, values, type, nens) -> None:
    # Copy template GRIB message and modify headers
    assert False
    out_grib = template_grib.copy()
    out_grib.set("type", type)
    out_grib.set("perturbationNumber", 0)
    out_grib.set("numberOfForecastsInEnsemble", nens)
    out_grib.set_array("values", values)

    fdb.archive(out_grib.get_buffer())


def main(args=None):
    import argparse
    from os import path

    # import pyfdb
    from pproc.Config import VariableTree

    # arguments
    parser = argparse.ArgumentParser(
        description="Calculate ensemble means and standard deviations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config-file", help="Configuration file", required=True)
    parser.add_argument("--config-node", help="Configuration node", nargs="+")

    parser.add_argument("--no-mars", action="store_false", dest="mars")
    parser.add_argument("--write", action="store_true")

    parser.add_argument("--class", default="od", dest="klass")
    parser.add_argument("--stream", default="enfo")
    parser.add_argument("--expver", default=1)
    parser.add_argument("--date", required=True)
    parser.add_argument("--time", choices=(0, 6, 12, 18), default=0)
    parser.add_argument("--grid", default="O640")
    parser.add_argument("--number", default="1/to/50")
    args = parser.parse_args()

    date = datetime.strptime(args.date, "%Y%m%d")

    # variables
    tree = VariableTree(args.config_file)
    var = tree.variables(*map(lambda n: int(n) if n.isdigit() else n, args.config_node))
    var.update(vars(args))
    var["class"] = var.pop("klass")
    print(var)

    target = (
        "param={param},"
        + ("levtype=sfc", "levelist={levelist}")[bool(var["levelist"])]
        + ",step={step}"
    )

    pf = {
        "_verb": "retrieve",
        "class": var["class"],
        "type": "pf",
        "stream": var["stream"],
        "param": var["param"],
        "date": var["date"],
        "time": var["time"],
        "levtype": var["levtype"],
        "levelist": var["levelist"],
        "resol": "av",
        "grid": var["grid"],
        "expver": var["expver"],
        "step": var["step"],
        "number": var["number"],
        "target": f"'{target}'",
    }

    cf = {"_verb": "retrieve", "type": "cf", "number": "off"}

    r = lambda d: ",".join(f"{k}={v}" for k, v in d.items() if v)[6:]
    request = f"""
    {r(pf)}
    {r(cf)}
    """
    print(request)

    fdb = None  # pyfdb.FDB()

    if var["mars"]:
        assert mars(request)
        assert path.exists(target)

    for level in parse_range(var["levelist"]):
        for step in parse_range(var["step"]):
            param = var["param"]
            levtype = var["levtype"]

            fn = f"""param={param},levtype={levtype}{("",f",levelist={level}")[bool(level)]},step={step}"""
            assert path.exists(fn)

            grib = Grib(fn)

            nens = len(grib.ids)
            print(f"nens={nens}")
            assert nens == 51

            mean = calculate_mean(grib.ids)
            if var["write"]:
                write_fdb(fdb, grib.ids[0], mean, "em", nens)

            stddev = calculate_stddev(grib.ids)
            if var["write"]:
                write_fdb(fdb, grib.ids[0], stddev, "es", nens)

    # fdb.flush()


if __name__ == "__main__":
    import sys

    main(sys.argv)
