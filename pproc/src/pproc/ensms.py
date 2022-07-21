#!/usr/bin/env python3

from io import BytesIO

from datetime import datetime
import numpy as np

import eccodes
import mir
# import pyfdb


def parse_range(r: str):
    if not r:
        return (None,)

    l = r.lower().split("/")
    if len(l) == 5 and l[1] == "to" and l[3] == "by":
        return range(int(l[0]), int(l[2]) + 1, int(l[4]))

    if len(l) == 3 and l[1] == "to":
        return range(int(l[0]), int(l[2]) + 1, 1)

    return sorted(set(map(int, l)))


def mars(input: str = ""):
    from os import environ
    from subprocess import run

    p = run(
        "mars",
        capture_output=True,
        text=True,
        input=input,
        env=environ,
    )
    if p.returncode:
        print(f"stdout:\n{p.stdout}\n")
        print(f"stderr:\n{p.stderr}\n")

    return p.returncode == 0


def main(args=None):
    import argparse
    from os import path

    from pproc.Config import VariableTree, postproc_keys

    # arguments
    parser = argparse.ArgumentParser(
        description="Calculate ensemble means and standard deviations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config-file", help="Configuration file", required=True)
    parser.add_argument("--config-node", help="Configuration node", required=True, nargs="+")

    parser.add_argument("--no-mars", action="store_false", dest="mars")
    parser.add_argument("--write-grib", action="store_true")
    parser.add_argument("--write-fdb", action="store_true")
    parser.add_argument("--std-corrected-sample", help="corrected sample standard deviation", action="store_true")
    parser.add_argument("--metkit-share-dir", help="Metkit configuration directory", default="/usr/local/apps/ecmwf-toolbox/2022.05.0.0/GNU/11.2/share/metkit")

    parser.add_argument("--class", default="od", dest="klass")
    parser.add_argument("--stream", default="enfo")
    parser.add_argument("--expver", default=1)
    parser.add_argument("--date", required=True)
    parser.add_argument("--time", choices=(0, 6, 12, 18), default=0, type=int)
    parser.add_argument("--number", default="1/to/50")

    parser.add_argument("--grid", default="O640")
    parser.add_argument("--resol", default="av")
    parser.add_argument("--intgrid", default=None)
    parser.add_argument("--truncation", default=None)

    args = parser.parse_args()
    print(args)

    date = datetime.strptime(args.date, "%Y%m%d")
    pp_keys = postproc_keys(args.metkit_share_dir)

    # variables
    tree = VariableTree(args.config_file)
    var = tree.variables(*map(lambda n: int(n) if n.isdigit() else n, args.config_node))
    var.update(vars(args))
    var["class"] = var.pop("klass")
    print(var)

    # assert one parameter only
    assert "/" not in str(var["param"])

    # assert fdb or mars-client
    fdb = None  # pyfdb.FDB()
    assert bool(fdb) != bool(var["mars"])

    tens = len(parse_range(var["number"])) + 1  # nens + 1

    for levelist in parse_range(var["levelist"]):
        for step in parse_range(var["step"]):

            if var["mars"]:
                target = f"""param={var["param"]},levtype={var["levtype"]}{("",f",levelist={levelist}")[bool(levelist)]},step={step}"""
                request = f"""
retrieve,
    class    = {var["class"]},
    type     = pf,
    stream   = {var["stream"]},
    param    = {var["param"]},
    date     = {var["date"]},
    time     = {var["time"]},
    levtype  = {var["levtype"]},
    levelist = {levelist},
    expver   = {var["expver"]},
    step     = {step},
    number   = {var["number"]},
    target   = '{target}'

retrieve,
    type   = cf,
    number = off
"""
# grid=[{var["grid"]}, intgrid=source, truncation=none,

                print(request)
                # assert mars(request)
                assert path.exists(target)

                # post-process
                pp = {key: var[key] for key in pp_keys if var.get(key, None)}
                if pp:
                    buf = bytearray(64 * 1024 * 1024)
                    out = mir.GribMemoryOutput(buf)
                    # out = BytesIO() 
                    # out = mir.GribFileOutput("x.grib")

                    inp = mir.GribFileInput(target)

                    job = mir.Job(**pp)
                    job.execute(inp, out)

                    eccodes_reader = eccodes.MemoryReader(out)
                    # eccodes_reader = eccodes.MemoryReader(out.getvalue())
                    # eccodes_reader = eccodes.FileReader("x.grib")
                else:
                    eccodes_reader = eccodes.FileReader(target)

                messages = list(eccodes_reader)

            # check data
            # - for expected number of messages
            # - for absence of missing values (otherwise need to convert to/from nan and use np.nanmean/np.nanstd)
            assert tens == len(messages)
            assert not any(m["missingValuesPresent"] for m in messages)

            data = np.asarray([m.get_array("values") for m in messages])

            # mean across forecast ensemble members (perturbed and control)
            # copy template GRIB message, modify headers and write/archive
            mean = np.mean(data, axis=0)

            out_grib = messages[0].copy()
            out_grib.set("type", "em")
            out_grib.set("perturbationNumber", 0)
            out_grib.set("numberOfForecastsInEnsemble", tens)
            out_grib.set_array("values", mean)

            if var["write_grib"]:
                assert not path.exists("em.grib")
                with open("em.grib", "wb") as fout:
                    out_grib.write_to(fout)

            if var["write_fdb"]:
                fdb.archive(out_grib.get_buffer())

            # standard deviation across forecast ensemble members (perturbed and control)
            # copy template GRIB message, modify headers and write/archive
            stddev = np.std(data, axis=0, ddof=int(var["std_corrected_sample"]))

            out_grib = messages[0].copy()
            out_grib.set("type", "es")
            out_grib.set("perturbationNumber", 0)
            out_grib.set("numberOfForecastsInEnsemble", tens)
            out_grib.set_array("values", stddev)

            if var["write_grib"]:
                assert not path.exists("es.grib")
                with open("es.grib", "wb") as fout:
                    out_grib.write_to(fout)

            if var["write_fdb"]:
                fdb.archive(out_grib.get_buffer())

    # fdb.flush()


if __name__ == "__main__":
    import sys

    main(sys.argv)
