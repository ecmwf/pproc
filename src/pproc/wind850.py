#!/usr/bin/env python3

from os import path

import eccodes
# import pyfdb


def parse_range(r: str):
    if not r:
        return (None,)

    if isinstance(r, list):
        return sorted(set(r))

    l = r.lower().split("/")
    if len(l) == 5 and l[1] == "to" and l[3] == "by":
        return range(int(l[0]), int(l[2]) + 1, int(l[4]))

    if len(l) == 3 and l[1] == "to":
        return range(int(l[0]), int(l[2]) + 1, 1)

    return sorted(set(map(int, l)))


class Wind:
    def __init__(self, vo, d, u, v) -> None:
        self.vo = vo
        self.d = d
        self.u = u
        self.v = v


def parse_wind_paramids(param: str, metkit_share_dir) -> Wind:
    from pproc.Config import ParamId

    uv = param.split("/")
    assert len(uv) == 2

    paramid = ParamId(metkit_share_dir)
    u, v = map(lambda p: paramid.id(int(p) if p.isdigit() else p), uv)
    vo, d = paramid.vod((u, v))

    return Wind(vo, d, u, v)


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


def wind_mean_sd_eps(wind, levelist, step, expver, var: dict, pp: dict) -> list:
    """
    Calculate ensemble (type=cf/pf) mean and standard deviation of wind speed
    """

    assert var["mars"]

    target = f"param=vo_d,levelist={levelist},step={step}.eps.grib"
    request = f"""
retrieve,
    date     = {var["date"]},
    time     = {var["time"]},
    number   = {var["number"]},
    stream   = enfo,
    levelist = {levelist},
    levtype  = pl,
    expver   = {expver},
    type     = pf,
    step     = {step},
    class    = {var["class"]},
    param    = {"/".join((str(wind.vo), str(wind.d)))},
    target   = '{target}'
retrieve,
    type   = cf,
    number = off
"""
    from io import BytesIO
    import numpy as np
    import mir

    print(request)
    # assert mars(request)
    assert path.exists(target)

    out = BytesIO()
    inp = mir.MultiDimensionalGribFileInput(target, 2)

    job = mir.Job(vod2uv="1", **pp)
    job.execute(inp, out)

    out.seek(0)
    reader = eccodes.StreamReader(out)
    messages = list(reader)

    tens = (len(parse_range(var["number"])) + 1) * 2  # (nens + 1)*(vo, d)
    assert len(messages) == tens

    u = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind.u])
    v = np.asarray([m.get_array("values") for m in messages if m["paramId"] == wind.v])
    assert u.shape == v.shape
    assert u.shape[0] + v.shape[0] == tens

    ws = np.sqrt(u * u + v * v)
    mean = np.mean(ws, axis=0)
    stddev = np.std(ws, axis=0, ddof=int(var["std_corrected_sample"]))

    em = messages[0].copy()
    em.set("marsType", "em")
    em.set("indicatorOfParameter", "010")  # FIXME check?
    em.set("gribTablesVersionNo", 128)
    em.set_array("values", mean)

    es = messages[0].copy()
    es.set("marsType", "es")
    es.set("indicatorOfParameter", "010")  # FIXME check?
    es.set("gribTablesVersionNo", 128)
    es.set_array("values", stddev)

    return em, es


def wind_norm_det(wind, levelist, step, expver, var: dict, pp: dict) -> eccodes.Message:
    """
    Calculate deterministic (type=fc) wind speed
    """
    from io import BytesIO
    import numpy as np
    import mir

    assert var["mars"]

    target = f"param=vo_d,levelist={levelist},step={step}.det.grib"
    request = f"""
retrieve,
    date     = {var["date"]},
    time     = {var["time"]},
    levelist = {levelist},
    levtype  = pl,
    expver   = {expver},
    type     = fc,
    step     = {step},
    class    = {var["class"]},
    param    = {"/".join((str(wind.vo), str(wind.d)))},
    target   = '{target}'
"""

    print(request)
    # assert mars(request)
    assert path.exists(target)

    out = BytesIO()
    inp = mir.MultiDimensionalGribFileInput(target, 2)

    job = mir.Job(vod2uv="1", **pp)
    job.execute(inp, out)

    out.seek(0)
    reader = eccodes.StreamReader(out)
    messages = list(reader)

    assert len(messages) == 2
    assert messages[0]["paramId"] == wind.u
    assert messages[1]["paramId"] == wind.v

    uv = np.asarray([m.get_array("values") for m in messages])
    ws = np.linalg.norm(uv, axis=0)

    ws = messages[0].copy()
    ws.set_array("values", ws)

    return ws


def calculate_normalized_stddev_last_30_days():
    assert False


def main(args=None):
    from datetime import datetime
    import argparse

    from pproc.Config import VariableTree, postproc_keys

    # arguments
    parser = argparse.ArgumentParser(
        description="Calculate wind speed mean/standard deviation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config-file", help="Configuration file", required=True)
    parser.add_argument(
        "--config-node", help="Configuration node", required=True, nargs="+"
    )

    parser.add_argument("--no-mars", action="store_false", dest="mars")
    parser.add_argument("--write-grib", action="store_true")
    parser.add_argument("--write-fdb", action="store_true")
    parser.add_argument(
        "--std-corrected-sample",
        help="corrected sample standard deviation",
        action="store_true",
    )
    parser.add_argument(
        "--metkit-share-dir",
        help="Metkit configuration directory",
        default="/usr/local/apps/ecmwf-toolbox/2022.05.0.0/GNU/11.2/share/metkit",
    )

    parser.add_argument("--param", default="u/v")
    parser.add_argument("--class", default="od", dest="klass")
    parser.add_argument("--expver", default=1)
    parser.add_argument("--expver-da", default=None)
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

    # assert fdb or mars-client
    fdb = None  # pyfdb.FDB()
    assert bool(fdb) != args.mars

    # variables
    tree = VariableTree(args.config_file)
    var = tree.variables(*map(lambda n: int(n) if n.isdigit() else n, args.config_node))
    var.update(vars(args))
    var["class"] = var.pop("klass")
    print(var)

    # wind paramids
    wind = parse_wind_paramids(var["param"], args.metkit_share_dir)

    # post-processing keys
    pp_keys = postproc_keys(args.metkit_share_dir)
    pp = {key: var[key] for key in pp_keys if var.get(key, None)}
    
    for levelist in parse_range(var["levelist"]):
        for step in parse_range(var["step"]):

            # calculate mean/stddev of wind speed for type=pf/cf (eps)
            em, es = wind_mean_sd_eps(wind, levelist, step, var["expver"], var, pp)

            # calculate wind speed for type=fc (deterministic)
            expver = var.get("expver_da", None)
            if not expver:
                expver = var["expver"]

            det = wind_norm_det(wind, levelist, step, expver, var, pp)

            for message, filename in zip((em, es, det), ("em.grib", "es.grib", "det.grib")):
                if var["write_grib"]:
                    assert not path.exists(filename)
                    with open(filename, "wb") as fout:
                        message.write_to(fout)

                if var["write_fdb"]:
                    fdb.archive(message.get_buffer())

#     # fdb.flush()


if __name__ == "__main__":
    import sys

    main(sys.argv)
