# coding: utf-8

from collections.abc import MutableMapping
import re
from os import path


METKIT_SHARE_DIR = path.expanduser("~/git/metkit/share/metkit")


def _load_yaml(filename: str):
    from yaml import safe_load

    with open(filename, "r") as y:
        return safe_load(y)


class Config(MutableMapping):

    _id_wind_uv = dict()
    _id_wind_vod = dict()
    _id_name = dict()
    _id_paramtable_re = re.compile(r"^(\d{1,3})\.(\d{1,3})$")

    def __init__(self, file: str, metkit_share_dir: str = ""):
        self._file = path.expanduser(file)
        assert path.exists(self._file)

        self._config = _load_yaml(self._file)
        assert self._config

        if metkit_share_dir:
            if not Config._id_wind_uv and not Config._id_wind_vod:
                y = _load_yaml(path.join(metkit_share_dir, "param-matching.yaml"))
                for u, v, vo, d in y["wind"]:
                    u, v, vo, d = map(Config._id, (u, v, vo, d))
                    Config._id_wind_vod[u, v] = (vo, d)
                    Config._id_wind_uv[vo, d] = (u, v)

            if not Config._id_name:
                # NOTE: keep only the first space-free name
                y = _load_yaml(path.join(metkit_share_dir, "paramids.yaml"))
                for id, lst in y.items():
                    for n in filter(lambda n: " " not in n, lst):
                        n = n.lower()
                        if n not in Config._id_name:
                            Config._id_name[n] = id

    def __delitem__(self, key: str):
        del self._config[self._id(key)]

    def __getitem__(self, key: str):
        return self._config[self._id(key)]

    def __setitem__(self, key: str, value):
        self._config[self._id(key)] = value

    def __iter__(self):
        return iter(self._config)

    def __len__(self):
        return len(self._config)

    def __str__(self):
        return str(self._config)

    def __repr__(self):
        return repr(self._config)

    @staticmethod
    def _id(key) -> int:
        if isinstance(key, float):
            # float(PARAM.TABLE) -> paramId, table might have missing trailing 0's
            p, t = str(key).split(".", maxsplit=1)
            return Config.id_paramtable(p, t.ljust(3, "0"))

        if isinstance(key, str):
            # shortName -> paramId
            kl = key.lower()
            if kl in Config._id_name:
                return Config._id_name[kl]

            # str(PARAM.TABLE) -> paramId
            ti = Config._id_paramtable_re.search(key)
            if ti:
                p, t = ti.group(1), ti.group(2)
                return Config.id_paramtable(p, t.ljust(3, "0"))

        return int(key)

    @staticmethod
    def id_paramtable(param, table):
        table = int(table)
        return (0 if table == 128 else table * 1000) + int(param)

    def config(self, node=[], name: str = "_config"):
        k = lambda x: int(x) if x.isdigit() else x
        b = self._config
        d = b.get(k(name), dict()).copy()
        for n in node:
            b = b[k(n)]
            d.update(b.get(name, dict()))
        return d


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Configuration tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("file", help="Configuration file")
    parser.add_argument("--node", help="Configuration node", nargs="*", type=str)
    parser.add_argument(
        "--metkit-share-dir",
        help="Metkit configuration directory (~metkit/share/metkit)",
        default=METKIT_SHARE_DIR,
    )

    args = parser.parse_args()
    print(args)

    cfg = Config(file=args.file, metkit_share_dir=args.metkit_share_dir)
    print(cfg.config(args.node))


if __name__ == "__main__":
    from sys import exit

    exit(main())
