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
    _config = dict()
    _id_wind_uv = dict()
    _id_wind_vod = dict()
    _default = dict(request=dict(levtype="pl"), step1_mod_6=False)

    _id_from_name = dict()
    _id_from_param_table_re = re.compile(r"^(\d{1,3})\.(\d{1,3})$")

    def __init__(self):
        if not Config._config:
            Config._config = _load_yaml(path.join(path.dirname(__file__), "ensms.yaml"))

        if not Config._id_wind_vod:
            y = _load_yaml(path.join(METKIT_SHARE_DIR, "param-matching.yaml"))
            for u, v, vo, d in y["wind"]:
                u, v, vo, d = map(Config._id, (u, v, vo, d))
                Config._id_wind_vod[u, v] = (vo, d)
                Config._id_wind_uv[vo, d] = (u, v)

        if not Config._id_from_name:
            # NOTE: keep only the first space-free name
            y = _load_yaml(path.join(METKIT_SHARE_DIR, "paramids.yaml"))
            for id, lst in y.items():
                for name in filter(lambda name: " " not in name, lst):
                    if name not in Config._id_from_name:
                        Config._id_from_name[name] = id

    def __delitem__(self, key: str):
        del Config._config[self._id(key)]

    def __getitem__(self, key: str):
        return Config._config.get(self._id(key), Config._default)

    def __setitem__(self, key: str, value):
        Config._config[self._id(key)] = value

    def __iter__(self):
        return iter(Config._config)

    def __len__(self):
        return len(Config._config)

    def __str__(self):
        return str(Config._config)

    def __repr__(self):
        return repr(Config._config)

    @staticmethod
    def _id(key: str) -> int:
        def id_from_param_table(param, table):
            table = int(table)
            return (0 if table == 128 else table * 1000) + int(param)

        if isinstance(key, float):
            # float(PARAM.TABLE) -> paramId, table might have missing trailing 0's
            p, t = str(key).split(".", maxsplit=1)
            return id_from_param_table(p, t.ljust(3, "0"))

        if isinstance(key, str):
            # shortName -> paramId
            kl = key.lower()
            if kl in Config._id_from_name:
                return Config._id_from_name[kl]

            # str(PARAM.TABLE) -> paramId
            ti = Config._id_from_param_table_re.search(key)
            if ti:
                return id_from_param_table(ti.group(2), ti.group(1).ljust(3, "0"))

        return int(key)
