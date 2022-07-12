# coding: utf-8

from collections.abc import MutableMapping
import re
from os import path


def _load_yaml(filename: str):
    from yaml import safe_load

    with open(filename, "r") as y:
        return safe_load(y)


class Config(MutableMapping):
    _config = dict()

    _default = dict(request=dict(levtype="pl"), step1_mod_6=False)

    _id_from_name = _load_yaml(path.join(path.dirname(__file__), "paramId.yaml"))
    _id_from_param_table = re.compile(r"^(\d{1,3})\.(\d{1,3})$")

    def __init__(self):
        if not Config._config:
            Config._config = _load_yaml(path.join(path.dirname(__file__), "ensms.yaml"))

    def __delitem__(self, key: str):
        del Config._config[self._keytransform(key)]

    def __getitem__(self, key: str):
        return Config._config.get(self._keytransform(key), Config._default)

    def __setitem__(self, key: str, value):
        Config._config[self._keytransform(key)] = value

    def __iter__(self):
        return iter(Config._config)

    def __len__(self):
        return len(Config._config)

    def __str__(self):
        return str(Config._config)

    def __repr__(self):
        return repr(Config._config)

    @staticmethod
    def _keytransform(key: str) -> int:
        if isinstance(key, str):
            kl = key.lower()
            if kl in Config._id_from_name:
                return Config._id_from_name[kl]

        ti = Config._id_from_param_table.search(key)
        if ti:
            table = int(ti.group(2))
            param = int(ti.group(1))
            return (0 if table == 128 else table * 1000) + param

        return int(key)
