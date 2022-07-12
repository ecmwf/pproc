# coding: utf-8

from collections.abc import MutableMapping
import re
from os import path


def x(filename: str):
    from yaml import safe_load

    with open(path.join(path.dirname(__file__), filename), "r") as y:
        return safe_load(y)


class Hardcode(MutableMapping):
    _hard = dict()

    _default = dict(request=dict(levtype="pl"), step1_mod_6=False)

    # convert PARAM shortName to paramId
    _id_from_name = {
        "mx2t6": 121,
        "mn2t6": 122,
        "z": 129,
        "t": 130,
        "vo": 138,
        "msl": 151,
        "d": 155,
        "2t": 167,
        "10si": 207,
        "mx2t3": 228026,
        "mn2t3": 228027,
        "100si": 228249,
        "200si": 228241,
        "tmax": 3015,
        "tmin": 3016,
    }

    # convert PARAM id.table to paramId
    _re_table_id = re.compile(r"^(\d{1,3})\.(\d{1,3})$")

    def __init__(self):
        if not Hardcode._hard:
            Hardcode._hard = {
                id: dict(
                    request=dict(levtype="sfc", levelist=None),
                    step1_mod_6=True
                    if id in (121, 122)
                    else Hardcode._default["step1_mod_6"],
                )
                for id in (
                    121,
                    122,
                    151,
                    167,
                    207,
                    3015,
                    3016,
                    228026,
                    228027,
                    228241,
                    228249,
                )
            }

    def __delitem__(self, key: str):
        del Hardcode._hard[self._filter_param(key)]

    def __getitem__(self, key: str):
        return Hardcode._hard.get(self._filter_param(key), Hardcode._default)

    def __setitem__(self, key: str, value):
        Hardcode._hard[self._filter_param(key)] = value

    def __iter__(self):
        return iter(Hardcode._hard)

    def __len__(self):
        return len(Hardcode._hard)

    def __str__(self):
        return str(Hardcode._hard)

    def __repr__(self):
        return repr(Hardcode._hard)

    @staticmethod
    def _filter_param(key: str) -> int:
        if isinstance(key, str):
            kl = key.lower()
            if kl in Hardcode._id_from_name:
                return Hardcode._id_from_name[kl]

            ti = Hardcode._re_table_id.search(key)
            if ti:
                return int(ti.group(2)) * 1000 + int(ti.group(1))

        return int(key)
