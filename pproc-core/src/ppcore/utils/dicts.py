# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import collections
import itertools
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import TypeAlias
from typing import TypeVar

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
NestedDict: TypeAlias = Dict[K, V | "NestedDict"]


def dict_product(dic: collections.abc.Mapping[K, Iterable[V]]) -> Iterator[Dict[K, V]]:
    keys = list(dic.keys())
    its = tuple(dic.values())
    for vals in itertools.product(*its):
        yield dict(zip(keys, vals))


def dict_apply(func: Callable[[V], V], dic: NestedDict) -> NestedDict:
    modified_dic = dic.copy()
    for k, v in modified_dic.items():
        if isinstance(v, dict):
            modified_dic[k] = dict_apply(func, v)
        else:
            modified_dic[k] = func(v)
    return modified_dic


def deep_update(original: NestedDict, update: NestedDict) -> NestedDict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(original.get(key, None), dict):
            original[key] = deep_update(original[key], value)
        else:
            original[key] = value
    return original
