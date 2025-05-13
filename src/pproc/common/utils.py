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
from typing import Callable, Dict, Iterable, Iterator, TypeVar

K = TypeVar("K")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


def dict_product(dic: Dict[K, Iterable[V]]) -> Iterator[Dict[K, V]]:
    keys = list(dic.keys())
    its = tuple(dic.values())
    for vals in itertools.product(*its):
        yield dict(zip(keys, vals))


def delayed_map(delay: int, func: Callable[[T], U], it: Iterable[T]) -> Iterator[U]:
    sentinel = object()
    it = iter(it)
    queue = collections.deque(func(x) for x in itertools.islice(it, max(delay, 1)))
    while queue:
        yield queue.popleft()
        x = next(it, sentinel)
        if x is not sentinel:
            queue.append(func(x))
