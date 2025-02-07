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
