from dataclasses import dataclass
from typing import Union


@dataclass(init=False, order=True, frozen=True)
class Step:
    start: int
    end: int = None

    def __init__(self, start_or_step, end=None):
        _set_frozen = lambda attr, val: object.__setattr__(self, attr, val)
        if isinstance(start_or_step, Step):
            _set_frozen("start", start_or_step.start)
            _set_frozen("end", start_or_step.end)
        else:
            _set_frozen("start", int(start_or_step))
            _set_frozen("end", None if end is None else int(end))

    def __str__(self):
        if self.end is None:
            return f"{self.start}"
        return f"{self.start}-{self.end}"

    def __int__(self):
        if self.end is not None:
            raise ValueError("Cannot convert a step range to int")
        return self.start

    def is_range(self):
        return self.end is not None

    def next(self):
        if self.end is None:
            return Step(self.start + 1)
        return Step(self.start, self.end + 1)

    def decay(self):
        if self.end is None:
            return self.start
        return self


AnyStep = Union[int, Step]


def parse_step(s) -> AnyStep:
    if isinstance(s, (int, Step)):
        return s
    if not isinstance(s, str):
        raise TypeError(f"Cannot convert {type(s)} to a step or step range")
    start, sep, end = s.partition("-")
    if not sep:
        return int(start)
    return Step(int(start), int(end))


def step_to_coord(s: AnyStep) -> Union[int, str]:
    if isinstance(s, int):
        return s
    assert isinstance(s, Step)
    if s.is_range():
        return str(s)
    return int(s)
