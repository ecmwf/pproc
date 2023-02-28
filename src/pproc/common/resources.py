
from dataclasses import dataclass
import resource
import time
from typing import Optional, Union


def plural(n: Union[int, float], name: str) -> str:
    return name + ("s" if abs(n) >= 2 else "")


@dataclass
class TimeDecomposition:
    days: int
    hours: int
    minutes: int
    seconds: int
    microseconds: float

    def __init__(
            self,
            days: float = 0.,
            hours: float = 0.,
            minutes: float = 0.,
            seconds: float = 0.,
            microseconds: float = 0.
            ):
        self.days = int(days)
        hours += (days - self.days) * 24
        self.hours = int(hours)
        minutes += (hours - self.hours) * 60
        self.minutes = int(minutes)
        seconds += (minutes - self.minutes) * 60
        self.seconds = int(seconds)
        microseconds += (seconds - self.seconds) * 1e6
        seconds, self.microseconds = divmod(microseconds, 1000000)
        self.seconds += int(seconds)
        minutes, self.seconds = divmod(self.seconds, 60)
        self.minutes += int(minutes)
        hours, self.minutes = divmod(self.minutes, 60)
        self.hours += int(hours)
        days, self.hours = divmod(self.hours, 24)
        self.days += int(days)

    def total_seconds(self) -> float:
        return (
            self.days * 86400
            + self.hours * 3600
            + self.minutes * 60
            + self.seconds
            + self.microseconds * 1e-6
        )

    def pretty(self) -> str:
        raw = f"{self.total_seconds():g} s"
        vals = [
            (num, f"{num:g} {plural(num, name)}")
            for num, name in zip(
                [self.days, self.hours, self.minutes, self.seconds, self.microseconds, 0],
                ['day', 'hour', 'minute', 'second', 'microsecond', '']
            )
        ]
        for (num, main), (snum, sub) in zip(vals[:-1], vals[1:]):
            if num > 0:
                return raw + (f" ({main} {sub})" if snum > 0 else f" ({main})")
        return raw


def pretty_time(seconds: float) -> str:
    return TimeDecomposition(seconds=seconds).pretty()


def pretty_bytes(bytes: int, decimal: bool = False) -> str:
    factor = 1000 if decimal else 1024
    unit = "B" if decimal else "iB"
    raw = f"{bytes} {plural(bytes, 'byte')}"
    if bytes < factor:
        return raw
    scaled = bytes
    prefixes = ["k", "M", "G", "T", "P", "E"]
    for prefix in prefixes:
        scaled /= factor
        if scaled < factor:
            return raw + f" ({scaled:g} {prefix}{unit})"


@dataclass
class ResourceUsage:
    cpu: float
    mem: int

    def __init__(self, cpu: Optional[float] = None, mem: Optional[int] = None):
        res = None
        if cpu is None or mem is None:
            res = resource.getrusage(resource.RUSAGE_SELF)
        if cpu is None:
            cpu = res.ru_utime
        if mem is None:
            mem = res.ru_maxrss * 1024
        self.cpu = cpu
        self.mem = mem

    def __str__(self):
        return (
            f"CPU time: {pretty_time(self.cpu)}, " +
            f"memory: {pretty_bytes(self.mem)}"
        )


@dataclass
class ResourceMeter:
    start: float
    elapsed: float
    res: ResourceUsage

    def __init__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.
        self.res = ResourceUsage()

    def update(self, reset: bool = False):
        t = time.perf_counter()
        self.res = ResourceUsage()
        if reset:
            self.start = t
        self.elapsed = t - self.start
        return self

    def __str__(self):
        return f"wall time: {pretty_time(self.elapsed)}, " + str(self.res)