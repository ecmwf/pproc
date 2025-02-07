import dataclasses

import pytest

from pproc.common.steps import Step, parse_step, step_to_coord


def test_simple_step():
    s = Step(6)

    assert s.start == 6
    assert s.end is None

    assert str(s) == "6"
    assert int(s) == 6

    assert not s.is_range()

    assert s.next() == Step(7)

    assert s.decay() == 6

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.start = 7
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.end = 8

    assert s < Step(12)
    assert not s < Step(6)
    assert not s < Step(4)
    assert s <= Step(9)
    assert s <= Step(6)
    assert not s <= Step(3)
    assert s > Step(2)
    assert not s > Step(6)
    assert not s > Step(18)
    assert s >= Step(0)
    assert s >= Step(6)
    assert not s >= Step(15)
    assert s == Step(6)
    assert not s == Step(8)
    assert s != Step(5)
    assert not s != Step(6)

    t = Step(s)
    assert t == s


def test_simple_step_range():
    r = Step(12, 24)

    assert r.start == 12
    assert r.end == 24

    assert str(r) == "12-24"
    with pytest.raises(ValueError):
        int(r)

    assert r.is_range()

    assert r.next() == Step(12, 25)

    assert r.decay() == r

    with pytest.raises(dataclasses.FrozenInstanceError):
        r.start = 6
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.end = 18

    assert r < Step(12, 36)
    assert r < Step(18, 24)
    assert not r < Step(12, 24)
    assert not r < Step(6, 24)
    assert not r < Step(3, 9)
    assert r <= Step(12, 48)
    assert r <= Step(18, 36)
    assert r <= Step(12, 24)
    assert not r <= Step(12, 16)
    assert not r <= Step(8, 12)
    assert r > Step(12, 18)
    assert r > Step(9, 15)
    assert r > Step(4, 8)
    assert not r > Step(12, 24)
    assert not r > Step(12, 36)
    assert not r > Step(18, 24)
    assert r >= Step(12, 15)
    assert r >= Step(6, 9)
    assert r >= Step(6, 18)
    assert r >= Step(12, 24)
    assert not r >= Step(12, 36)
    assert not r >= Step(18, 36)
    assert r == Step(12, 24)
    assert not r == Step(12, 36)
    assert not r == Step(24, 48)
    assert r != Step(12, 15)
    assert r != Step(16, 18)
    assert not r != Step(12, 24)

    s = Step(r)
    assert r == s


@pytest.mark.parametrize(
    "inp, exp",
    [
        ("168", 168),
        (240, 240),
        (Step(360), Step(360)),
        ("72-96", Step(72, 96)),
        (Step(120, 168), Step(120, 168)),
    ],
)
def test_parse_step(inp, exp):
    assert parse_step(inp) == exp


def test_step_to_coord():
    assert step_to_coord(18) == 18
    assert step_to_coord(Step(12)) == 12
    assert step_to_coord(Step(0, 24)) == "0-24"
