from typing import Iterator

import numpy as np
import pytest

from pproc.common.accumulation import Aggregation, Mean, SimpleAccumulation
from pproc.common.accumulation_manager import AccumulationManager
from pproc.common.utils import dict_product


CONFIGS = {
    "single-step": {"step": {"coords": [[6]]}},
    "step-range": {
        "step": {"operation": "mean", "coords": [{"from": 24, "to": 48, "by": 6}]}
    },
    "multi-step": {"step": {"coords": [[6], [12], [18], [24]]}},
    "multi-step-range": {
        "step": {
            "operation": "mean",
            "coords": [
                {"from": i + 6, "to": i + 24, "by": 6} for i in range(0, 25, 12)
            ],
        }
    },
    "multi-step-range-sameend": {
        "step": {
            "operation": "sum",
            "coords": [
                {"from": 12, "to": 48, "by": 12},
                {"from": 24, "to": 48, "by": 6},
            ],
        }
    },
    "onechunk-multidim": {
        "levelist": {"coords": [[1000]]},
        "hdate": {"coords": [[20230204]]},
        "step": {"coords": [[24]]},
    },
    "single-multidim": {
        "levelist": {"coords": [[250], [500], [1000]]},
        "step": {"coords": [[12], [24]]},
    },
    "multidim-multistep": {
        "hdate": {"coords": [[20200118, 20210118, 20220118, 20230118]]},
        "step": {
            "operation": "mean",
            "coords": [
                {"from": i + 6, "to": i + 24, "by": 6} for i in range(0, 25, 12)
            ],
        },
    },
}

EXPECT_ACCUMS = {
    "single-step": {"step": (Aggregation, [[6]])},
    "step-range": {"step": (Mean, [range(24, 49, 6)])},
    "multi-step": {"step": (Aggregation, [[6], [12], [18], [24]])},
    "multi-step-range": {
        "step": (Mean, [range(i + 6, i + 25, 6) for i in range(0, 25, 12)])
    },
    "multi-step-range-sameend": {
        "step": (SimpleAccumulation, [range(12, 49, 12), range(24, 49, 6)])
    },
    "onechunk-multidim": {
        "levelist": (Aggregation, [[1000]]),
        "hdate": (Aggregation, [[20230204]]),
        "step": (Aggregation, [[24]]),
    },
    "single-multidim": {
        "levelist": (Aggregation, [[250], [500], [1000]]),
        "step": (Aggregation, [[12], [24]]),
    },
    "multidim-multistep": {
        "hdate": (Aggregation, [[20200118, 20210118, 20220118, 20230118]]),
        "step": (Mean, [range(i + 6, i + 25, 6) for i in range(0, 25, 12)]),
    },
}

EXPECT_COORDS = {
    "single-step": {"step": {6}},
    "step-range": {"step": {24, 30, 36, 42, 48}},
    "multi-step": {"step": {6, 12, 18, 24}},
    "multi-step-range": {"step": {6, 12, 18, 24, 30, 36, 42, 48}},
    "multi-step-range-sameend": {"step": {12, 24, 30, 36, 42, 48}},
    "onechunk-multidim": {"levelist": {1000}, "hdate": {20230204}, "step": {24}},
    "single-multidim": {"levelist": {250, 500, 1000}, "step": {12, 24}},
    "multidim-multistep": {
        "hdate": {20200118, 20210118, 20220118, 20230118},
        "step": {6, 12, 18, 24, 30, 36, 42, 48},
    },
}


@pytest.mark.parametrize(
    "config, accums, coords",
    [
        pytest.param(config, EXPECT_ACCUMS[name], EXPECT_COORDS[name], id=name)
        for name, config in CONFIGS.items()
    ],
)
def test_create(config, accums, coords):
    mgr = AccumulationManager(config)
    assert mgr.coords == coords
    acc_coords_all = list(dict_product({key: coo for key, (_, coo) in accums.items()}))
    for acc in mgr.accumulations.values():
        matched = True
        for i, acc_coords in enumerate(acc_coords_all):
            if set(d.key for d in acc.dims) != set(acc_coords.keys()):
                matched = False
                continue
            for dim in acc.dims:
                if type(dim.accumulation) != accums[dim.key][0]:
                    matched = False
                    break
                if dim.accumulation.coords != acc_coords[dim.key]:
                    matched = False
                    break
            if matched:
                break
        assert matched
        acc_coords_all.pop(i)
    assert not acc_coords_all


def assert_empty(it: Iterator) -> None:
    __tracebackhide__ = True  # For pytest
    for x in it:
        assert False, f"Expected empty iterator, got <{x!r}, ...>"


@pytest.mark.parametrize(
    "config, checkpoints",
    [
        pytest.param(
            CONFIGS["single-step"], {6: {"accum0": [6.0, 12.0, 24.0]}}, id="single-step"
        ),
        pytest.param(
            CONFIGS["step-range"],
            {48: {"accum0": [36.0, 72.0, 144.0]}},
            id="step-range",
        ),
        pytest.param(
            CONFIGS["multi-step"],
            {
                6: {"accum0": [6.0, 12.0, 24.0]},
                12: {"accum1": [12.0, 24.0, 48.0]},
                18: {"accum2": [18.0, 36.0, 72.0]},
                24: {"accum3": [24.0, 48.0, 96.0]},
            },
            id="multi-step",
        ),
        pytest.param(
            CONFIGS["multi-step-range"],
            {
                24: {"accum0": [15.0, 30.0, 60.0]},
                36: {"accum1": [27.0, 54.0, 108.0]},
                48: {"accum2": [39.0, 78.0, 156.0]},
            },
            id="multi-step-range",
        ),
        pytest.param(
            CONFIGS["multi-step-range-sameend"],
            {
                48: {"accum0": [120.0, 240.0, 480.0], "accum1": [180.0, 360.0, 720.0]},
            },
            id="multi-step-range-sameend",
        ),
    ],
)
def test_feed_singledim(config, checkpoints):
    mgr = AccumulationManager(config)
    assert len(mgr.accumulations) == sum(len(chp) for chp in checkpoints.values())
    todo = set(mgr.accumulations.keys())
    sorted_steps = sorted(mgr.coords["step"])
    values = np.array([1.0, 2.0, 4.0])
    for step in sorted_steps:
        completed = mgr.feed({"step": step}, step * values)
        expected = checkpoints.get(step)
        if expected is None:
            assert_empty(completed)
            continue

        completed = list(completed)
        assert len(completed) == len(expected)
        for name, accum in completed:
            assert name in expected
            assert name in todo
            assert accum.is_complete()
            np.testing.assert_equal(accum.values, expected[name])
            todo.remove(name)
    assert not todo


@pytest.mark.parametrize(
    "config, checkpoints",
    [
        pytest.param(
            CONFIGS["onechunk-multidim"],
            {(1000, 20230204, 24): {"accum0": [1.0, 2.0, 4.0]}},
            id="onechunk-multidim",
        ),
        pytest.param(
            CONFIGS["single-multidim"],
            {
                (250, 12): {"accum0": [1.0, 2.0, 4.0]},
                (250, 24): {"accum1": [2.0, 4.0, 8.0]},
                (500, 12): {"accum2": [3.0, 6.0, 12.0]},
                (500, 24): {"accum3": [4.0, 8.0, 16.0]},
                (1000, 12): {"accum4": [5.0, 10.0, 20.0]},
                (1000, 24): {"accum5": [6.0, 12.0, 24.0]},
            },
            id="single-multidim",
        ),
        pytest.param(
            CONFIGS["multidim-multistep"],
            {
                (20230118, 24): {
                    "accum0": [
                        [2.5, 5.0, 10.0],
                        [10.5, 21.0, 42.0],
                        [18.5, 37.0, 74.0],
                        [26.5, 53.0, 106.0],
                    ]
                },
                (20230118, 36): {
                    "accum1": [
                        [4.5, 9.0, 18.0],
                        [12.5, 25.0, 50.0],
                        [20.5, 41.0, 82.0],
                        [28.5, 57.0, 114.0],
                    ]
                },
                (20230118, 48): {
                    "accum2": [
                        [6.5, 13.0, 26.0],
                        [14.5, 29.0, 58.0],
                        [22.5, 45.0, 90.0],
                        [30.5, 61.0, 122.0],
                    ]
                },
            },
            id="multidim-multistep",
        ),
    ],
)
def test_feed_multidim(config, checkpoints):
    mgr = AccumulationManager(config)
    assert len(mgr.accumulations) == sum(len(chp) for chp in checkpoints.values())
    todo = set(mgr.accumulations.keys())
    sorted_coords = {key: sorted(coords) for key, coords in mgr.coords.items()}
    values = np.array([1.0, 2.0, 4.0])
    for i, coords in enumerate(dict_product(sorted_coords)):
        completed = mgr.feed(coords, (i + 1) * values)
        expected = checkpoints.get(tuple(coords.values()))
        if expected is None:
            assert_empty(completed)
            continue

        completed = list(completed)
        assert len(completed) == len(expected)
        for name, accum in completed:
            assert name in expected
            assert name in todo
            assert accum.is_complete()
            np.testing.assert_equal(accum.values, expected[name])
            todo.remove(name)
    assert not todo


@pytest.mark.parametrize(
    "config, todel, rest, coords",
    [
        pytest.param(
            CONFIGS["single-multidim"],
            ["accum0", "accum1", "accum2", "accum4"],
            ["accum3", "accum5"],
            {"levelist": {500, 1000}, "step": {24}},
            id="simgle-multidim",
        ),
        pytest.param(
            CONFIGS["multidim-multistep"],
            ["accum1"],
            ["accum0", "accum2"],
            {
                "hdate": {20200118, 20210118, 20220118, 20230118},
                "step": {6, 12, 18, 24, 30, 36, 42, 48},
            },
            id="multidim-multistep",
        ),
    ],
)
def test_delete(config, todel, rest, coords):
    mgr = AccumulationManager(config)
    mgr.delete(todel)
    assert sorted(mgr.accumulations.keys()) == rest
    assert mgr.coords == coords
