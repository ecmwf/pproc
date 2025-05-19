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
    "factory-default": {"step": {"type": "default", "coords": [[12], [18]]}},
    "factory-legacy": {
        "step": {
            "type": "legacywindow",
            "windows": [{"periods": [{"range": [24, 24]}, {"range": [48, 48]}]}],
        }
    },
    "factory-multidim": {
        "hdate": {"coords": [[20220610], [20230610]]},
        "step": {
            "type": "legacywindow",
            "windows": [
                {"window_operation": "mean", "periods": [{"range": [0, 24, 6]}]}
            ],
        },
    },
    "factory-dateseq-bracket": {
        "date": {
            "type": "dateseq",
            "sequence": "ecmwf-mon-thu",
            "coords": [
                {
                    "bracket": {
                        "before": 2,
                        "after": 2,
                        "date": "20240718",
                        "strict": False,
                    }
                },
                {"bracket": {"before": 2, "after": 2, "date": "20240725"}},
            ],
        },
        "step": {"coords": [[24]]},
    },
    "factory-dateseq-range": {
        "date": {
            "type": "dateseq",
            "operation": "mean",
            "sequence": {
                "type": "monthly",
                "days": [5, 13, 17, 29],
            },
            "coords": [
                {"range": {"from": "20240820", "to": "20240920"}},
                {"range": {"from": "20240905", "to": "20241005", "include_end": False}},
            ],
        },
        "step": {"coords": [[24]]},
    },
    "factory-monthly-steps": {
        "step": {
            "operation": "mean",
            "type": "stepseq",
            "sequence": {
                "type": "monthly",
                "date": "20241001",
            },
            "coords": {
                "from": 0,
                "to": 5160,
                "by": 6,
            },
            "include_start_step": True,
        }
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
    "factory-default": {"step": (Aggregation, [[12], [18]])},
    "factory-legacy": {"step": (Aggregation, [[24], [48]])},
    "factory-multidim": {
        "hdate": (Aggregation, [[20220610], [20230610]]),
        "step": (Mean, [[6, 12, 18, 24]]),
    },
    "factory-dateseq-bracket": {
        "date": (
            Aggregation,
            [
                ["20240711", "20240715", "20240718", "20240722", "20240725"],
                ["20240718", "20240722", "20240729", "20240801"],
            ],
        ),
        "step": (Aggregation, [[24]]),
    },
    "factory-dateseq-range": {
        "date": (
            Mean,
            [
                ["20240829", "20240905", "20240913", "20240917"],
                ["20240905", "20240913", "20240917", "20240929"],
            ],
        ),
        "step": (Aggregation, [[24]]),
    },
    "factory-monthly-steps": {
        "step": (
            Mean,
            [
                list(range(0, 745, 6)),
                list(range(744, 1465, 6)),
                list(range(1464, 2209, 6)),
                list(range(2208, 2953, 6)),
                list(range(2952, 3625, 6)),
                list(range(3624, 4369, 6)),
                list(range(4368, 5089, 6)),
            ],
        )
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
    "factory-default": {"step": {12, 18}},
    "factory-legacy": {"step": {24, 48}},
    "factory-multidim": {"hdate": {20220610, 20230610}, "step": {6, 12, 18, 24}},
    "factory-dateseq-bracket": {
        "date": {
            "20240711",
            "20240715",
            "20240718",
            "20240722",
            "20240725",
            "20240729",
            "20240801",
        },
        "step": {24},
    },
    "factory-dateseq-range": {
        "date": {
            "20240829",
            "20240905",
            "20240913",
            "20240917",
            "20240929",
        },
        "step": {24},
    },
    "factory-monthly-steps": {"step": {x for x in range(0, 5089, 6)}},
}


@pytest.mark.parametrize(
    "config, accums, coords",
    [
        pytest.param(config, EXPECT_ACCUMS[name], EXPECT_COORDS[name], id=name)
        for name, config in CONFIGS.items()
    ],
)
def test_create(config, accums, coords):
    mgr = AccumulationManager.create(config)
    assert mgr.coords == coords
    assert mgr.sorted_coords() == {key: sorted(coo) for key, coo in coords.items()}
    acc_coords_all = list(dict_product({key: coo for key, (_, coo) in accums.items()}))
    for name, acc in mgr.accumulations.items():
        matched = False
        for i, acc_coords in enumerate(acc_coords_all):
            matched = True
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
        assert matched, f"{name} not matched"
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
            CONFIGS["single-step"], {6: {"6": [6.0, 12.0, 24.0]}}, id="single-step"
        ),
        pytest.param(
            CONFIGS["step-range"],
            {48: {"24-48": [36.0, 72.0, 144.0]}},
            id="step-range",
        ),
        pytest.param(
            CONFIGS["multi-step"],
            {
                6: {"6": [6.0, 12.0, 24.0]},
                12: {"12": [12.0, 24.0, 48.0]},
                18: {"18": [18.0, 36.0, 72.0]},
                24: {"24": [24.0, 48.0, 96.0]},
            },
            id="multi-step",
        ),
        pytest.param(
            CONFIGS["multi-step-range"],
            {
                24: {"6-24": [15.0, 30.0, 60.0]},
                36: {"18-36": [27.0, 54.0, 108.0]},
                48: {"30-48": [39.0, 78.0, 156.0]},
            },
            id="multi-step-range",
        ),
        pytest.param(
            CONFIGS["multi-step-range-sameend"],
            {
                48: {"12-48": [120.0, 240.0, 480.0], "24-48": [180.0, 360.0, 720.0]},
            },
            id="multi-step-range-sameend",
        ),
    ],
)
def test_feed_singledim(config, checkpoints):
    mgr = AccumulationManager.create(config)
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
            {
                (1000, 20230204, 24): {
                    "levelist_1000:hdate_20230204:step_24": [1.0, 2.0, 4.0]
                }
            },
            id="onechunk-multidim",
        ),
        pytest.param(
            CONFIGS["single-multidim"],
            {
                (250, 12): {"levelist_250:step_12": [1.0, 2.0, 4.0]},
                (250, 24): {"levelist_250:step_24": [2.0, 4.0, 8.0]},
                (500, 12): {"levelist_500:step_12": [3.0, 6.0, 12.0]},
                (500, 24): {"levelist_500:step_24": [4.0, 8.0, 16.0]},
                (1000, 12): {"levelist_1000:step_12": [5.0, 10.0, 20.0]},
                (1000, 24): {"levelist_1000:step_24": [6.0, 12.0, 24.0]},
            },
            id="single-multidim",
        ),
        pytest.param(
            CONFIGS["multidim-multistep"],
            {
                (20230118, 24): {
                    "hdate_20200118-20230118:step_6-24": [
                        [2.5, 5.0, 10.0],
                        [10.5, 21.0, 42.0],
                        [18.5, 37.0, 74.0],
                        [26.5, 53.0, 106.0],
                    ]
                },
                (20230118, 36): {
                    "hdate_20200118-20230118:step_18-36": [
                        [4.5, 9.0, 18.0],
                        [12.5, 25.0, 50.0],
                        [20.5, 41.0, 82.0],
                        [28.5, 57.0, 114.0],
                    ]
                },
                (20230118, 48): {
                    "hdate_20200118-20230118:step_30-48": [
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
    mgr = AccumulationManager.create(config)
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
            [
                "levelist_250:step_12",
                "levelist_250:step_24",
                "levelist_500:step_12",
                "levelist_1000:step_12",
            ],
            ["levelist_500:step_24", "levelist_1000:step_24"],
            {"levelist": {500, 1000}, "step": {24}},
            id="simgle-multidim",
        ),
        pytest.param(
            CONFIGS["multidim-multistep"],
            ["hdate_20200118-20230118:step_18-36"],
            ["hdate_20200118-20230118:step_6-24", "hdate_20200118-20230118:step_30-48"],
            {
                "hdate": {20200118, 20210118, 20220118, 20230118},
                "step": {6, 12, 18, 24, 30, 36, 42, 48},
            },
            id="multidim-multistep",
        ),
    ],
)
def test_delete(config, todel, rest, coords):
    mgr = AccumulationManager.create(config)
    mgr.delete(todel)
    assert sorted(mgr.accumulations.keys()) == sorted(rest)
    assert mgr.coords == coords
