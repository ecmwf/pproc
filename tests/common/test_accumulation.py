# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pytest

from pproc.common.accumulation import (
    Accumulator,
    Aggregation,
    Difference,
    DifferenceRate,
    Dimension,
    Histogram,
    Integral,
    Mean,
    SimpleAccumulation,
    WeightedMean,
    StandardDeviation,
    DeaccumulationWrapper,
    convert_coords,
    convert_dim,
    convert_dims,
    convert_range,
    create_accumulation,
)


def test_simple_acc():
    acc = SimpleAccumulation("add", range(0, 13, 6))
    assert not acc.is_complete()
    data = np.array([1.0, 2.0, 3.0])
    acc.feed(0, 0.0 * data)
    assert not acc.is_complete()
    acc.feed(6, data)
    assert not acc.is_complete()
    acc.feed(12, data**2)
    assert acc.is_complete()
    np.testing.assert_almost_equal(acc.get_values(), [2.0, 6.0, 12.0])


def test_accumulation_contains():
    assert 12 in SimpleAccumulation("add", range(0, 25, 6))
    assert "b" in Difference(["a", "b"])
    assert 6 in Integral(0, [4, 6])


def test_aggregation():
    agg = Aggregation([250, 500, 1000])
    assert not agg.is_complete()
    data = np.array([4.0, 9.0, 16.0])
    agg.feed(500, data)
    agg.feed(1000, 0.5 * data)
    agg.feed(250, 2 * data)
    assert agg.is_complete()
    np.testing.assert_almost_equal(
        agg.get_values(), [[8.0, 18.0, 32.0], [4.0, 9.0, 16.0], [2.0, 4.5, 8.0]]
    )


def test_convert_range():
    assert convert_range({"to": 5}) == range(6)
    assert convert_range({"from": 2, "to": 10}) == range(2, 11)
    assert convert_range({"to": 12, "by": 6}) == range(0, 13, 6)
    assert convert_range({"from": 4, "to": 16, "by": 4}) == range(4, 17, 4)
    with pytest.raises(ValueError):
        convert_range({"from": 8, "by": 8})


def test_convert_coords():
    assert convert_coords({"from": 2, "to": 10, "by": 4}) == range(2, 11, 4)
    assert convert_coords([1, 2, 4]) == [1, 2, 4]
    assert convert_coords(
        [{"from": 0, "to": 24, "by": 6}, {"from": 36, "to": 48, "by": 12}]
    ) == [0, 6, 12, 18, 24, 36, 48]
    assert convert_coords([6, {"from": 12, "to": 36, "by": 12}, 48]) == [
        6,
        12,
        24,
        36,
        48,
    ]
    with pytest.raises(ValueError):
        convert_coords({"from": 4})
    with pytest.raises(ValueError):
        convert_coords([1, 2, {"by": 7}, 4])


@pytest.mark.parametrize(
    "config, acc_cls, used_coords, exp_values",
    [
        pytest.param(
            {"operation": "sum", "coords": {"to": 4, "by": 2}},
            SimpleAccumulation,
            [0, 2, 4],
            [[9.0, 18.0, 33.0], [9.0, 30.0, 87.0]],
            id="sum",
        ),
        pytest.param(
            {"operation": "minimum", "coords": {"to": 4, "by": 2}},
            SimpleAccumulation,
            [0, 2, 4],
            [[1.0, 4.0, 9.0], [1.0, 8.0, 27.0]],
            id="min",
        ),
        pytest.param(
            {"operation": "maximum", "coords": {"to": 4, "by": 2}},
            SimpleAccumulation,
            [0, 2, 4],
            [[5.0, 8.0, 13.0], [5.0, 12.0, 31.0]],
            id="max",
        ),
        pytest.param(
            {"operation": "integral", "coords": [0, 3, 4]},
            Integral,
            [3, 4],
            [[17.0, 29.0, 49.0], [17.0, 45.0, 121.0]],
            id="integral",
        ),
        pytest.param(
            {"operation": "difference", "coords": [0, 4]},
            Difference,
            [0, 4],
            [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
            id="difference",
        ),
        pytest.param(
            {"operation": "difference", "coords": [4]},
            Difference,
            [4],
            [[5.0, 8.0, 13.0], [5.0, 12.0, 31.0]],
            id="difference_nostart",
        ),
        pytest.param(
            {"operation": "difference_rate", "coords": [0, 4]},
            DifferenceRate,
            [0, 4],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            id="difference_rate",
        ),
        pytest.param(
            {"operation": "difference_rate", "coords": [0, 4], "factor": 0.5},
            DifferenceRate,
            [0, 4],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            id="difference_rate_factor",
        ),
        pytest.param(
            {"operation": "difference_rate", "coords": [4], "factor": 0.125},
            DifferenceRate,
            [4],
            [[10.0, 16.0, 26.0], [10.0, 24.0, 62.0]],
            id="difference_rate_factor_nostart",
        ),
        pytest.param(
            {"operation": "mean", "coords": {"to": 4, "by": 2}},
            Mean,
            [0, 2, 4],
            [[3.0, 6.0, 11.0], [3.0, 10.0, 29.0]],
            id="mean",
        ),
        pytest.param(
            {"operation": "weighted_mean", "coords": [0, 3, 4]},
            WeightedMean,
            [3, 4],
            [[4.25, 7.25, 12.25], [4.25, 11.25, 30.25]],
            id="weighted_mean",
        ),
        pytest.param(
            {
                "operation": "histogram",
                "coords": {"to": 4, "by": 2},
                "bins": [0.5, 5.5, 10.5, 20.5, 40.5],
            },
            Histogram,
            [0, 2, 4],
            [
                [[1.0, 1.0 / 3.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 2.0 / 3.0, 1.0 / 3.0], [0.0, 2.0 / 3.0, 0.0]],
                [[0.0, 0.0, 2.0 / 3.0], [0.0, 1.0 / 3.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            id="histogram",
        ),
        pytest.param(
            {
                "operation": "histogram",
                "coords": [0, 2, 3, 4],
                "bins": [-0.5, 0.5, 1.5, 2.5],
                "mod": 3.0,
                "normalise": False,
            },
            Histogram,
            [0, 2, 3, 4],
            [
                [[1.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
                [[2.0, 2.0, 1.0], [2.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]],
            ],
            id="histogram-mod",
        ),
        pytest.param(
            {
                "operation": "histogram",
                "coords": {"to": 4, "by": 2},
                "bins": [0.5, 5.5, 10.5, 20.5, 40.5],
                "scale_out": 6.0,
                "normalise": True,
            },
            Histogram,
            [0, 2, 4],
            [
                [[6.0, 2.0, 0.0], [6.0, 0.0, 0.0]],
                [[0.0, 4.0, 2.0], [0.0, 4.0, 0.0]],
                [[0.0, 0.0, 4.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0]],
            ],
            id="histogram-scale",
        ),
        pytest.param(
            {"operation": "aggregation", "coords": {"to": 4, "by": 2}},
            Aggregation,
            [0, 2, 4],
            [
                [[1.0, 4.0, 9.0], [1.0, 8.0, 27.0]],
                [[3.0, 6.0, 11.0], [3.0, 10.0, 29.0]],
                [[5.0, 8.0, 13.0], [5.0, 12.0, 31.0]],
            ],
            id="aggregation",
        ),
        pytest.param(
            {"coords": {"from": 2, "to": 4, "by": 2}},
            Aggregation,
            [2, 4],
            [
                [[3.0, 6.0, 11.0], [3.0, 10.0, 29.0]],
                [[5.0, 8.0, 13.0], [5.0, 12.0, 31.0]],
            ],
            id="default",
        ),
        pytest.param(
            {"operation": "mean", "coords": {"to": 4}, "sequential": True},
            Mean,
            [0, 2, 3, 4],
            [[3.25, 6.25, 11.25], [3.25, 10.25, 29.25]],
            id="mean_sequential",
        ),
        pytest.param(
            {"operation": "standard_deviation", "coords": {"to": 4, "by": 2}},
            StandardDeviation,
            [0, 2, 4],
            np.ones((2, 3)) * 1.632993161855452,
            id="std",
        ),
        pytest.param(
            {"operation": "mean", "coords": {"to": 4, "by": 2}, "deaccumulate": True},
            DeaccumulationWrapper,
            [0, 2, 4],
            [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
            id="deaccum-mean",
        ),
        pytest.param(
            {
                "operation": "sum",
                "coords": {"to": 4},
                "deaccumulate": True,
                "sequential": True,
            },
            DeaccumulationWrapper,
            [0, 2, 3, 4],
            [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]],
            id="deaccum-sum-seq",
        ),
    ],
)
def test_accumulations(config, acc_cls, used_coords, exp_values):
    acc = create_accumulation(config)
    assert type(acc) is acc_cls

    data = np.array([[1.0, 4.0, 9.0], [1.0, 8.0, 27.0]])
    for c in [0, 2, 3, 4]:
        assert not acc.is_complete()
        assert acc.feed(c, data + c) == (c in used_coords)
    assert acc.is_complete()
    np.testing.assert_almost_equal(acc.get_values(), exp_values)

    acc.reset()
    for c in [0, 2, 3, 4]:
        assert not acc.is_complete()
        assert acc.feed(c, data + c) == (c in used_coords)
    assert acc.is_complete()
    np.testing.assert_almost_equal(acc.get_values(), exp_values)


def test_std_zero():
    acc = create_accumulation(
        {"operation": "standard_deviation", "coords": {"to": 4, "by": 2}}
    )

    data = np.ones((2, 3)) * 0.1
    for c in [0, 2, 4]:
        acc.feed(c, data)
    np.testing.assert_almost_equal(acc.get_values(), np.zeros((2, 3)))


@pytest.mark.parametrize(
    "config, exp_values",
    [
        pytest.param(
            {
                "operation": "filter",
                "filter_op": "max",
                "filter_index": 0,
                "coords": {"from": 1, "to": 3},
                "sequential": True,
            },
            [
                [[5.0, 9.0, 9.0], [3.0, 8.0, 18.0]],
                [[1.0, 8.0, 27.0], [6.0, 24.0, 27.0]],
            ],
            id="filter-neighbours-0",
        ),
        pytest.param(
            {
                "operation": "filter",
                "filter_op": "max",
                "filter_index": 0,
                "neighbours": [-1],
                "neighbours_op": "mean",
                "coords": {"from": 1, "to": 3},
                "sequential": True,
            },
            [[[3.5, 5.0, 7.0], [2.5, 5.0, 11.5]], [[1.5, 6.0, 22.5], [5, 20.0, 40.5]]],
            id="filter-neighbours-neg1",
        ),
        pytest.param(
            {
                "operation": "filter",
                "filter_op": "min",
                "filter_index": 0,
                "neighbours": [1],
                "neighbours_op": "mean",
                "coords": {"from": 1, "to": 3},
                "sequential": True,
            },
            [[[1.5, 5.0, 3.0], [1.5, 5.0, 3]], [[2.5, 6.0, 13.5], [3, 20.0, 67.5]]],
            id="filter-neighbours-pos1",
        ),
    ],
)
def test_filter_accum(config, exp_values):
    acc = create_accumulation(config)

    step1 = np.array(
        [[[5.0, 1.0, 1.0], [1.0, 4.0, 18.0]], [[1.0, 4.0, 9.0], [2.0, 8.0, 27.0]]]
    )
    step2 = np.array(
        [[[2.0, 9.0, 5.0], [2.0, 2.0, 5.0]], [[2.0, 8.0, 18.0], [4.0, 16.0, 54.0]]]
    )
    step3 = np.array(
        [[[1.0, 4.0, 9.0], [3.0, 8.0, 1.0]], [[3.0, 12.0, 27.0], [6.0, 24.0, 81.0]]]
    )
    for index, data in enumerate([step1, step2, step3]):
        acc.feed(index + 1, data)
    assert acc.is_complete()
    np.testing.assert_almost_equal(acc.get_values(), exp_values)

    acc.reset()
    for index, data in enumerate([step1, step2, step3]):
        acc.feed(index + 1, data)
    assert acc.is_complete()
    np.testing.assert_almost_equal(acc.get_values(), exp_values)


def test_convert_dim():
    acc = Mean(range(25, 6))
    assert convert_dim(("step", acc)) == Dimension("step", acc)
    assert convert_dim(Dimension("step", acc)) == Dimension("step", acc)


def test_convert_dims():
    dims = {"step": Mean(range(24, 49, 3)), "levelist": Aggregation([250, 500, 1000])}
    expected = [Dimension(key, acc) for key, acc in dims.items()]
    assert convert_dims([Dimension(key, acc) for key, acc in dims.items()]) == expected
    assert convert_dims(list(dims.items())) == expected
    assert convert_dims(dims.items()) == expected
    assert convert_dims(dims) == expected
    assert (
        convert_dims([Dimension("step", dims["step"]), ("levelist", dims["levelist"])])
        == expected
    )


@pytest.mark.parametrize(
    "config, grib_key_values",
    [
        pytest.param(
            {
                "coords": [1],
                "operation": "aggregation", 
            }, 
            {"step": "1", "timeRangeIndicator": 0}, id="inst"
        ),
        pytest.param(
            {
                "coords": [0],
                "operation": "aggregation",
            },
            {"step": "0", "timeRangeIndicator": 1},
            id="inst-0",
        ),
        pytest.param(
            {
                "coords": [260],
                "operation": "aggregation",
            },
            {"step": "260", "timeRangeIndicator": 10},
            id="inst-260",
        ),
        pytest.param(
            {
                "coords": [1, 2], 
                "operation": "maximum", 
            }, 
            {"stepRange": "1-2", "stepType": "max"}, id="range"
        ),
        pytest.param(
            {
                "coords": list(range(320, 361)),
                "operation": "maximum",
            },
            {"stepRange": "320-360", "stepType": "max", "unitOfTimeRange": 11},
            id="range-360",
        ),
        pytest.param(
            {
                "coords": [1, 2],
                "operation": "mean",
                "metadata": {"stepType": "avg", "numberIncludedInAverage": 2},
            },
            {"numberIncludedInAverage": 2, "stepRange": "1-2", "stepType": "avg"},
            id="extra",
        ),
        pytest.param(
            {
                "coords": [1, 2],
                "operation": "mean",
                "name": {
                    "type": "default", 
                    "length": 2,
                }, 
            },
            {"stepRange": "0-2", "stepType": "max"},
            id="name",
        ),
        pytest.param(
            {
                "coords": range(6, 721, 6),
                "operation": "mean",
                "name": {
                    "type": "monthly", 
                    "date": "20250601",
                }, 
            },
            {"stepRange": "0-720", "stepType": "max", "unitOfTimeRange": 11},
            id="name-monthly",
        ),
    ],
)
def test_grib_header(config, grib_key_values):
    accum = create_accumulation(config)
    header = accum.grib_keys("step")
    assert header == grib_key_values


def test_accumulator_contains():
    assert {"step": 12} in Accumulator({"step": Mean(range(6, 24, 6))})
    assert {"hdate": 20200301, "step": 6, "example": "a"} in Accumulator(
        {
            "hdate": Aggregation([y * 10000 + 301 for y in range(2010, 2021)]),
            "step": SimpleAccumulation("maximum", range(0, 24, 3)),
            "example": Difference(["a", "b"]),
        }
    )
    assert {"example": "c"} not in Accumulator({"example": Difference(["a", "b"])})
    assert {"step": 12, "levelist": 250} not in Accumulator(
        {
            "step": Aggregation(range(3, 24, 3)),
            "levelist": Integral(10, [100, 500, 1000]),
        }
    )

    with pytest.raises(KeyError):
        {"step": 6} in Accumulator(
            {
                "step": SimpleAccumulation("minimum", range(6, 24, 6)),
                "levelist": SimpleAccumulation("add", [100, 500, 1000]),
            }
        )


def test_accumulator_getitem():
    acc = Accumulator({"step": Mean(range(6, 24, 6))})
    assert acc["step"] is acc.dims[0].accumulation
    with pytest.raises(KeyError):
        acc["date"]

    acc = Accumulator(
        {
            "hdate": Aggregation([y * 10000 + 301 for y in range(2010, 2021)]),
            "step": SimpleAccumulation("maximum", range(0, 24, 3)),
            "example": Difference(["a", "b"]),
        }
    )
    for dim in acc.dims:
        assert acc[dim.key] is dim.accumulation


def test_accumulator():
    expvers = ["0001", "0002"]
    levels = [250, 500, 1000]
    steps = range(0, 25, 6)
    acc = Accumulator(
        {
            "expver": Difference(expvers),
            "levelist": Aggregation(levels),
            "step": Mean(steps),
        }
    )

    data = np.array([[1.0, 2.0, 5.0], [1.0, 3.0, 8.0]])

    for expver in expvers:
        assert not acc.is_complete()
        for level in levels:
            assert not acc.is_complete()
            for step in steps:
                assert not acc.is_complete()
                values = int(expver) * level * step * data
                acc.feed({"expver": expver, "levelist": level, "step": step}, values)

    assert acc.is_complete()

    np.testing.assert_almost_equal(
        acc.values,
        [
            [[3000.0, 6000.0, 15000.0], [3000.0, 9000.0, 24000.0]],
            [[6000.0, 12000.0, 30000.0], [6000.0, 18000.0, 48000.0]],
            [[12000.0, 24000.0, 60000.0], [12000.0, 36000.0, 96000.0]],
        ],
    )


def test_create_accumulator():
    config = {
        "hdate": {
            "operation": "aggregation",
            "coords": [20000101, 20010101, 20020101],
        },
        "step": {
            "operation": "maximum",
            "coords": {"from": 24, "to": 48, "by": 6},
        },
    }
    acc = Accumulator.create(config)

    assert acc.dims[0].key == "hdate"
    assert type(acc.dims[0].accumulation) is Aggregation
    assert acc.dims[0].accumulation.coords == [20000101, 20010101, 20020101]

    assert acc.dims[1].key == "step"
    assert type(acc.dims[1].accumulation) is SimpleAccumulation
    assert acc.dims[1].accumulation.operation is np.maximum
    assert acc.dims[1].accumulation.coords == range(24, 49, 6)


def test_multi_accum():
    accum = create_accumulation({"coords": [1, 2], "operation": "sum"})
    accum2 = create_accumulation({"coords": [1, 2], "operation": "sum"})
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(1, step_values)
    accum2.feed(1, step_values)
    step_values = np.array([[1, 2, 3], [2, 4, 6]])
    accum.feed(2, step_values * 2)
    values = accum.get_values()
    assert values is not None
    np.testing.assert_equal(values, np.array([[3, 6, 9], [6, 12, 18]]))
    accum2.feed(2, step_values)
    values2 = accum2.get_values()
    assert values2 is not None
    np.testing.assert_equal(values2, np.array([[2, 4, 6], [4, 8, 12]]))
