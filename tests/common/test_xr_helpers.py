# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import multiprocessing

import numpy as np
import pytest
import xarray as xr

from pproc.common import parallel
from pproc.common.xr_helpers import DatasetBuilder


def make_da(name="foo", lons=(0.0, 10.0, 20.0), region=None, val=None):
    values = np.arange(len(lons), dtype=float) + 1.0
    if val is not None:
        values = np.full(len(lons), float(val))
    da = xr.DataArray(values, coords={"lon": list(lons)}, dims=["lon"], name=name)
    if region is not None:
        da = da.expand_dims({"region": [region]})
    return da


def make_da_blocks(**kwargs):
    return [
        make_da(lons=(0.0, 10.0, 20.0), region="nh", **kwargs),
        make_da(lons=(30.0, 40.0, 50.0), region="nh", **kwargs),
        make_da(lons=(0.0, 10.0, 20.0), region="sh", **kwargs, val=1.0),
        make_da(lons=(30.0, 40.0, 50.0), region="sh", **kwargs, val=2.0),
    ]


def commit(target, dataarrays):
    for da in dataarrays:
        target.stage(da)
    target.commit()


class TestDatasetBuilder:
    @pytest.fixture
    def builder(self):
        return DatasetBuilder()

    def test_to_dataset_without_commits(self, builder):
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_stage_does_not_commit(self, builder):
        builder.stage(make_da())
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_commit_without_stages(self, builder):
        builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_commit_rejects_variables_covering_different_blocks(self, builder):
        builder.stage(make_da("foo"))
        builder.stage(make_da("foo", lons=(30.0, 40.0, 50.0)))
        builder.stage(make_da("bar"))
        with pytest.raises(ValueError):
            builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_commit_rejects_sparse_cube(self, builder):
        a, b, c, d = make_da_blocks()
        builder.stage(a)
        builder.stage(b)
        builder.stage(d)
        with pytest.raises(ValueError):
            builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())
        builder.stage(c)  # complete the cube to allow commiting
        builder.commit()
        expected = xr.combine_by_coords([a, b, c, d])
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_commit_rejects_duplicates(self, builder):
        builder.stage(make_da())
        builder.stage(make_da())
        with pytest.raises(ValueError):
            builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_to_dataset_one_dataarray(self, builder):
        da = make_da()
        commit(builder, [da])
        xr.testing.assert_identical(builder.to_dataset(), da.to_dataset())

    def test_to_dataset_two_blocks(self, builder):
        first, second, _, _ = make_da_blocks()
        commit(builder, [first, second])
        xr.testing.assert_identical(
            builder.to_dataset(), xr.concat([first, second], dim="lon").to_dataset()
        )

    def test_to_dataset_two_variables(self, builder):
        foo, bar = make_da("foo"), make_da("bar")
        commit(builder, [foo, bar])
        result = builder.to_dataset()
        assert list(result.data_vars) == ["foo", "bar"]
        xr.testing.assert_identical(result, xr.merge([foo, bar]))

    def test_to_dataset_two_variables_two_blocks(self, builder):
        a, b, _, _ = make_da_blocks()
        c = a.rename("bar")
        d = b.rename("bar")
        commit(builder, [a, c, b, d])  # interleaved, to cover out-of-order arrival
        expected = xr.combine_by_coords([a, b, c, d])
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_to_dataset_two_variables_one_with_extra_dim(self, builder):
        foo = make_da(name="foo", region=None)
        bar_nh = make_da(name="bar", region="nh")
        bar_sh = make_da(name="bar", region="sh")
        commit(builder, [foo, bar_nh, bar_sh])
        expected = xr.combine_by_coords([foo, bar_nh, bar_sh])
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_to_dataset_is_repeatable(self, builder):
        a, b, c, d = make_da_blocks()
        commit(builder, [a, b])
        # Reading twice yields the same dataset
        xr.testing.assert_identical(builder.to_dataset(), builder.to_dataset())
        # Reading doesn't affect subsequent commits
        commit(builder, [c, d])
        expected = xr.combine_by_coords([a, b, c, d])
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_later_commit_overrides_earlier(self, builder):
        commit(builder, [make_da(region="nh", val=1.0), make_da(region="sh", val=1.0)])
        commit(builder, [make_da(region="nh", val=9.0)])
        expected = xr.concat(
            [make_da(region="nh", val=9.0), make_da(region="sh", val=1.0)], dim="region"
        ).to_dataset()
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_commits_may_form_a_sparse_cube(self, builder):
        a, b, c, _ = make_da_blocks()
        # Each commit is a cube on its own, the union of the two is not
        commit(builder, [b])
        commit(builder, [a, c])
        expected = xr.combine_by_coords([a, b, c], join="outer", fill_value=np.nan)
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_commits_may_stage_a_variable(self, builder):
        commit(builder, [make_da("foo")])
        commit(builder, [make_da("bar")])
        assert sorted(builder.to_dataset().data_vars) == ["bar", "foo"]

    def test_parallel_commits_reach_the_parent(self, builder):
        manager = multiprocessing.Manager()
        builder.enable_parallel(manager)
        blocks = make_da_blocks()
        parallel.parallel_processing(
            commit,
            [(builder, [block]) for block in blocks],
            2,
        )
        expected = xr.combine_by_coords(blocks)
        xr.testing.assert_identical(expected, builder.to_dataset())
