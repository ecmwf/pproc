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


def make_ds(attrs=None, var_attrs=None, lon_attrs=None, name="foo", **kwargs):
    """Dataset with attributes on the dataset, its variable and the lon coordinate."""
    ds = make_da(name=name, **kwargs).to_dataset()
    ds.attrs.update(attrs or {})
    ds[name].attrs.update(var_attrs or {})
    ds["lon"].attrs.update(lon_attrs or {})
    return ds


def commit(target, dataarrays):
    for da in dataarrays:
        target.stage(da)
    target.commit()


class TestDatasetBuilder:
    @pytest.fixture
    def builder(self):
        return DatasetBuilder()

    @pytest.fixture
    def manager(self):
        with multiprocessing.Manager() as manager:
            yield manager

    # __init__

    def test_init_combine_kwargs_passed_through(self):
        a = make_da(lons=(0.0, 10.0))
        b = make_da(lons=(20.0, 30.0))
        a.attrs["origin"] = "a"
        b.attrs["origin"] = "b"
        with pytest.raises(ValueError):
            commit(DatasetBuilder(), [a, b])
        builder = DatasetBuilder({"combine_attrs": "drop"})
        commit(builder, [a, b])
        assert builder.to_dataset().attrs == {}

    def test_init_combine_kwargs_cannot_override_join_and_fill_value(self):
        builder = DatasetBuilder({"join": "outer", "fill_value": 0})
        a, b, _, d = make_da_blocks()
        builder.stage(a)
        builder.stage(b)
        builder.stage(d)
        with pytest.raises(ValueError):
            builder.commit()

    def test_init_combine_kwargs_has_internal_copy(self):
        kwargs = {"combine_attrs": "drop"}
        DatasetBuilder(kwargs)
        assert kwargs == {"combine_attrs": "drop"}

    # enable_parallel

    def test_enable_parallel_keeps_existing_commits(self, builder, manager):
        a, b, _, _ = make_da_blocks()
        commit(builder, [a])
        builder.enable_parallel(manager)
        commit(builder, [b])
        xr.testing.assert_identical(builder.to_dataset(), xr.combine_by_coords([a, b]))

    def test_enable_parallel_commits_reach_the_parent(self, builder, manager):
        builder.enable_parallel(manager)
        blocks = make_da_blocks()
        parallel.parallel_processing(
            commit,
            [(builder, [block]) for block in blocks],
            2,
        )
        expected = xr.combine_by_coords(blocks)
        xr.testing.assert_identical(expected, builder.to_dataset())

    # stage

    def test_stage_returns_self(self, builder):
        assert builder.stage(make_da()) is builder

    def test_stage_does_not_commit(self, builder):
        builder.stage(make_da())
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_stage_rejects_unnamed_dataarray(self, builder):
        with pytest.raises(ValueError):
            builder.stage(make_da(name=None))

    def test_stage_dataset(self, builder):
        foo, bar = make_da("foo"), make_da("bar")
        builder.stage(foo.to_dataset())
        builder.stage(bar)
        builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.merge([foo, bar]))

    # commit

    def test_commit_allows_method_chaining(self, builder):
        assert builder.commit() is builder
        assert builder.stage(make_da()).commit() is builder

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

    def test_commit_loads_dask_arrays(self, builder):
        pytest.importorskip("dask")
        commit(builder, [make_da().chunk()])
        assert not builder.to_dataset().chunks

    def test_commit_does_not_mutate_inputs(self, builder):
        a, b, _, _ = make_da_blocks()
        a_orig, b_orig = a.copy(deep=True), b.copy(deep=True)
        commit(builder, [a, b])
        xr.testing.assert_identical(a, a_orig)
        xr.testing.assert_identical(b, b_orig)

    # reset_staged

    def test_reset_staged_allows_method_chaining(self, builder):
        assert builder.reset_staged() is builder

    def test_reset_staged_discards_pending_by_default(self, builder):
        builder.stage(make_da())
        builder.reset_staged()
        builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_reset_staged_after_failed_commit(self, builder):
        builder.stage(make_da())
        builder.stage(make_da())
        with pytest.raises(ValueError):
            builder.commit()
        builder.reset_staged()
        commit(builder, [make_da()])
        xr.testing.assert_identical(builder.to_dataset(), make_da().to_dataset())

    # undo_commit

    def test_undo_commit_allows_method_chaining(self, builder):
        assert builder.stage(make_da()).commit().undo_commit() is builder

    def test_undo_commit_without_commits(self, builder):
        with pytest.raises(RuntimeError):
            builder.undo_commit()

    def test_undo_commit_removes_last_commit(self, builder):
        a, b, c, d = make_da_blocks()
        commit(builder, [a, b])
        commit(builder, [c, d])
        builder.undo_commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.combine_by_coords([a, b]))

    def test_undo_commit_restores_overridden_values(self, builder):
        commit(builder, [make_da(val=1.0)])
        commit(builder, [make_da(lons=(10.0,), val=9.0)])
        builder.undo_commit()
        xr.testing.assert_identical(
            builder.to_dataset(), make_da(val=1.0).to_dataset()
        )

    def test_undo_commit_without_restage_leaves_staging_empty(self, builder):
        a, b, c, d = make_da_blocks()
        commit(builder, [a, b])
        commit(builder, [c, d])
        builder.undo_commit()
        builder.commit()  # nothing staged, nothing happens
        xr.testing.assert_identical(builder.to_dataset(), xr.combine_by_coords([a, b]))

    def test_undo_commit_restage_recommits_identically(self, builder):
        a, b, c, d = make_da_blocks()
        commit(builder, [a, b])
        commit(builder, [c, d])
        before = builder.to_dataset()
        builder.undo_commit(stage=True)
        builder.commit()
        xr.testing.assert_identical(builder.to_dataset(), before)

    # squash_commits

    def test_squash_commits_allows_method_chaining(self, builder):
        assert builder.squash_commits() is builder

    def test_squash_commits_empty(self, builder):
        builder.squash_commits()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_squash_commits_preserves_dataset(self, builder):
        commit(builder, [make_da(region="nh", val=1.0), make_da(region="sh", val=1.0)])
        commit(builder, [make_da(region="nh", val=9.0)])
        before = builder.to_dataset()
        builder.squash_commits()
        xr.testing.assert_identical(builder.to_dataset(), before)

    def test_squash_commits_leaves_single_commit(self, builder):
        a, b, _, _ = make_da_blocks()
        commit(builder, [a])
        commit(builder, [b])
        builder.squash_commits()
        builder.undo_commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_squash_commits_then_later_commit_overrides(self, builder):
        commit(builder, [make_da(region="nh", val=1.0), make_da(region="sh", val=1.0)])
        builder.squash_commits()
        commit(builder, [make_da(region="nh", val=9.0)])
        expected = xr.concat(
            [make_da(region="nh", val=9.0), make_da(region="sh", val=1.0)], dim="region"
        ).to_dataset()
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_squash_commits_on_shared_list(self, builder, manager):
        a, b, _, _ = make_da_blocks()
        builder.enable_parallel(manager)
        commit(builder, [a])
        commit(builder, [b])
        expected = xr.combine_by_coords([a, b])
        builder.squash_commits()
        xr.testing.assert_identical(builder.to_dataset(), expected)
        builder.undo_commit()
        xr.testing.assert_identical(builder.to_dataset(), xr.Dataset())

    def test_squash_commits_preserves_merged_attrs(self, builder):
        commit(builder, [make_ds({"title": "a", "only_a": 1})])
        commit(builder, [make_ds({"title": "b"}, lons=(30.0,))])
        before = builder.to_dataset()
        builder.squash_commits()
        xr.testing.assert_identical(builder.to_dataset(), before)
        assert builder.to_dataset().attrs == {"title": "b", "only_a": 1}

    # to_dataset

    def test_to_dataset_without_commits(self, builder):
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

    def test_to_dataset_two_variables_two_commits(self, builder):
        foo, bar = make_da("foo"), make_da("bar")
        commit(builder, [foo])
        commit(builder, [bar])
        xr.testing.assert_identical(builder.to_dataset(), xr.merge([foo, bar]))

    def test_to_dataset_is_repeatable(self, builder):
        a, b, c, d = make_da_blocks()
        commit(builder, [a, b])
        # Reading twice yields the same dataset
        xr.testing.assert_identical(builder.to_dataset(), builder.to_dataset())
        # Reading doesn't affect subsequent commits
        commit(builder, [c, d])
        expected = xr.combine_by_coords([a, b, c, d])
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_to_dataset_result_is_copy(self, builder):
        commit(builder, [make_da()])
        builder.to_dataset()["foo"].values[:] = 99.0
        xr.testing.assert_identical(builder.to_dataset(), make_da().to_dataset())

    def test_to_dataset_later_commit_overrides_earlier(self, builder):
        commit(builder, [make_da(region="nh", val=1.0), make_da(region="sh", val=1.0)])
        commit(builder, [make_da(region="nh", val=9.0)])
        expected = xr.concat(
            [make_da(region="nh", val=9.0), make_da(region="sh", val=1.0)], dim="region"
        ).to_dataset()
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_to_dataset_three_commits_latest_wins(self, builder):
        commit(builder, [make_da(val=1.0)])
        commit(builder, [make_da(lons=(10.0, 20.0), val=2.0)])
        commit(builder, [make_da(lons=(20.0,), val=3.0)])
        expected = make_da()
        expected.values[:] = [1.0, 2.0, 3.0]
        xr.testing.assert_identical(builder.to_dataset(), expected.to_dataset())

    def test_to_dataset_three_commits_with_gap(self, builder):
        # The middle commit does not overlap the newest one
        commit(builder, [make_da(val=1.0)])
        commit(builder, [make_da(lons=(30.0,), val=5.0)])
        commit(builder, [make_da(lons=(10.0, 30.0), val=3.0)])
        expected = make_da(lons=(0.0, 10.0, 20.0, 30.0))
        expected.values[:] = [1.0, 3.0, 1.0, 3.0]
        xr.testing.assert_identical(builder.to_dataset(), expected.to_dataset())

    def test_to_dataset_overlapping_commits_different_variables(self, builder):
        lons = (0.0, 10.0)
        commit(
            builder,
            [make_da("foo", lons=lons, val=1.0), make_da("bar", lons=lons, val=7.0)],
        )
        commit(builder, [make_da("foo", lons=(10.0,), val=2.0)])
        foo = make_da("foo", lons=lons)
        foo.values[:] = [1.0, 2.0]
        bar = make_da("bar", lons=lons, val=7.0)
        xr.testing.assert_identical(builder.to_dataset(), xr.merge([foo, bar]))

    def test_to_dataset_later_nan_does_not_override(self, builder):
        commit(builder, [make_da(val=1.0)])
        later = make_da()
        later.values[:] = [np.nan, 9.0, 9.0]
        commit(builder, [later])
        np.testing.assert_array_equal(
            builder.to_dataset()["foo"].values, [1.0, 9.0, 9.0]
        )

    def test_to_dataset_earlier_nan_is_filled_by_later(self, builder):
        earlier = make_da()
        earlier.values[:] = [1.0, np.nan, 3.0]
        commit(builder, [earlier])
        commit(builder, [make_da(val=9.0)])
        np.testing.assert_array_equal(
            builder.to_dataset()["foo"].values, [9.0, 9.0, 9.0]
        )

    def test_to_dataset_commits_may_form_a_sparse_cube(self, builder):
        a, b, c, _ = make_da_blocks()
        # Each commit is a cube on its own, the union of the two is not
        commit(builder, [b])
        commit(builder, [a, c])
        expected = xr.combine_by_coords([a, b, c], join="outer", fill_value=np.nan)
        xr.testing.assert_identical(builder.to_dataset(), expected)

    def test_to_dataset_outer_join_promotes_int_to_float(self, builder):
        a, b, _, _ = make_da_blocks()
        commit(builder, [a.astype(int)])
        assert builder.to_dataset()["foo"].dtype == int
        commit(builder, [b.astype(int)])
        assert builder.to_dataset()["foo"].dtype == float

    def test_to_dataset_identical_coordinates_preserve_int(self, builder):
        commit(builder, [make_da(val=1).astype(int)])
        commit(builder, [make_da(val=2).astype(int)])
        result = builder.to_dataset()
        assert result["foo"].dtype == int
        np.testing.assert_array_equal(result["foo"].values, [2, 2, 2])

    def test_to_dataset_attrs_are_merged_per_key(self, builder):
        commit(builder, [make_ds({"title": "a", "only_a": 1}, lons=(0.0, 10.0))])
        commit(builder, [make_ds({"title": "b", "only_b": 2}, lons=(20.0, 30.0))])
        assert builder.to_dataset().attrs == {"title": "b", "only_a": 1, "only_b": 2}

    def test_to_dataset_later_commit_without_attrs_keeps_earlier_attrs(self, builder):
        commit(builder, [make_ds({"title": "a"}, {"units": "K"}, {"units": "deg"})])
        commit(builder, [make_ds(lons=(30.0, 40.0))])
        result = builder.to_dataset()
        assert result.attrs == {"title": "a"}
        assert result["foo"].attrs == {"units": "K"}
        assert result["lon"].attrs == {"units": "deg"}

    def test_to_dataset_variable_attrs_are_merged_per_key(self, builder):
        commit(builder, [make_ds(var_attrs={"units": "K", "long_name": "temp"})])
        commit(builder, [make_ds(var_attrs={"units": "degC"}, lons=(30.0, 40.0))])
        assert builder.to_dataset()["foo"].attrs == {
            "units": "degC",
            "long_name": "temp",
        }

    def test_to_dataset_attrs_of_variables_from_different_commits(self, builder):
        commit(builder, [make_ds(var_attrs={"units": "K"}, name="foo")])
        commit(builder, [make_ds(var_attrs={"units": "m"}, name="bar")])
        result = builder.to_dataset()
        assert result["foo"].attrs == {"units": "K"}
        assert result["bar"].attrs == {"units": "m"}

    def test_to_dataset_attrs_over_three_commits(self, builder):
        commit(builder, [make_ds({"a": 1, "title": "first"})])
        commit(builder, [make_ds({"b": 2, "title": "second"}, lons=(30.0,))])
        commit(builder, [make_ds(lons=(40.0,))])  # specifies nothing
        assert builder.to_dataset().attrs == {"a": 1, "b": 2, "title": "second"}

    def test_to_dataset_merging_attrs_does_not_modify_commits(self, builder):
        commit(builder, [make_ds({"title": "a"}, {"units": "K"})])
        commit(builder, [make_ds({"title": "b"}, {"units": "m"}, lons=(30.0,))])
        builder.to_dataset()
        builder.undo_commit()
        result = builder.to_dataset()
        assert result.attrs == {"title": "a"}
        assert result["foo"].attrs == {"units": "K"}

    def test_to_dataset_drops_encoding(self, builder):
        ds = make_ds(lons=(0.0, 10.0))
        ds["foo"].encoding.update({"dtype": "int16", "scale_factor": 0.5})
        ds["lon"].encoding.update({"dtype": "float32"})
        ds.encoding.update({"unlimited_dims": {"lon"}})
        commit(builder, [ds])
        result = builder.to_dataset()
        assert result["foo"].encoding == {}
        assert result["lon"].encoding == {}
        assert result.encoding == {}
