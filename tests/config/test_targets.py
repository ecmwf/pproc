# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import numpy as np
import pydantic
import pytest
import xarray as xr

from pproc.common import parallel
from pproc.config.targets import NetCDFTarget


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
    """Write and flush, as an entrypoint iteration does."""
    for da in dataarrays:
        target.write(da)
    target.flush()


class TestNetCDFTarget:
    @pytest.fixture
    def target(self, tmpdir):
        return NetCDFTarget(path=str(tmpdir / "test.nc"))

    def test_nothing_is_written_without_a_commit(self, target):
        target.flush()  # nothing staged
        assert not os.path.exists(target.path)
        target.write(make_da())  # staged, but not flushed
        assert not os.path.exists(target.path)
        target.write(make_da())  # duplicate will be rejected by flush
        with pytest.raises(ValueError):
            target.flush()
        assert not os.path.exists(target.path)

    def test_targets_do_not_share_writes(self, tmpdir):
        first = NetCDFTarget(path=str(tmpdir / "first.nc"), format="netcdf")
        second = NetCDFTarget(path=str(tmpdir / "second.nc"), format="netcdf")
        commit(first, [make_da("foo")])
        commit(second, [make_da("bar")])
        xr.testing.assert_identical(
            xr.load_dataset(first.path), make_da("foo").to_dataset()
        )
        xr.testing.assert_identical(
            xr.load_dataset(second.path), make_da("bar").to_dataset()
        )

    def test_flush_single(self, target):
        da = make_da()
        commit(target, [da])
        xr.testing.assert_identical(xr.load_dataset(target.path), da.to_dataset())
        # Empty flush doesn't change anything
        target.flush()
        xr.testing.assert_identical(xr.load_dataset(target.path), da.to_dataset())

    def test_flush_twice(self, target):
        first, second = make_da(region="nh", val=1.0), make_da(region="sh", val=2.0)
        commit(target, [first])
        commit(target, [second])
        expected = xr.concat([first, second], dim="region").to_dataset()
        xr.testing.assert_identical(xr.load_dataset(target.path), expected)

    def test_flush_keeps_the_previous_commit_on_failure(self, target):
        commit(target, [make_da(region="nh", val=1.0)])
        committed = xr.load_dataset(target.path)

        with patch.object(xr.Dataset, "to_netcdf", side_effect=OSError("interrupted")):
            target.write(make_da(region="sh", val=2.0))
            with pytest.raises(OSError):
                target.flush()

        xr.testing.assert_identical(xr.load_dataset(target.path), committed)
        # The object does not claim more than is on disk
        xr.testing.assert_identical(target.to_dataset(), xr.load_dataset(target.path))

    def test_flush_can_be_retried_after_a_failure(self, target):
        da = make_da(region="nh", val=1.0)
        with patch.object(xr.Dataset, "to_netcdf", side_effect=OSError("interrupted")):
            target.write(da)
            with pytest.raises(OSError):
                target.flush()
        target.flush()  # the queue survived, so the retry really commits
        xr.testing.assert_identical(xr.load_dataset(target.path), da.to_dataset())

    def test_clean_removes_debris_from_an_interrupted_commit(self, target, tmpdir):
        commit(target, [make_da()])
        before = set(os.listdir(str(tmpdir)))
        # A commit killed between writing and moving into place leaves debris
        with patch("os.replace", side_effect=OSError("killed")):
            target.write(make_da(lons=(30.0, 40.0, 50.0)))
            with pytest.raises(OSError):
                target.flush()
        assert set(os.listdir(str(tmpdir))) > before
        target.clean()
        assert os.listdir(str(tmpdir)) == ["test.nc"]

    def test_parallel_commits_are_not_lost(self, target):
        target.enable_parallel()
        regions = ["nh", "sh", "eq", "pl"]
        parallel.parallel_processing(
            commit,
            [
                (target, [make_da(region=r, val=float(i))])
                for i, r in enumerate(regions)
            ],
            2,
        )
        written = xr.load_dataset(target.path)["foo"]
        np.testing.assert_array_equal(np.sort(written.region.values), sorted(regions))
        for i, region in enumerate(regions):
            np.testing.assert_array_equal(written.sel(region=region).values, float(i))

    def test_recovery_without_existing_file(self, target):
        target.enable_recovery()
        xr.testing.assert_identical(target.to_dataset(), xr.Dataset())

    def test_recovery_loads_existing_file(self, target):
        da = make_da()
        commit(target, [da])
        assert os.path.isfile(target.path)
        resumed = NetCDFTarget(
            path=target.path, format="netcdf"
        )  # as a resumed run would create
        resumed.enable_recovery()
        xr.testing.assert_identical(resumed.to_dataset(), da.to_dataset())

    def test_a_fresh_run_overwrites_an_existing_file(self, target):
        # Recovery is opt-in, so without it the previous results are replaced
        commit(target, [make_da(region="nh", val=1.0)])
        assert os.path.isfile(target.path)
        fresh = NetCDFTarget(path=target.path, format="netcdf")
        commit(fresh, [make_da(region="sh", val=2.0)])
        xr.testing.assert_identical(
            xr.load_dataset(fresh.path), make_da(region="sh", val=2.0).to_dataset()
        )

    def test_recovery_keeps_and_overrides_previous_results(self, target):
        a, b, c, d = make_da_blocks()
        a99 = a.copy(data=np.full_like(a.data, fill_value=99.0))
        commit(target, [a, c])
        resumed = NetCDFTarget(path=target.path, format="netcdf")
        resumed.enable_recovery()
        # nh is recomputed, and a block that was never reached is added
        commit(resumed, [a99])  # "recomputed"
        commit(resumed, [b, d])  # new block
        expected = xr.combine_by_coords([a99, b, c, d])
        xr.testing.assert_identical(resumed.to_dataset(), expected)
        xr.testing.assert_identical(xr.load_dataset(resumed.path), expected)

    def test_recovery_after_an_interrupted_commit(self, target):
        # The interrupted commit left the previous one readable, so a resumed run
        # picks it up instead of failing on a half-written file
        a, b, _, _ = make_da_blocks()
        commit(target, [a])
        with patch.object(xr.Dataset, "to_netcdf", side_effect=OSError("interrupted")):
            target.write(b)
            with pytest.raises(OSError):
                target.flush()

        resumed = NetCDFTarget(path=target.path)
        resumed.enable_recovery()
        xr.testing.assert_identical(resumed.to_dataset(), a.to_dataset())
