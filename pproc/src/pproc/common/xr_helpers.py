# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools
from collections.abc import Mapping, MutableSequence
from multiprocessing.managers import SyncManager
from typing import Any, Self, cast

import xarray as xr


def _combine_first(older: xr.Dataset, newer: xr.Dataset) -> xr.Dataset:
    merged = newer.combine_first(older)
    # combine_first only keeps the attributes of newer. Merge them per key instead,
    # in line with the values, so what newer does not specify is kept from older.
    merged.attrs = {**older.attrs, **newer.attrs}
    for name, var in merged.variables.items():  # data variables and coordinates
        var.attrs = {
            **(older.variables[name].attrs if name in older.variables else {}),
            **(newer.variables[name].attrs if name in newer.variables else {}),
        }
    return merged


class DatasetBuilder:
    """Transactional xarray.Dataset builder.

    Collects Datasets and DataArrays in a staging area (local per process). When
    the staged contents form a proper datacube, they can be committed. The final
    dataset is created by outer-joining all committed cubes in the order they
    were committed in.

    Parameters
    ----------
    combine_kwargs
        Arguments for the :py:func:`xarray.combine_by_coords` call that merges
        the objects in the staging area. The `join` and `fill_value` parameters
        cannot be overridden.

    Notes
    -----
    Commits are merged such that values and attributes from later commits take
    precedence over those from older commits. For values, NaNs are treated as
    missing values and later NaNs do not override earlier non-NaN values. A
    missing attribute in a later commit will not remove an attribute from an
    earlier commit. Variable encodings are always dropped. Race conditions
    arise when multiple concurrent commits set values over a common range of
    coordinates.

    Because commits that do not cover identical coordinates are outer-joined and
    gaps are filled with NaN, integer variables can be promoted to float. Only
    single commits and commits over identical coordinates will keep their dtype.
    """

    def __init__(self, combine_kwargs: Mapping[str, Any] | None = None) -> None:
        self._staged: list[xr.DataArray | xr.Dataset] = []
        # oldest first, a manager list proxy after enable_parallel
        self._committed: MutableSequence[xr.Dataset] = []
        # Copy to leave the caller's dict untouched
        kwargs = {} if combine_kwargs is None else dict(combine_kwargs)
        # Always enforce full datacubes
        kwargs["join"] = "exact"
        kwargs["fill_value"] = None
        self._combine_kwargs = kwargs

    def enable_parallel(self, manager: SyncManager) -> None:
        """Use the manager to replace the commit list with a shared one."""
        self._committed = manager.list(self._committed)

    def stage(self, ds: xr.DataArray | xr.Dataset) -> Self:
        """Add an object to the staging area."""
        if isinstance(ds, xr.DataArray) and ds.name is None:
            raise ValueError("DataArrays must have a name")
        self._staged.append(ds)
        return self

    def commit(self) -> Self:
        """Merge the staging area and commit the resulting dataset."""
        if not self._staged:
            return self
        # Always a Dataset: combine_by_coords only returns a DataArray when given
        # unnamed ones, which stage() rejects. Named DataArrays become variables.
        ds_staged = cast(
            xr.Dataset, xr.combine_by_coords(self._staged, **self._combine_kwargs)
        )
        self._committed.append(ds_staged.load())  # force computation of values
        self.reset_staged()
        return self

    def reset_staged(self) -> Self:
        """Clear the staging area."""
        self._staged.clear()
        return self

    def undo_commit(self, *, stage: bool = False) -> Self:
        """Undo the last commit, optionally re-staging the committed dataset."""
        if not self._committed:
            raise RuntimeError("no commits to undo")
        removed = self._committed.pop()
        if stage:
            self.stage(removed)
        return self

    def squash_commits(self) -> Self:
        """Merge all commits and re-initialise the commit list with the result."""
        self._committed[:] = [self.to_dataset()]
        return self

    def to_dataset(self) -> xr.Dataset:
        """Merge all commits and return the result."""
        if not self._committed:
            return xr.Dataset()
        if len(self._committed) == 1:
            ds = self._committed[0].copy(deep=True)
        else:
            # Merge with later commits overriding earlier ones
            ds = functools.reduce(_combine_first, self._committed)
        # Drop encoding
        for var in ds.variables.values():
            var.encoding = {}
        ds.encoding = {}
        return ds
