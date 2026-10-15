# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools

import xarray as xr


class DatasetBuilder:
    """Transactional xarray.Dataset builder.

    Collects Datasets and DataArrays in a staging area. When the staged contents
    form a proper datacube, they can be committed. The final dataset is created
    by outer-joining all committed cubes in the order they were committed in.

    While commits can be submitted from concurrent units of execution after
    setting up a shared commit list with :py:meth:`DatasetBuilder.enable_parallel`,
    the result of :py:meth:`DatasetBuilder.to_dataset` is dependent on the
    commit order. Race conditions arise when multiple commits set values over a
    common range of coordinates.

    Parameters
    ----------
    combine_kwargs
        Arguments for the :py:func:`xarray.combine_by_coords` call that merges
        the objects in the staging area. The `join` and `fill_value` parameters
        cannot be overridden.
    """

    def __init__(self, combine_kwargs=None):
        self._staged: list[xr.DataArray | xr.Dataset] = []
        self._committed: list[xr.Dataset] = []  # oldest first
        if combine_kwargs is None:
            combine_kwargs = {}
        # Always enforce full datacubes
        combine_kwargs["join"] = "exact"
        combine_kwargs["fill_value"] = None
        self._combine_kwargs = combine_kwargs

    def enable_parallel(self, manager):
        """Use the manager to replace the commit list with a shared one."""
        self._committed = manager.list(self._committed)

    def stage(self, ds):
        """Add an object to the staging area."""
        if isinstance(ds, xr.DataArray) and ds.name is None:
            raise ValueError("DataArrays must have a name")
        self._staged.append(ds)
        return self

    def commit(self):
        """Merge the staging area and commit the resulting dataset."""
        if not self._staged:
            return
        ds_staged = xr.combine_by_coords(self._staged, **self._combine_kwargs)
        self._committed.append(ds_staged.load())  # force computation of values
        self.reset_staged()
        return self

    def reset_staged(self):
        """Clear the staging area."""
        self._staged.clear()
        return self

    def undo_commit(self, *, stage=False):
        """Undo the last commit, optionally re-staging the committed dataset."""
        if not self._committed:
            raise RuntimeError("no commits to undo")
        removed = self._committed.pop()
        if stage:
            self.stage(removed)
        return self

    def squash_commits(self):
        """Merge all commits and re-initialise the commit list with the result."""
        self._committed[:] = [self.to_dataset()]
        return self

    def to_dataset(self) -> xr.Dataset:
        """Merge all commits and return the result."""
        if not self._committed:
            return xr.Dataset()
        # Merge with later commits overriding earlier ones
        return functools.reduce(
            lambda older, newer: newer.combine_first(older),
            self._committed,
        )
