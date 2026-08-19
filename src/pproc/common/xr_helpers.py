# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import functools

import xarray as xr


class DatasetBuilder:
    """Transactional xarray.Dataset builder.

    Collects Datasets and DataArrays in a staging area. When the staged contents
    form a proper datacube, they can be committed. The final dataset is created
    by outer-joining all committed cubes.
    """

    def __init__(self, combine_kwargs=None):
        self._staged: list[xr.DataArray | xr.Dataset] = []
        self._committed: list[xr.Dataset] = []  # oldest first
        if combine_kwargs is None:
            combine_kwargs = {}
        combine_kwargs["join"] = "exact"  # always enforce full datacubes
        self._combine_kwargs = combine_kwargs

    def enable_parallel(self, manager):
        self._committed = manager.list(self._committed)

    def stage(self, ds):
        self._staged.append(ds)

    def commit(self):
        if not self._staged:
            return
        ds_staged = xr.combine_by_coords(self._staged, **self._combine_kwargs)
        self._committed.append(ds_staged.load())  # force computation of values
        self.reset_staged()

    def reset_staged(self):
        self._staged.clear()

    def undo_commit(self, *, stage=False):
        if not self._committed:
            raise RuntimeError("no commits to undo")
        removed = self._committed.pop()
        if stage:
            self.stage(removed)

    def squash_commits(self):
        self._committed[:] = [self.to_dataset()]

    def to_dataset(self) -> xr.Dataset:
        if not self._committed:
            return xr.Dataset()
        # Merge wth later commits overriding earlier ones
        return functools.reduce(
            lambda newer, older: newer.combine_first(older),
            reversed(self._committed),
        )
