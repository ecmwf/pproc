# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import functools
import multiprocessing
import os
from typing import Any, Literal, Optional, Union
from typing_extensions import Self
import yaml

import eccodes
import numpy as np
import pyfdb
import xarray as xr
from annotated_types import Annotated
from conflator import ConfigModel
from filelock import FileLock
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from pproc.config import utils

_manager = None


def _shared_list():
    global _manager
    if _manager is None:
        _manager = multiprocessing.Manager()
    return _manager.list()


class Target(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return

    def flush(self):
        return

    def write(self, message):
        raise NotImplementedError

    def enable_recovery(self):
        pass

    def enable_parallel(self):
        pass

    def clean(self):
        pass


class NullTarget(Target):
    type_: Literal["null"] = Field("null", alias="type")

    def write(self, message):
        pass


class FileTarget(Target):
    type_: Literal["file"] = Field("file", alias="type")
    path: str
    clean_lock: bool = True

    _opened_files: list[str] = []

    @property
    def lock(self) -> FileLock:
        return FileLock(self.path + ".lock", thread_local=False)

    @property
    def mode(self):
        if self.path not in self._opened_files:
            self._opened_files.append(self.path)
            return "wb"
        return "ab"

    def enable_recovery(self):
        raise NotImplementedError("Recovery is not implemented for FileTarget")

    def enable_parallel(self):
        self._opened_files = _shared_list()

    def write(self, message):
        with self.lock:
            with open(self.path, self.mode) as file:
                message.write_to(file)

    def clean(self):
        if self.clean_lock:
            if os.path.exists(self.lock.lock_file):
                os.remove(self.lock.lock_file)


class FileSetTarget(Target):
    type_: Literal["fileset"] = Field("fileset", alias="type")
    path: str
    clean_locks: bool = True

    _file_locks: dict[str, FileLock] = {}
    _opened_files: list[str] = []
    _lock_paths: list[str] = []

    def mode(self, path: str):
        if path not in self._opened_files:
            self._opened_files.append(path)
            return "wb"
        return "ab"

    def enable_recovery(self):
        raise NotImplementedError("Recovery is not implemented for FileSetTarget")

    def enable_parallel(self):
        self._opened_files = _shared_list()
        self._lock_paths = _shared_list()

    def write(self, message):
        path = self.path.format_map(message)
        with self._file_locks.setdefault(path, FileLock(path + ".lock")) as lock:
            if lock.lock_file not in self._lock_paths:
                self._lock_paths.append(lock.lock_file)
            with open(path, self.mode(path)) as file:
                message.write_to(file)

    def clean(self):
        if self.clean_locks:
            for lock_file in self._lock_paths:
                if os.path.exists(lock_file):
                    os.remove(lock_file)


class FDBTarget(Target):
    type_: Literal["fdb"] = Field("fdb", alias="type")
    config: Optional[str] = None
    _fdb: Optional[pyfdb.FDB] = None

    @property
    def fdb(self):
        if self._fdb is None:
            fdb_config = None
            if self.config is not None:
                with open(self.config, "r") as conf_file:
                    fdb_config = yaml.safe_load(conf_file)
            self._fdb = pyfdb.FDB(config=fdb_config)
        return self._fdb

    def write(self, message):
        message.items(namespace="mars")
        self.fdb.archive(message.get_buffer())

    def flush(self):
        self.fdb.flush()


class OverrideTargetWrapper(ConfigModel, Target):
    wrapped: Annotated[
        Union[NullTarget, FileTarget, FileSetTarget, FDBTarget],
        Field(default_factory=NullTarget, discriminator="type_"),
    ]
    overrides: Annotated[
        dict,
        BeforeValidator(utils.validate_overrides),
    ]

    @model_validator(mode="before")
    def validate_source(cls, data: Any) -> Any:
        if "wrapped" not in data:
            return {
                "overrides": data.pop("overrides", {}),
                "wrapped": data,
            }
        return data

    def __enter__(self):
        self.wrapped.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self.wrapped.__exit__(exc_type, exc_value, traceback)

    def flush(self):
        return self.wrapped.flush()

    def enable_recovery(self):
        return self.wrapped.enable_recovery()

    def enable_parallel(self):
        return self.wrapped.enable_parallel()

    def write(self, message):
        message.set(self.overrides)
        self.wrapped.write(message)

    def clean(self):
        return self.wrapped.clean()


class CombineKwargs(BaseModel):
    """Arguments for :func:`xarray.combine_by_coords`."""

    model_config = ConfigDict(extra="forbid")  # join and fill value set by XarrayTarget

    compat: str = "no_conflicts"
    data_vars: str = "all"
    coords: str = "minimal"
    combine_attrs: str = "drop_conflicts"


class XarrayTarget(Target):
    """Transactional xarray.Dataset builder with (optional) flushing to disk.

    DataArrays are staged in the local process until flushed. Only complete
    datacubes can be flushed (to ensure safe merging).

    Formats:
    - "netcdf" (replaces file on every flush)
    - "memory" (only collects in memory, writing is left to entrypoint)
    """

    type_: Literal["xarray"] = Field("xarray", alias="type")
    path: str | None = None
    format: Literal["netcdf", "memory"] = "netcdf"
    clean_lock: bool = True
    combine_kwargs: CombineKwargs = CombineKwargs()

    _staged: list[xr.DataArray] = []
    _committed: list[xr.Dataset] = []  # oldest first

    @model_validator(mode="after")
    def validate_path(self):
        if self.format != "memory" and self.path is None:
            raise ValueError(f"path is required for format {self.format}")
        return self

    @property
    def lock(self) -> FileLock:
        return FileLock(self.path + ".lock", thread_local=False)

    @property
    def _tmp_path(self) -> str:
        return self.path + ".tmp"

    def flush(self):
        """Commit everything written/staged since the last flush."""
        if not self._staged:
            return
        ds_staged = xr.combine_by_coords(
            self._staged,
            join="exact",  # enforce full datacubes
            fill_value=np.nan,
            **self.combine_kwargs.model_dump(),
        ).load()  # force computation of values
        if self.format == "memory":
            self._committed.append(ds_staged)
        elif self.format == "netcdf":
            with self.lock:
                ds_out = self._consolidate([*self._committed, ds_staged])
                ds_out.to_netcdf(self._tmp_path)
                os.replace(self._tmp_path, self.path)
                # Replace collected parts with consolidated dataset while lock is aquired
                self._committed[:] = [ds_out]
        else:
            raise NotImplementedError(f"format {self.format}")
        self._staged.clear()

    def write(self, ds):
        self._staged.append(ds)

    def enable_recovery(self):
        if self.format == "memory":
            raise NotImplementedError("Recovery is not supported with format memory")
        if os.path.exists(self.path):
            self._committed.append(xr.load_dataset(self.path))

    def enable_parallel(self):
        self._committed = _shared_list()

    def clean(self):
        if self.path is None:
            return
        if os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        if self.clean_lock and os.path.exists(self.lock.lock_file):
            os.remove(self.lock.lock_file)

    @staticmethod
    def _consolidate(generations: list[xr.Dataset]) -> xr.Dataset:
        """Merge generations given oldest first, later ones overriding earlier ones."""
        if not generations:
            return xr.Dataset()
        return functools.reduce(
            lambda newer, older: newer.combine_first(older), reversed(generations)
        )

    def as_dataset(self) -> xr.Dataset:
        return self._consolidate(list(self._committed))
