# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import multiprocessing
import os
from typing import Any, Literal, Optional, Union
from typing_extensions import Self
import yaml

import pyfdb
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
    _lock: FileLock = None

    @model_validator(mode="after")
    def create_lock(self) -> Self:
        if self._lock is None:
            self._lock = FileLock(self.path + ".lock", thread_local=False)
        return self

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
        with self._lock:
            with open(self.path, self.mode) as file:
                message.write_to(file)

    def clean(self):
        if self.clean_lock:
            if os.path.exists(self._lock.lock_file):
                os.remove(self._lock.lock_file)


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
