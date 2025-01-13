import multiprocessing
import os
from typing import Any, Literal, Optional, Union

import eccodes
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


def remove_duplicate(path: str, message: eccodes.Message):
    """
    Removes existing message in file specified by path if it has mars keys
    matching those in message
    """
    if os.path.exists(path):
        mars_keys = ",".join(
            [f"{key}={value}" for key, value in message.items(namespace="mars")]
        )
        file_messages = [
            ",".join([f"{key}={value}" for key, value in msg.items(namespace="mars")])
            for msg in eccodes.FileReader(path)
        ]
        if mars_keys in file_messages:
            print(f"Deleting duplicate message {mars_keys} in file {path}")
            duplicate_index = file_messages.index(mars_keys)
            with open(f"{path}.temp", "wb") as temp_file:
                for msg_index, msg in enumerate(eccodes.FileReader(path)):
                    if msg_index == duplicate_index:
                        continue
                    msg.write_to(temp_file)
            os.rename(f"{path}.temp", path)


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


class NullTarget(Target):
    type_: Literal["null"] = Field("null", alias="type")

    def write(self, message):
        pass


class FileTarget(Target):
    type_: Literal["file"] = Field("file", alias="type")
    path: str
    _mode: str = "wb"
    _lock: FileLock = None
    _track_truncated: list[str] = []
    _overwrite_existing: bool = False

    @model_validator(mode="after")
    def create_lock(self) -> Any:
        if self._lock is None:
            self._lock = FileLock(self.path + ".lock", thread_local=False)

    @property
    def mode(self):
        if self.path not in self._track_truncated:
            self._track_truncated += [self.path]
            return self._mode
        return "ab"

    def enable_recovery(self):
        self._mode = "ab"
        self._overwrite_existing = True

    def enable_parallel(self):
        self._track_truncated = _shared_list()

    def write(self, message):
        with self._lock:
            if self._overwrite_existing:
                remove_duplicate(self.path, message)
            with open(self.path, self.mode) as file:
                message.write_to(file)


class FileSetTarget(Target):
    type_: Literal["fileset"] = Field("fileset", alias="type")
    path: str
    _mode: str = "wb"
    _file_locks: dict[str, FileLock] = {}
    _track_truncated: list[str] = []
    _overwrite_existing: bool = False

    def mode(self, path: str):
        if path not in self._track_truncated:
            self._track_truncated += [path]
            return self._mode
        return "ab"

    def enable_recovery(self):
        self._mode = "ab"
        self._overwrite_existing = True

    def enable_parallel(self):
        self._track_truncated = _shared_list()

    def write(self, message):
        path = self.path.format_map(message)
        with self._file_locks.get(path, FileLock(path + ".lock")):
            if self._overwrite_existing:
                remove_duplicate(path, message)
            with open(path, self.mode(path)) as file:
                message.write_to(file)


class FDBTarget(Target):
    type_: Literal["fdb"] = Field("fdb", alias="type")
    _fdb: Optional[pyfdb.FDB] = None

    @property
    def fdb(self):
        if self._fdb is None:
            self._fdb = pyfdb.FDB()
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
