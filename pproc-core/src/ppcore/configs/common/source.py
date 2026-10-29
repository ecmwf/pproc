# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Literal, Annotated, Union
from pydantic import Field

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel


class FDBSource(PProcBaseModel):
    name: Literal["fdb"] = "fdb"
    config: Optional[str] = None
    stream: bool = True


class MARSSource(PProcBaseModel):
    name: Literal["mars"] = "mars"


class FileSource(PProcBaseModel):
    name: Literal["file"] = "file"
    path: str


class FilePatternSource(PProcBaseModel):
    name: Literal["file-pattern"] = "file-pattern"
    pattern: str
    hive_partitioning: bool = True


Source = Annotated[
    Union[FileSource, FilePatternSource, FDBSource, MARSSource],
    Field(discriminator="name"),
]
