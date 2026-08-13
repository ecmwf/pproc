# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0


from typing import Literal, Optional, Annotated, Union
from pydantic import Field

from earthkit.workflows.plugins.pproc.utils.pydantic_utils import PProcBaseModel


class NullTarget(PProcBaseModel):
    name: Literal["null"] = "null"


class FileTarget(PProcBaseModel):
    name: Literal["file"] = "file"
    file: str
    append: bool = False


class FilePatternTarget(PProcBaseModel):
    name: Literal["file-pattern"] = "file-pattern"
    file: str
    append: bool = False


class FDBTarget(PProcBaseModel):
    name: Literal["fdb"] = "fdb"
    config: Optional[str] = None


class ZarrTarget(PProcBaseModel):
    name: Literal["zarr"] = "zarr"
    earthkit_to_xarray_kwargs: Optional[dict[str, str]] = None
    xarray_to_zarr_kwargs: Optional[dict[str, str]] = None


Target = Annotated[
    Union[FileTarget, FilePatternTarget, FDBTarget, NullTarget, ZarrTarget],
    Field(discriminator="name"),
]
