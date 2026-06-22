# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Literal, Optional

from pydantic import BaseModel


class NullTarget(BaseModel):
    name: Literal["null"] = "null"


class FileTarget(BaseModel):
    name: Literal["file"] = "file"
    file: str
    append: bool = False


class FilePatternTarget(BaseModel):
    name: Literal["file-pattern"] = "file-pattern"
    file: str
    append: bool = False


class FDBTarget(BaseModel):
    name: Literal["fdb"] = "fdb"
    config: Optional[str] = None


class ZarrTarget(BaseModel):
    name: Literal["zarr"] = "zarr"
    earthkit_to_xarray_kwargs: Optional[dict] = None
    xarray_to_zarr_kwargs: Optional[dict] = None
