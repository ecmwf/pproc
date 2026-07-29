# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import shutil
from io import BytesIO
from typing import Optional
import logging

import mir
from earthkit.data import FieldList, settings
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from earthkit.data.sources import from_source
from earthkit.data.sources.file import FileSource
from earthkit.data.sources.stream import StreamSource
from meters import ResourceMeter

# Set cache policy to "temporary" to avoid "database is locked" errors when
# for wind when executing across multiple workers
settings.set("cache-policy", "temporary")

logger = logging.getLogger(__name__)


def mir_job(
    input: mir.MultiDimensionalGribFileInput,  # type: ignore[ty:unresolved-attribute]
    mir_options: dict,
    cache: Optional[str] = None,
) -> FieldList:
    # Convert vod2uv bool to string int
    if "vod2uv" in mir_options:
        mir_options["vod2uv"] = "1" if mir_options["vod2uv"] else "0"
    job = mir.Job(**mir_options)  # type: ignore[ty:unresolved-attribute]
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    if cache is None:
        return StreamSource(stream, read_all=True).mutate().to_fieldlist()

    with open(cache, "wb") as o, stream as i:
        shutil.copyfileobj(i, o)
    return FileSource(cache).mutate()


def fdb_retrieve(request: dict, stream: bool = False, **kwargs) -> FieldList:
    mir_options = request.pop("interpolate", None)
    if mir_options is None:
        return from_source("fdb", request, stream=stream, read_all=True, **kwargs)  # type: ignore[ty:invalid-return-type]

    if mir_options.get("vod2uv", False):
        stream = False

    reader: FieldList = from_source(
        "fdb", request, stream=stream, read_all=False, **kwargs
    )  # type: ignore[ty:invalid-assignment]
    if stream:
        return mir_job(reader._source._stream, mir_options)  # type: ignore[ty:unresolved-attribute]

    if mir_options.get("vod2uv", False):
        if len(request["param"]) != 2:
            raise ValueError("Wind vod2uv requires two parameters")
        inp = mir.MultiDimensionalGribFileInput(reader.path, 2)  # type: ignore[ty:unresolved-attribute]
    else:
        inp = mir.GribFileInput(reader.path)  # type: ignore[ty:unresolved-attribute]
    return mir_job(inp, mir_options)


def mars_retrieve(request: dict, **kwargs) -> FieldList:
    mir_options = request.pop("interpolate", None)
    cache = request.pop("cache", None)
    cache_path = None if cache is None else cache.format_map(request)
    ds: FieldList = from_source("mars", request, **kwargs)  # type: ignore[ty:invalid-assignment]
    if mir_options is None:
        return ds

    if mir_options.get("vod2uv", False):
        if len(request["param"]) != 2:
            raise ValueError("Wind vod2uv requires two parameters")
        inp = mir.MultiDimensionalGribFileInput(ds.path, 2)  # type: ignore[ty:unresolved-attribute]
    else:
        inp = mir.GribFileInput(ds.path)  # type: ignore[ty:unresolved-attribute]
    return mir_job(inp, mir_options, cache_path)


def _transform_steps(steps, step_type: type = str):
    if isinstance(steps, (int, str)):
        steps = [steps]
    return list(map(step_type, steps))


def _transform_request(request: dict, step_type: type = str):
    try:
        paramId = int(request["param"])
        del request["param"]
        request["paramId"] = paramId
    except ValueError:
        pass
    if request.get("date", None) is not None:
        request["date"] = int(request["date"])
    if request.get("time", None) is not None:
        time = int(request["time"])
        request["time"] = time if time % 100 == 0 else time * 100
    if request.get("step", None) is not None:
        request["step"] = _transform_steps(request["step"], step_type)
    return request


def file_retrieve(name: str, request: dict, **kwargs) -> FieldList:
    mir_options = request.pop("interpolate", None)
    if mir_options is not None:
        raise NotImplementedError()
    file_ds: FieldList = from_source(name, **kwargs)  # type: ignore[ty:invalid-assignment]
    if len(request) > 0:
        treq = _transform_request(request)
        ds = file_ds.sel(treq)
        if len(ds) == 0:
            try:
                treq = _transform_request(request, int)
                ds = file_ds.sel(treq)
            except ValueError:
                pass
        return ds
    return file_ds


def retrieve(
    sources: list[dict],
    requests: list[dict],
    dtype: Optional[str] = None,
) -> FieldList:
    ds: FieldList = from_source("empty")  # type: ignore
    for request in requests:
        with ResourceMeter(f"Retrieve {request}"):
            for source in sources:
                if isinstance(source, str):
                    source = {"name": source}
                name = source.pop("name")
                try:
                    logger.debug(f"Trying source {name}")
                    if name == "fdb":
                        source_ds = fdb_retrieve(request=request, **source)
                    elif name == "mars":
                        source_ds = mars_retrieve(request=request, **source)
                    elif name in ["file", "file-pattern"]:
                        source_ds = file_retrieve(
                            name, request=request, **source
                        ).order_by("paramId")
                    else:
                        raise NotImplementedError(f"Source {source} not supported.")
                    assert (
                        len(source_ds) > 0
                    ), f"No data retrieved from {source} for request {request}"
                    break
                except AssertionError:
                    logger.info(
                        f"No data retrieved from source {source} for request {request}"
                    )
                    continue
            if len(source_ds) == 0:
                raise ValueError(
                    f"No data retrieved from sources {sources} for request {request}"
                )
        ds += FieldList.from_array(
            source_ds.to_array(flatten=True, dtype=dtype),
            [
                StandAloneGribMetadata(metadata._handle)
                for metadata in source_ds.metadata()
            ],
        )
    return ds


def write(data: FieldList, name: str, **kwargs):
    if name == "null":
        return

    with ResourceMeter(f"Write {data.ls(namespace='mars')} to {name}"):
        data.to_target(target=name, **kwargs)
