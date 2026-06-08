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


def mir_job(
    input: mir.MultiDimensionalGribFileInput,
    mir_options: dict,
    cache: Optional[str] = None,  # type: ignore[ty:unresolved-attribute]
) -> FieldList:
    job = mir.Job(**mir_options)  # type: ignore[ty:unresolved-attribute]
    stream = BytesIO()
    job.execute(input, stream)
    stream.seek(0)
    if cache is None:
        return StreamSource(stream, read_all=True).mutate()

    with open(cache, "wb") as o, stream as i:
        shutil.copyfileobj(i, o)
    return FileSource(cache).mutate()


def fdb_retrieve(request: dict, *, stream: bool = True) -> FieldList:
    mir_options = request.pop("interpolate", None)
    if mir_options is None:
        return from_source("fdb", request, read_all=True, stream=stream)  # type: ignore[ty:invalid-return-type]

    if mir_options.get("vod2uv", "0") == "1":
        stream = False

    reader: FieldList = from_source("fdb", request, stream=stream)  # type: ignore[ty:invalid-assignment]
    if stream:
        return mir_job(reader._source._stream, mir_options)  # type: ignore[ty:unresolved-attribute]

    if mir_options.get("vod2uv", "0") == "1":
        if len(request["param"]) != 2:
            raise ValueError("Wind vod2uv requires two parameters")
        inp = mir.MultiDimensionalGribFileInput(reader.path, 2)  # type: ignore[ty:unresolved-attribute]
    else:
        inp = mir.GribFileInput(reader.path)  # type: ignore[ty:unresolved-attribute]
    return mir_job(inp, mir_options)


def mars_retrieve(request: dict) -> FieldList:
    mir_options = request.pop("interpolate", None)
    cache = request.pop("cache", None)
    cache_path = None if cache is None else cache.format_map(request)
    ds: FieldList = from_source("mars", request)  # type: ignore[ty:invalid-assignment]
    if mir_options is None:
        return ds

    if mir_options.get("vod2uv", "0") == "1":
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


def file_retrieve(path: str, request: dict) -> FieldList:
    mir_options = request.pop("interpolate", None)
    if mir_options is not None:
        raise NotImplementedError()
    location = path.format_map(request)
    file_ds: FieldList = from_source("file", location)  # type: ignore[ty:invalid-assignment]
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


def retrieve_multi_sources(requests: list[dict], **kwargs) -> FieldList:
    ret = None
    for req in requests:
        try:
            ret = retrieve_single_source(req, **kwargs)
            break
        except AssertionError:
            continue
    assert ret is not None, f"No data retrieved from requests: {requests}"
    return ret


def retrieve_single_source(request: dict, **kwargs) -> FieldList:
    req = request.copy()
    source = req.pop("source")
    if source == "fdb":
        ret_sources = fdb_retrieve(req, **kwargs)
    elif source == "mars":
        ret_sources = mars_retrieve(req)
    elif source == "fileset":
        path = req.pop("location")
        ret_sources = file_retrieve(path, req).order_by("paramId")
    else:
        raise NotImplementedError(f"Source {source} not supported.")
    assert len(ret_sources) > 0, f"No data retrieved from {source} for request {req}"
    return ret_sources


def retrieve(request: dict | list[dict], **kwargs):
    with ResourceMeter(f"retrieve {request}, {kwargs}"):
        if isinstance(request, dict):
            res = retrieve_single_source(request, **kwargs)
        else:
            res = retrieve_multi_sources(request, **kwargs)
        ret = FieldList.from_array(
            res.values,
            [StandAloneGribMetadata(metadata._handle) for metadata in res.metadata()],
        )
        return ret


def write(data: FieldList, target: dict) -> dict:
    if target["name"] == "null":
        return target
    data.to_target(**target)
    return target
