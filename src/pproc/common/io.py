# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from dataclasses import dataclass
from io import BytesIO
import os
from typing import Optional, Union, List, Dict, Any
from typing_extensions import Self

import numpy as np
import xarray as xr
import yaml

import eccodes
import pyfdb
import mir
import earthkit.data
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from earthkit.data.readers.grib.codes import GribCodesHandle

from pproc.config.targets import (
    FDBTarget,
    FileTarget,
    FileSetTarget,
    NullTarget,
    OverrideTargetWrapper,
)
from pproc.config.io import split_location
from pproc.common.grib_helpers import construct_message


@dataclass
class GRIBFields:
    template: Union[None, eccodes.Message]
    dims: List
    data: Dict

    def to_xarray(self):
        coords = {}
        dim_sizes = {}
        for dim in self.dims:
            set_coords = set()
            for key in self.data.keys():
                key_dict = eval(key)
                set_coords.add(key_dict[dim])
            dim_sizes[dim] = len(set_coords)
            coords[dim] = sorted(list(set_coords))

        # add values dimensions, no coords
        ndata = self.template.get_size("values")
        dim_sizes["data"] = ndata
        dims = self.dims.copy()
        dims.append("data")

        data_np = np.empty(tuple(dim_sizes.values()))
        for key, value in self.data.items():
            key_dict = eval(key)
            indexes = [coords[dim].index(coord) for dim, coord in key_dict.items()]
            data_np[tuple(indexes)] = value
        da = xr.DataArray(
            data_np,
            name=self.template["shortName"],
            coords=coords,
            dims=dims,
            attrs={"grib_template": self.template},
        )

        return da


def extract(keys, message):
    """Extract the given keys values from a message

    Parameters
    ----------
    keys: string
    message: eccodes.Message
        Message

    Returns
    -------
    tuple
        Values of the extracted keys"""
    res = {}
    for key in keys:
        if isinstance(key, str):
            res[key] = message.get(key)
        else:
            raise ValueError(
                f"Key format {type(key)} for {key} not supported, on support strings"
            )
            # res.append(key(message))
    return str(res)


def missing_to_nan(message, data=None):
    """Replace missing values by NaN

    Parameters
    ----------
    message: eccodes.Message
        GRIB message
    data: numpy array, optional
        Values array to use instead of the 'values' field. Modified in place.

    Returns
    -------
    numpy array
        Data with NaN for missing values
    """
    if data is None:
        data = message.get_array("values")
    if message.get("bitmapPresent"):
        missing = message.get("missingValue")
        data[data == missing] = np.nan
    return data


def nan_to_missing(message, data, missing=None):
    """Replace NaN by missing values

    Parameters
    ----------
    message: eccodes.Message
        GRIB message, keys will be set according to the missing value
    data: numpy array
        Data array (not added to the message), modified in place
    missing: float, optional
        Value to use instead of the 'missingValue' field

    Returns
    -------
    numpy array
        Data with NaN replaced by `missing`
    """
    if missing is None:
        missing = message.get("missingValue")
    missing_mask = np.isnan(data)
    if np.any(missing_mask):
        data[missing_mask] = missing
        message.set("missingValue", missing)
        message.set("bitmapPresent", 1)
    return data


def read_grib_messages(messages, dims=()):
    """Read all input messages with coordinates grouping

    Parameters
    ----------
    messages: grib messages
    dims: tuple of strings

    Returns
    -------
    GRIBFields
        GRIBFields object, containing the messages values, the dimensions and a grib template
    """
    fields = None
    for message in messages:
        if fields is None:
            fields = GRIBFields(message, dims, {})
        key = extract(dims, message)
        fields.data[key] = missing_to_nan(message)
    return fields


def mir_wind_input(fdb_reader, request, cached_file=None):
    list_keys = [key for key in request if isinstance(request[key], (list, range))]
    if not cached_file:
        cached_file = (
            "_".join(
                [
                    (f"{key}{len(value)}" if key in list_keys else f"{key}{value}")
                    for key, value in request.items()
                ]
            )
            + ".grb"
        )
    fields = earthkit.data.from_source("stream", fdb_reader, read_all=True)
    # Mir expects vo and d fields to be paired, so param must be last in order
    fields = fields.order_by([x for x in list_keys if x != "param"])
    fields.to_target("file", cached_file)
    if os.path.getsize(cached_file) == 0:
        raise RuntimeError(f"No data retrieved for request {request}")
    return mir.MultiDimensionalGribFileInput(cached_file, 2), cached_file


def fdb_retrieve(fdb, request, mir_options=None):
    """Retrieve grib messages from FDB from request and returns fdb reader object
    If mir options specified, also performs interpolation

    Parameters
    ----------
    messages: grib messages
    dims: tuple of strings
    mir_options: dict

    Returns
    -------
    FDB Reader
        FDB Reader object, containing the messages requested
    """
    fdb_reader = fdb.retrieve(request)
    if mir_options:
        cached_file = None
        if mir_options.get("vod2uv", False):
            mir_options = mir_options.copy()
            mir_options["vod2uv"] = "1"
            fdb_reader, cached_file = mir_wind_input(fdb_reader, request)
        job = mir.Job(**mir_options)
        stream = BytesIO()
        job.execute(fdb_reader, stream)
        stream.seek(0)
        fdb_reader = stream
        if cached_file:
            os.remove(cached_file)
    return fdb_reader


def fdb_read(fdb, request, mir_options=None):
    """Load grib messages from FDB from request and returns Xarray DataArray
    If mir options specified, also performs interpolation

    Parameters
    ----------
    messages: grib messages
    dims: tuple of strings
    mir_options: dict

    Returns
    -------
    Xarray DataArray
        Xarray DataArray object, containing the data and the associated coordinates
        together with a grib template in the attributes
    """

    fdb_reader = fdb_retrieve(fdb, request, mir_options)
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    if not eccodes_reader.peek():
        raise RuntimeError(f"No data retrieved for request {request}")
    fields_dims = [key for key in request if isinstance(request[key], (list, range))]
    fields = read_grib_messages(eccodes_reader, fields_dims)
    if fields is None:
        raise Exception(
            f"Could not perform the following retrieve:\n{yaml.dump(request)}"
        )

    return fields.to_xarray()


def fdb_read_to_file(fdb, request, file_out, mir_options=None, mode="wb"):
    """Load grib messages from FDB from request and writes to temporary file

    Parameters
    ----------
    messages: grib messages
    dims: tuple of strings

    Returns
    -------
    Xarray DataArray
        Xarray DataArray object, containing the data and the associated coordinates
        together with a grib template in the attributes
    """
    fdb_reader = fdb_retrieve(fdb, request, mir_options)
    outfile = open(file_out, mode)
    for data in iter((lambda: fdb_reader.read(4096)), b""):
        outfile.write(data)
    if os.path.getsize(file_out) == 0:
        raise RuntimeError(f"No data retrieved for request {request}")


def fdb_write_ufunc(data, coords, fdb, template):

    message = (
        template.copy()
    )  # are we always copying the full message with the data values?

    for key, value in coords:
        if len(value) > 1:
            raise Exception(
                "Can't have more than one coordinate in the parallel write function"
            )
        message.set(key, value.values[0])

    # Set GRIB data and write to FDB
    message.set_array("values", data)
    nan_to_missing(message, data)
    fdb.write(message)


def iterate_xarray(func, args, data_array, core_dims="data"):
    if list(data_array.dims) == list(core_dims):
        return func(data_array, *args)
    else:
        for sub_array in data_array:
            return iterate_xarray(func, args, sub_array, core_dims)


def write_message(target, template, data_array):
    message = (
        template.copy()
    )  # are we always copying the full message with the data values?
    for key, value in data_array.coords:
        if len(value) > 1:
            raise Exception(
                "Can't have more than one coordinate in the parallel write function"
            )
        message.set(key, value.values)
    # Set GRIB data and write to FDB
    message.set_array("values", data_array.values)
    nan_to_missing(message, data_array.values)
    target.write(message)


def write(target, template, attributes, data_array):

    message = template.copy()
    for key, value in attributes.items():
        message[key] = value

    iterate_xarray(write_message, (target, template), data_array, "data")
    # xr.apply_ufunc(fdb_write_ufunc, data_array, data_array.coords,
    #                input_core_dims=[['data'], []],
    #                dask='parallelized',
    #                kwargs={'fdb': fdb, 'template': template})


def target_factory(target_option, out_file=None, fdb=None, overrides=None):
    if target_option == "fdb":
        target = FDBTarget(_fdb=fdb)
    elif target_option == "file":
        assert out_file is not None
        target = FileTarget(path=out_file)
    elif target_option == "fileset":
        assert out_file is not None
        target = FileSetTarget(path=out_file)
    elif target_option == "null" or target_option is None:
        return NullTarget()
    else:
        raise ValueError(
            f"Target {target_option} not supported, accepted values are 'fdb', 'file', 'fileset', and 'null'"
        )
    if overrides:
        return OverrideTargetWrapper(target=target, overrides=overrides)
    return target


def write_grib(target, template, data, metadata: dict, missing=None):
    out_keys = {}
    if hasattr(template, "extra"):
        out_keys.update(template.extra)
    out_keys.update(metadata)
    bits_per_value = out_keys.pop("bitsPerValue", template["bitsPerValue"])
    message = construct_message(template, out_keys)

    data = nan_to_missing(message, data, missing)
    message.set("bitsPerValue", bits_per_value)
    message.set_array("values", data)

    if np.isnan(data).any():
        n_missing1 = len(data[data == missing])
        n_missing2 = message.get("numberOfMissing")
        if n_missing1 != n_missing2:
            raise Exception(
                f"Number of missing values in the message not consistent, is {n_missing1} and should be {n_missing2}"
            )

    target.write(message)


class FDBNotOpenError(RuntimeError):
    pass


def fdb(create: bool = True) -> pyfdb.FDB:
    instance = getattr(fdb, "_instance", None)
    if instance is None:
        if not create:
            raise FDBNotOpenError("FDB not open")
        instance = pyfdb.FDB()
        fdb._instance = instance
    return instance


def target_from_location(
    loc: Optional[str], overrides: Optional[Dict[str, Any]] = None
):
    type_ = "null"
    ident = ""
    if loc is not None:
        type_, ident = split_location(loc, default="file")
    return target_factory(type_, out_file=ident, overrides=overrides)


# ---------------------------------------------------------------------------
# In-memory GRIB codec helpers
# ---------------------------------------------------------------------------
#
# These helpers expose a numpy <-> bytes codec for GRIB messages so callers
# (mir wrappers, in-memory pipelines, CLIs) can move data around without
# touching the filesystem. They sit alongside the existing ``write_grib``
# helper, which writes to a :class:`Target` rather than returning bytes.
#
# Pattern decision D-A: ``encode_grib`` is polymorphic on its ``template``
# argument (raw GRIB ``bytes`` *or* a :class:`GribMetadata` instance), and a
# paired ``decode_grib_with_metadata`` is exported alongside the dict-form
# ``decode_grib`` for downstream consumers that need the canonical metadata
# type.

# Metadata keys returned by :func:`decode_grib` for every message. Tuned to
# cover the GRIB section-1 / section-4 identification fields plus the keys
# needed to reproduce the wire-level packing on a round trip.
_DECODE_METADATA_KEYS: tuple = (
    "edition",
    "centre",
    "subCentre",
    "discipline",
    "parameterCategory",
    "parameterNumber",
    "paramId",
    "shortName",
    "name",
    "units",
    "typeOfLevel",
    "level",
    "gridType",
    "gridName",
    "N",
    "Ni",
    "Nj",
    "numberOfDataPoints",
    "numberOfValues",
    "packingType",
    "bitsPerValue",
    "bitmapPresent",
    "missingValue",
    "dataDate",
    "dataTime",
    "stepType",
    "stepRange",
)


def _collect_metadata(message) -> Dict[str, Any]:
    """Return a JSON-friendly metadata dict for a decoded GRIB message."""
    meta: Dict[str, Any] = {}
    for key in _DECODE_METADATA_KEYS:
        try:
            value = message.get(key)
        except (eccodes.KeyValueNotFoundError, KeyError, RuntimeError):
            continue
        if value is None:
            continue
        # eccodes returns the literal string "MISSING" for keys that exist
        # but are unset on the wire (e.g. ``Ni``/``Nj`` on reduced grids);
        # we keep it so the dict is faithful to the message.
        meta[key] = value
    return meta


def _read_messages(grib_bytes: bytes):
    """Yield every :class:`eccodes.Message` contained in ``grib_bytes``."""
    if not isinstance(grib_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"grib_bytes must be bytes-like, got {type(grib_bytes).__name__}"
        )
    reader = eccodes.MemoryReader(bytes(grib_bytes))
    for message in reader:
        yield message


def decode_grib(grib_bytes: bytes) -> "tuple[np.ndarray, dict]":
    """Decode a single GRIB message from ``grib_bytes``.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
        Float64 values array (with GRIB missing values replaced by NaN) and
        a metadata dict covering the standard identification + packing keys.
    """
    iterator = _read_messages(grib_bytes)
    try:
        message = next(iterator)
    except StopIteration:
        raise ValueError("grib_bytes does not contain any GRIB messages") from None

    values = message.get_array("values").astype(np.float64, copy=False)
    values = missing_to_nan(message, values)
    metadata = _collect_metadata(message)
    return values, metadata


def decode_grib_with_metadata(
    grib_bytes: bytes,
) -> "tuple[np.ndarray, GribMetadata]":
    """Decode a single GRIB message and return a :class:`GribMetadata`.

    Mirrors :func:`decode_grib` but exposes the canonical pproc metadata
    type, which preserves the eccodes handle for downstream consumers
    (``write_message``, ``fdb_write_ufunc``, ...). See Pattern decision D-A.
    """
    iterator = _read_messages(grib_bytes)
    try:
        message = next(iterator)
    except StopIteration:
        raise ValueError("grib_bytes does not contain any GRIB messages") from None

    values = message.get_array("values").astype(np.float64, copy=False)
    values = missing_to_nan(message, values)
    metadata = GribMetadata(message._handle)
    return values, metadata


def decode_multi_grib(grib_bytes: bytes, count: int) -> "list[tuple[np.ndarray, dict]]":
    """Decode ``count`` consecutive GRIB messages from ``grib_bytes``.

    The buffer must contain at least ``count`` messages; if it contains
    fewer, a :class:`ValueError` is raised so callers (in particular the
    ``mir-compute``-style multi-message inputs that bundle fields with a
    shell ``cat``) can fail loudly on malformed input.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")

    results: List[tuple] = []
    iterator = _read_messages(grib_bytes)
    for _ in range(count):
        try:
            message = next(iterator)
        except StopIteration:
            raise ValueError(
                f"grib_bytes contains only {len(results)} message(s), "
                f"but {count} were requested"
            ) from None
        values = message.get_array("values").astype(np.float64, copy=False)
        values = missing_to_nan(message, values)
        results.append((values, _collect_metadata(message)))
    return results


def encode_grib(
    values: np.ndarray,
    template: "Union[bytes, bytearray, memoryview, GribMetadata, eccodes.Message]",
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Encode ``values`` as a GRIB message by cloning ``template``.

    Parameters
    ----------
    values:
        Float64 numpy array. ``np.nan`` entries are translated to the
        message's missing value with ``bitmapPresent`` flipped on.
    template:
        Either a raw GRIB byte string (the wire bytes of a reference
        message) or a :class:`GribMetadata` instance. Pattern decision D-A
        keeps the call site flexible: pipelines that already hold a
        ``GribMetadata`` can pass it directly without re-serialising.
    metadata:
        Optional dict of GRIB key/value overrides applied to the cloned
        template before the values are written (using the same convention
        as :func:`write_grib` / :func:`construct_message`).
    """
    if isinstance(template, (bytes, bytearray, memoryview)):
        try:
            base = next(_read_messages(template))
        except StopIteration:
            raise ValueError("template bytes do not contain a GRIB message") from None
    elif isinstance(template, eccodes.Message):
        base = template
    else:
        raise TypeError(
            "template must be bytes-like or a GribMetadata/eccodes.Message, "
            f"got {type(template).__name__}"
        )

    if metadata:
        # Apply caller-supplied overrides via the existing pproc convention
        # (handles edition switches, MISSING sentinels, array-valued keys,
        # and the operational ``packingType`` default for edition 2).
        out_keys: Dict[str, Any] = {}
        if hasattr(base, "extra"):
            out_keys.update(base.extra)
        out_keys.update(metadata)
        bits_per_value = out_keys.pop("bitsPerValue", base.get("bitsPerValue"))
        message = construct_message(base, out_keys)
        message.set("bitsPerValue", bits_per_value)
    else:
        # Plain clone path. Avoids ``construct_message``'s edition-2
        # ``packingType=grid_ccsds`` default, which would re-quantise a
        # losslessly-packed (e.g. ``grid_ieee``) template and break
        # bit-identical round-tripping.
        message = base.copy()

    data = np.asarray(values, dtype=np.float64).copy()
    data = nan_to_missing(message, data)
    message.set_array("values", data)

    return message.get_buffer()


class GribMetadata(eccodes.Message):
    def __init__(self, handle, headers_only: bool = False):
        new_handle = eccodes.codes_clone(handle, headers_only=headers_only)
        self.extra = {"bitsPerValue": eccodes.codes_get(handle, "bitsPerValue", int)}
        super().__init__(new_handle)

    def __getstate__(self) -> dict:
        ret = {"_handle": self.get_buffer(), "extra": self.extra}
        return ret

    def __setstate__(self, state: dict):
        state["_handle"] = eccodes.MemoryReader(state["_handle"])._next_handle()
        self.__dict__.update(state)

    def set(self, *args, check_values: bool = True):
        super().set(*args, check_values=check_values)
        if isinstance(args[0], dict):
            for key in self.extra.keys():
                if key in args[0]:
                    self.extra[key] = args[0][key]
        elif args[0] in self.extra:
            self.extra[args[0]] = args[1]

    def copy(self) -> Self:
        """Create a copy of the current message"""
        clone = self.__class__(eccodes.codes_clone(self._handle))
        clone.extra = self.extra.copy()
        return clone

    def to_ekmetadata(self) -> StandAloneGribMetadata:
        return StandAloneGribMetadata(
            GribCodesHandle(eccodes.codes_clone(self._handle), None, None)
        )
