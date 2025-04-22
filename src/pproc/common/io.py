from dataclasses import dataclass
from io import BytesIO
import os
from typing import Optional, Union, List, Dict, Any

import numpy as np
import xarray as xr
import yaml

import eccodes
import pyfdb
import mir

from pproc.config.targets import (
    FDBTarget,
    FileTarget,
    FileSetTarget,
    NullTarget,
    OverrideTargetWrapper,
)
from pproc.config.io import split_location


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
    if not cached_file:
        cached_file = (
            "_".join(
                [
                    f"{key}{len(value)}"
                    if isinstance(value, (list, range))
                    else f"{key}{value}"
                    for key, value in request.items()
                ]
            )
            + ".grb"
        )
    outfile = open(cached_file, "wb")
    for data in iter((lambda: fdb_reader.read(4096)), b""):
        outfile.write(data)
    if os.path.getsize(cached_file) == 0:
        raise RuntimeError(f"No data retrieved for request {request}")
    return mir.MultiDimensionalGribFileInput(cached_file, 2)


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
        if mir_options.get("vod2uv", False):
            mir_options = mir_options.copy()
            mir_options["vod2uv"] = "1"
            fdb_reader = mir_wind_input(fdb_reader, request)
        job = mir.Job(**mir_options)
        stream = BytesIO()
        job.execute(fdb_reader, stream)
        stream.seek(0)
        fdb_reader = stream
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


def write_grib(target, template, data, missing=-9999):

    message = template.copy()

    # replace missing values if any
    is_missing = np.isnan(data).any()
    if is_missing:
        data[np.isnan(data)] = missing
        message.set("missingValue", missing)
        message.set("bitmapPresent", 1)

    message.set_array("values", data)

    if is_missing:
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


class GribMetadata(eccodes.Message):
    def __init__(self, message: eccodes.Message, headers_only: bool = False):
        handle = eccodes.codes_clone(message._handle, headers_only=headers_only)
        super().__init__(handle)
        self.set_array("values", np.zeros(message.data.size))
        self.set("bitsPerValue", message.get("bitsPerValue"))

    def __getstate__(self) -> dict:
        ret = {"_handle": self.get_buffer()}
        return ret

    def __setstate__(self, state: dict):
        state["_handle"] = eccodes.MemoryReader(state["_handle"])._next_handle()
        self.__dict__.update(state)
