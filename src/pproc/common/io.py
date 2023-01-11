from io import BytesIO
import yaml
import itertools

import numpy as np
import xarray as xr
from dataclasses import dataclass, field
from typing import Union, Any, List, Dict

import eccodes
import pyfdb
import mir

@dataclass
class GRIBFields:
    template: Union[None, eccodes.Message]
    dims: List
    data: Dict

    def to_xarray(self):
        coords = {}
        dim_sizes = {}
        for i, dim in enumerate(self.dims):
            coords[dim] = []
            for key_tuple in self.data.keys():
                coords[dim].append(key_tuple[i])
            dim_sizes[dim] = len(coords[dim])
        
        # add values dimensions, no coords
        ndata = self.template.get_size('values')
        dim_sizes['data'] = ndata
        dims = self.dims.copy()
        dims.append('data')
        
        data_np = np.empty(tuple(dim_sizes.values()))
        for key_tuple, value in self.data.items():
            indexes = [coords[self.dims[i]].index(key) for i, key in enumerate(key_tuple)]
            data_np[indexes] = value
        da = xr.DataArray(data_np, coords=coords, dims=dims, attrs={'grib_template': self.template})

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
    res = []
    for key in keys:
        if isinstance(key, str):
            res.append(message.get(key))
        else:
            raise ValueError(f'Key format {type(key)} for {key} not supported, on support strings')
            # res.append(key(message))
    return tuple(res)


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
        data = message.get_array('values')
    if message.get('bitmapPresent'):
        missing = message.get('missingValue')
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
        missing = message.get('missingValue')
    missing_mask = np.isnan(data)
    if np.any(missing_mask):
        data[missing_mask] = missing
        message.set('missingValue', missing)
        message.set('bitmapPresent', 1)
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
        job = mir.Job(mir_options)
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
    fields_dims = [key for key in request if isinstance(request[key], (list, range))]
    fields = read_grib_messages(eccodes_reader, fields_dims)
    if fields is None:
        raise Exception(f"Could not perform the following retrieve:\n{yaml.dump(request)}")

    return fields.to_xarray()


def fdb_read_to_file(fdb, request, file_out, mir_options=None, mode='wb'):
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


def fdb_write_ufunc(data, coords, fdb, template):

    message = template.copy()  # are we always copying the full message with the data values?

    for key, value in coords:
        if len(value) > 1:
            raise Exception("Can't have more than one coordinate in the parallel write function")
        message.set(key, value.values[0])
    
    # Set GRIB data and write to FDB
    message.set_array("values", data)
    nan_to_missing(message, data)
    fdb.write(message)


def iterate_xarray(func, args, data_array, core_dims='data'):
    if list(data_array.dims) == list(core_dims):
        return func(data_array, *args)
    else:
        for sub_array in data_array:
            return iterate_xarray(func, args, sub_array, core_dims)


def write_message(target, template, data_array): 
    message = template.copy()  # are we always copying the full message with the data values?
    for key, value in data_array.coords:
        if len(value) > 1:
            raise Exception("Can't have more than one coordinate in the parallel write function")
        message.set(key, value.values)  
    # Set GRIB data and write to FDB
    message.set_array("values", data_array.values)
    nan_to_missing(message, data_array.values)
    target.write(message)


def write(target, template, attributes, data_array):

    message = template.copy()
    for key, value in attributes.items():
        message[key] = value

    iterate_xarray(write_message, (target, template), data_array, 'data')
    # xr.apply_ufunc(fdb_write_ufunc, data_array, data_array.coords,
    #                input_core_dims=[['data'], []],
    #                dask='parallelized',
    #                kwargs={'fdb': fdb, 'template': template})


class Target:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return


class FileTarget:
    def __init__(self, path, mode="wb"):
        self.file = open(path, mode)

    def __enter__(self):
        return self.file.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.file.__exit__(exc_type, exc_value, traceback)

    def write(self, message):
        message.write_to(self.file)


class FDBTarget:
    def __init__(self, fdb):
        self.fdb = fdb

    def write(self, message):
        self.fdb.archive(message.get_buffer())


def target_factory(target_option, out_file=None, fdb=None):
    if target_option == 'fdb':
        assert fdb is not None
        target = FDBTarget(fdb)
    elif target_option == 'file':
        assert out_file is not None
        target = FileTarget(out_file)
    else:
        raise ValueError(f"Target {target_option} not supported, accepted values are 'fdb' and 'file' ")
    return target


def write_grib(target, template, data, missing=-9999):

    message = template.copy()

    # replace missing values if any
    is_missing = np.isnan(data).any()
    if is_missing:
        data[np.isnan(data)] = missing
        message.set('missingValue', missing)
        message.set('bitmapPresent', 1)
    
    message.set_array('values', data)

    if is_missing:
        n_missing1 = len(data[data==missing])
        n_missing2 = message.get('numberOfMissing')
        if n_missing1 != n_missing2:
            raise Exception(f'Number of missing values in the message not consistent, is {n_missing1} and should be {n_missing2}')

    target.write(message)
