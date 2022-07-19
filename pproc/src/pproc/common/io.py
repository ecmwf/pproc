import numpy as np
import xarray as xr
import eccodes
from dataclasses import dataclass, field
from typing import Union, Any, List, Dict


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


def fdb_read(fdb, request):
    """Load grib messages from FDB from request and returns Xarray DataArray

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

    fdb_reader = fdb.retrieve(request)
    fields_dims = [key for key in request if isinstance(request[key], (list, range))]
    eccodes_reader = eccodes.StreamReader(fdb_reader)
    fields = read_grib_messages(eccodes_reader, fields_dims)

    return fields.to_xarray()


def fdb_write_ufunc(data, fdb, template, values_to_set):

    message = template.copy()
    for key, value in values_to_set.items():
        message[key] = value
    
    # Set GRIB data and write to FDB
    message.set_array("values", data)
    fdb.write(message)


# def write_fdb(data_array, values_to_set):

    # for data in data_array:
        

    # out_grib = template_grib.copy()
    # out_grib.set("step", step)
    # out_grib.set("type", "ep")
    # out_grib.set("paramId", threshold["out_paramid"])
    # out_grib.set("localDefinitionNumber", 5)
    # out_grib.set("localDecimalScaleFactor", 2)
    # out_grib.set("thresholdIndicator", 2)
    # out_grib.set("upperThreshold", threshold["value"])

    # # Set GRIB data and write to FDB
    # out_grib.set_array("values", data)
    # fdb.archive(out_grib.get_buffer())

