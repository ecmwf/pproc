# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any, Dict, Optional, List
import numpy as np

import eccodes

from pproc.config.targets import Target
from pproc.common.io import nan_to_missing
from pproc.common.grib_helpers import construct_message


def postprocess(
    ens: np.ndarray,
    metadata: List[eccodes.GRIBMessage],
    target: Target,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out_accum_key: Optional[str] = None,
    out_accum_values: Optional[List[Any]] = None,
    out_keys: Optional[Dict[str, Any]] = None,
):
    """Post-process data and write to target

    Parameters
    ----------
    ens: numpy array (..., npoints)
        Ensemble data (all dimensions but the last are squashed together)
    metadata: list of eccodes.GRIBMessage
        GRIB templates for output
    target: Target
        Target to write to
    vmin: float, optional
        Minimum output value
    vmax: float, optional
        Maximum output value
    out_accum_key: str, optional
        Accumulation key to set on the output, if number of output fields does not match inputs
    out_accum_values: list, optional
        Accumulation values to set on the output, if number of output fields does not match inputs
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    out_arrays = ens.reshape((-1, ens.shape[-1]))
    if len(out_arrays) != len(metadata):
        if out_accum_key is None:
            raise ValueError(
                "out_accum_key must be set if number of output fields is different from input fields"
            )
        if out_accum_values is not None and len(out_accum_values) != len(out_arrays):
            raise ValueError(
                "out_accum_values must be the same length as the number of output fields"
            )
    for i, field in enumerate(out_arrays):
        if vmin is not None or vmax is not None:
            np.clip(field, vmin, vmax, out=field)

        grib_keys = {} if out_keys is None else out_keys.copy()
        if len(out_arrays) != len(metadata):
            grib_keys[out_accum_key] = (
                i if not out_accum_values else out_accum_values[i]
            )
            template = metadata[0]
        else:
            template = metadata[i]
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, field))
        target.write(message)
