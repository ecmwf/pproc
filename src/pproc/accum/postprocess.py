from typing import Any, Dict, Optional, List
import numpy as np

import eccodes

from pproc.common.io import Target, nan_to_missing
from pproc.common.grib_helpers import construct_message


def postprocess(
    ens: np.ndarray,
    metadata: List[eccodes.GRIBMessage],
    target: Target,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out_paramid: Optional[str] = None,
    out_accum_key: Optional[str] = None,
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
    out_paramid: str, optional
        Parameter ID to set on the output
    out_accum_key: str, optional
        Accumulation key to set on the output, if number of output fields does not match inputs
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    out_arrays = ens.reshape((-1, ens.shape[-1]))
    if len(out_arrays) != len(metadata) and out_accum_key is None:
        raise ValueError(
            "out_accum_key must be set if number of output fields is different from input fields"
        )
    for i, field in enumerate(out_arrays):
        if vmin is not None or vmax is not None:
            np.clip(field, vmin, vmax, out=field)

        grib_keys = out_keys.copy()
        if len(out_arrays) != len(metadata):
            grib_keys[out_accum_key] = i
            template = metadata[0]
        else:
            template = metadata[i]
        if out_paramid is not None:
            grib_keys["paramId"] = out_paramid
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, field))
        target.write(message)
