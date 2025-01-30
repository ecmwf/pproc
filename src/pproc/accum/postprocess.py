from typing import Any, Dict, Optional

import eccodes
import numpy as np

from pproc.config.targets import Target
from pproc.common.io import nan_to_missing
from pproc.common.grib_helpers import construct_message


def postprocess(
    ens: np.ndarray,
    template: eccodes.GRIBMessage,
    target: Target,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out_keys: Optional[Dict[str, Any]] = None,
):
    """Post-process data and write to target

    Parameters
    ----------
    ens: numpy array (..., npoints)
        Ensemble data (all dimensions but the last are squashed together)
    template: eccodes.GRIBMessage
        GRIB template for output
    target: Target
        Target to write to
    vmin: float, optional
        Minimum output value
    vmax: float, optional
        Maximum output value
    out_keys: dict, optional
        Extra GRIB keys to set on the output
    """
    start_index = template.get("number", 0)
    for i, field in enumerate(ens.reshape((-1, ens.shape[-1]))):
        if vmin is not None or vmax is not None:
            np.clip(field, vmin, vmax, out=field)

        index = start_index + i
        tp = (
            "pf" if index > 0 and template.get("type") == "cf" else template.get("type")
        )

        out_keys = {} if out_keys is None else out_keys
        if callable(out_keys):
            out_keys = out_keys(tp)

        grib_keys = {
            **out_keys,
            "perturbationNumber": index,
        }
        grib_keys.setdefault("type", tp)
        message = construct_message(template, grib_keys)
        message.set_array("values", nan_to_missing(message, field))
        target.write(message)
