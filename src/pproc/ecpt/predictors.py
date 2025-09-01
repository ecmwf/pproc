import numpy as np
import datetime

import eccodes
import earthkit.data
from earthkit.data import FieldList
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from earthkit.data.readers.grib.codes import GribCodesHandle

from pproc.common.io import GribMetadata
from pproc.common.param_requester import ParamRequester
from pproc.config.param import ParamConfig
from pproc.config.types import ECPointConfig


def to_ekmetadata(metadata: list[GribMetadata]) -> list[StandAloneGribMetadata]:
    return [
        StandAloneGribMetadata(
            GribCodesHandle(eccodes.codes_clone(x._handle), None, None)
        )
        for x in metadata
    ]


def _local_solar_time(hour: int, longitudes: np.ndarray) -> np.ndarray:
    lst_pos = np.where(longitudes >= 0, hour + (longitudes / 15), 0)
    temp_pos = np.where(lst_pos >= 24, lst_pos - 24, lst_pos)
    lst_neg = np.where(longitudes < 0, hour - abs((longitudes / 15)), 0)
    temp_neg = np.where(lst_neg < 0, lst_neg + 24, lst_neg)
    return temp_pos + temp_neg


def lst(
    config: ECPointConfig, param: ParamConfig, window: str, inputs: FieldList
) -> np.ndarray:
    tp = inputs.sel(param="tp")
    start, end = map(int, window.split("-"))
    date_end = datetime.datetime.fromisoformat(tp[0].metadata("valid_datetime"))
    date_mid = date_end - datetime.timedelta(hours=(end - start) / 2)
    hour = date_mid.hour
    lon = tp[0].metadata().geography.longitudes()
    return np.broadcast_to(_local_solar_time(hour, lon), tp.values.shape)


def sdfor(
    config: ECPointConfig, param: ParamConfig, window: str, inputs: FieldList
) -> np.ndarray:
    shape = inputs.sel(param="tp").values.shape
    sdfor = inputs.sel(param="sdfor").values
    return np.broadcast_to(sdfor, shape)


def ws(
    config: ECPointConfig, param: ParamConfig, window: str, inputs: FieldList
) -> np.ndarray:
    return np.sqrt(
        inputs.sel(param="u").values ** 2 + inputs.sel(param="v").values ** 2
    )


def _ratio(var_num, var_den):
    den_zero = var_den == 0
    ratio_mapped = var_num / np.where(den_zero, -9999, var_den)
    return np.where(den_zero, 0, ratio_mapped)


def cpr(
    config: ECPointConfig, param: ParamConfig, window: str, inputs: FieldList
) -> np.ndarray:
    return _ratio(inputs.sel(param="cp").values, inputs.sel(param="tp").values)


PREDICTORS = {
    "sdfor": sdfor,
    "lst": lst,
    "ws": ws,
    "cpr": cpr,
}


def compute_predictors(
    config: ECPointConfig, param: ParamConfig, window: str, inputs: FieldList
):
    pred = []
    for predictor in config.predictors:
        if predictor in PREDICTORS:
            pred.append(
                PREDICTORS[predictor](
                    config, param.dependencies.get(predictor, param), window, inputs
                )
            )
        else:
            selected = inputs.sel(param=predictor)
            if len(selected) == 0:
                raise ValueError(f"No data found for predictor {predictor}")
            pred.append(selected.values)
    if not np.all([x.shape == pred[0].shape for x in pred]):
        raise ValueError(
            f"Shapes of all predictors should be the same. Got {[x.shape for x in pred]}"
        )
    return np.asarray(pred)
