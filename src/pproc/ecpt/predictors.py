import numpy as np
import datetime
import functools

import eccodes
import earthkit.data
from earthkit.data import FieldList
from earthkit.data.readers.grib.metadata import StandAloneGribMetadata
from earthkit.data.readers.grib.codes import GribCodesHandle

from pproc.common.io import GribMetadata
from pproc.common.param_requester import ParamRequester
from pproc.config.param import ParamConfig
from pproc.config.types import ECPointConfig
from pproc.config.preprocessing import Expression


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
    config: ECPointConfig, param: ParamConfig, step_range: str, inputs: FieldList
) -> np.ndarray:
    tp = inputs.sel(param=config.predictant)
    start, end = map(int, step_range.split("-"))
    date_end = datetime.datetime.fromisoformat(tp[0].metadata("valid_datetime"))
    date_mid = date_end - datetime.timedelta(hours=(end - start) / 2)
    hour = date_mid.hour
    lon = tp[0].metadata().geography.longitudes()
    return _local_solar_time(hour, lon)


def wind_speed(
    name: str,
    config: ECPointConfig,
    param: ParamConfig,
    step_range: str,
    inputs: FieldList,
) -> np.ndarray:
    components = {
        "ws": ["u", "v"],
        "10si": ["10u", "10v"],
        "100si": ["100u", "100v"],
        "200si": ["200u", "200v"],
    }
    u, v = components[name]
    return np.sqrt(inputs.sel(param=u).values ** 2 + inputs.sel(param=v).values ** 2)


def _ratio(var_num, var_den):
    den_zero = var_den == 0
    ratio_mapped = var_num / np.where(den_zero, -9999, var_den)
    return np.where(den_zero, 0, ratio_mapped)


def cp_ratio(
    config: ECPointConfig, param: ParamConfig, step_range: str, inputs: FieldList
) -> np.ndarray:
    ratio = _ratio(inputs.sel(param="cp").values, inputs.sel(param="tp").values)
    return np.where(ratio > 1, 1, ratio)


def leaf_area_index(
    config: ECPointConfig, param: ParamConfig, step_range: str, inputs: FieldList
) -> np.ndarray:
    lai_h = inputs.sel(param="cvh").values * inputs.sel(param="lai_hv").values
    lai_l = inputs.sel(param="cvl").values * inputs.sel(param="lai_lv").values
    return lai_h + lai_l


PREDICTORS = {
    "local_solar_time": lst,
    "ws": functools.partial(wind_speed, "ws"),
    "10si": functools.partial(wind_speed, "10si"),
    "100si": functools.partial(wind_speed, "100si"),
    "200si": functools.partial(wind_speed, "200si"),
    "cp_ratio": cp_ratio,
    "lai": leaf_area_index,
}


def compute_predictors(
    config: ECPointConfig, param: ParamConfig, step_range: str, inputs: FieldList
):
    pred = []
    expected_shape = inputs.sel(param=config.predictant).values.shape
    for predictor in config.predictors:
        if isinstance(predictor, Expression):
            _, output = predictor.apply(
                [x.as_namespace("mars") for x in inputs.metadata()], inputs.values
            )
            pred_values = output[0]
        else:
            selected = inputs.sel(param=predictor)
            if len(selected) != 0:
                pred_values = selected.values
            elif predictor in PREDICTORS:
                pred_values = PREDICTORS[predictor](
                    config, param.dependencies.get(predictor, param), step_range, inputs
                )
            else:
                raise ValueError(
                    f"No data or computation method found for predictor {predictor}"
                )
        if pred_values.shape != expected_shape:
            pred_values = np.broadcast_to(pred_values, expected_shape)
        pred.append(pred_values)
    return np.asarray(pred)
