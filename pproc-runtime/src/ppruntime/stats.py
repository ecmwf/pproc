# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional, Literal
import array_api_compat
from earthkit.data import FieldList
from earthkit.meteo.extreme import array as extreme
from earthkit.meteo.stats import array as stats
from meters import metered

from ppruntime.metadata import resolve_metadata
from ppruntime import metadata as ppmetadata
from ppruntime.utils import new_fieldlist


logger = logging.getLogger(__name__)


Comparisons = Literal["<", ">", "<=", ">=", "==", "!="]


def comp_str2func(array_module, comparison: str):
    if comparison == "<=":
        return array_module.less_equal
    if comparison == "<":
        return array_module.less
    if comparison == ">=":
        return array_module.greater_equal
    return array_module.greater


@metered("mask", out=logger.debug)
def mask(
    arr: FieldList,
    lower_comparison: Comparisons,
    lower_value: float,
    upper_comparison: Optional[Comparisons] = None,
    upper_value: Optional[float] = None,
    *,
    metadata: Optional[dict] = None,
) -> FieldList:
    xp = array_api_compat.array_namespace(arr.values)
    # Find all locations where nan appears as an ensemble value
    is_nan = xp.isnan(arr.values)
    thresh = comp_str2func(xp, lower_comparison)(arr.values, lower_value)
    if upper_comparison is not None:
        thresh = thresh & comp_str2func(xp, upper_comparison)(arr.values, upper_value)
    res = xp.where(is_nan, xp.nan, thresh)
    return new_fieldlist(res, arr.metadata(), resolve_metadata(metadata))


def logical_and(
    arr1: FieldList,
    arr2: FieldList,
) -> FieldList:
    xp = array_api_compat.array_namespace(arr1.values, arr2.values)
    is_nan = xp.isnan(arr1.values) | xp.isnan(arr2.values)
    res = xp.where(is_nan, xp.nan, arr1.values & arr2.values)
    return new_fieldlist(res, arr1.metadata())


@metered("efi", out=logger.debug)
def efi(
    clim: FieldList,
    ens: FieldList,
    eps: float,
    *,
    metadata: Optional[dict] = None,
) -> FieldList:
    res = extreme.efi(clim.values, ens.values, eps)
    resolved_metadata = resolve_metadata(metadata)
    return new_fieldlist(
        res,
        [ens[0].metadata()],
        {**resolved_metadata, **ppmetadata.efi(clim, ens, resolved_metadata)},
    )


@metered("sot", out=logger.debug)
def sot(
    clim: FieldList,
    ens: FieldList,
    number: int,
    eps: float,
    *,
    metadata: Optional[dict] = None,
) -> FieldList:
    res = extreme.sot(clim.values, ens.values, number, eps)
    resolved_metadata = resolve_metadata(metadata)
    return new_fieldlist(
        res,
        [ens[0].metadata()],
        {
            **resolved_metadata,
            **ppmetadata.sot(clim, ens, resolved_metadata, number),
        },
    )


@metered("quantiles", out=logger.debug)
def quantiles(
    ens: FieldList,
    q_number: int,
    total_number: int,
    *,
    metadata: Optional[dict] = None,
) -> FieldList:
    quantile = q_number / total_number
    res = list(stats.iter_quantiles(ens.values, [quantile], method="numpy"))[0]
    resolved_metadata = resolve_metadata(metadata)
    return new_fieldlist(
        res,
        [ens[0].metadata()],
        {
            **resolved_metadata,
            **ppmetadata.quantiles(ens, resolved_metadata, q_number, total_number),
        },
    )
