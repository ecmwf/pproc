# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import array_api_compat
import earthkit.meteo.solar
from earthkit.data import FieldList
from earthkit.meteo.extreme import array as extreme
from earthkit.meteo.stats import array as stats
from earthkit.workflows.backends.earthkit import (
    Metadata,
    comp_str2func,
    new_fieldlist,
    resolve_metadata,
)
from meters import metered

from ppruntime import metadata


logger = logging.getLogger(__name__)


@metered("threshold", out=logger.debug)
def threshold(
    arr: FieldList,
    threshold: dict,
    *,
    metadata: Metadata = None,
) -> FieldList:
    xp = array_api_compat.array_namespace(arr.values)
    # Find all locations where nan appears as an ensemble value
    is_nan = xp.isnan(arr.values)
    thesh = comp_str2func(xp, comparison)(arr.values, value)
    res = xp.where(is_nan, xp.nan, thesh)
    return new_fieldlist(res, arr.metadata(), resolve_metadata(metadata, arr))


@metered("efi", out=logger.debug)
def efi(
    clim: FieldList,
    ens: FieldList,
    eps: float,
    *,
    metadata: Metadata = None,
) -> FieldList:
    res = extreme.efi(clim.values, ens.values, eps)
    resolved_metadata = resolve_metadata(metadata, clim, ens)
    return new_fieldlist(
        res,
        [ens[0].metadata()],
        {**resolved_metadata, **grib.efi(clim, ens, resolved_metadata)},
    )


@metered("sot", out=logger.debug)
def sot(
    clim: FieldList,
    ens: FieldList,
    number: int,
    eps: float,
    *,
    metadata: Metadata = None,
) -> FieldList:
    res = extreme.sot(clim.values, ens.values, number, eps)
    resolved_metadata = resolve_metadata(metadata, clim, ens)
    return new_fieldlist(
        res,
        [ens[0].metadata()],
        {
            **resolved_metadata,
            **grib.sot(clim, ens, resolved_metadata, number),
        },
    )


@metered("quantiles", out=logger.debug)
def quantiles(
    ens: FieldList, q_number: int, total_number: int, *, metadata: Metadata = None
) -> FieldList:
    quantile = q_number / total_number
    res = list(stats.iter_quantiles(ens.values, [quantile], method="numpy"))[0]
    resolved_metadata = resolve_metadata(metadata, ens)
    return new_fieldlist(
        res,
        [ens[0].metadata()],
        {
            **resolved_metadata,
            **grib.quantiles(ens, resolved_metadata, q_number, total_number),
        },
    )
