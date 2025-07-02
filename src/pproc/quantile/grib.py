# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import eccodes

from pproc.common.grib_helpers import construct_message


def quantiles_template(
    template: eccodes.GRIBMessage,
    pert_number: int,
    total_number: int,
    out_keys: dict,
) -> eccodes.GRIBMessage:

    edition = out_keys.get("edition", template.get("edition"))
    if edition not in (1, 2):
        raise ValueError(f"Unsupported GRIB edition {edition}")
    grib_keys = {**out_keys}
    if edition == 1:
        grib_keys.update(
            {
                "totalNumber": total_number,
                "perturbationNumber": pert_number,
            }
        )
    else:
        grib_keys.setdefault("productDefinitionTemplateNumber", 86)
        grib_keys.update(
            {
                "totalNumberOfQuantiles": total_number,
                "quantileValue": pert_number,
            }
        )
    message = construct_message(template, grib_keys)
    return message
