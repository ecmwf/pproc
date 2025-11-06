# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from pproc.schema.base import BaseSchema, dict_update
from pproc.schema.filters import _steplength, _selection, _steptype
from pproc.schema.utils import validate_request
from pproc.common.grib_helpers import fill_template_value
from pproc.common.utils import dict_apply


class ConfigSchema(BaseSchema):
    custom_filter = {
        "steplength": _steplength,
        "selection": _selection,
        "steptype": _steptype,
    }
    custom_update = {"interp_keys": dict_update}

    def config(self, output_request: dict) -> dict:
        output_request = validate_request(output_request)
        config = self.traverse(output_request)
        if "quantile" in output_request:
            numbers = np.zeros(len(output_request["quantile"]))
            totals = np.zeros(len(output_request["quantile"]))
            for index, quantile in enumerate(output_request["quantile"]):
                number, total = map(int, quantile.split(":"))
                numbers[index] = number
                totals[index] = total
            if (
                np.all(totals == totals[0])
                and np.all(np.diff(numbers) == 1)
                and len(numbers) == total + 1
            ):
                quantiles = int(totals[0])
            else:
                quantiles = list(numbers / totals)
            config["quantiles"] = quantiles
        if output_request["type"] == "sot":
            config["sot"] = output_request["number"]

        out_vals = output_request.copy()
        date = str(output_request["date"])
        out_vals.update(
            {
                "year": date[0:4],
                "month": date[4:6],
                "day": date[6:8],
                "steplength": _steplength(output_request, "step"),
            }
        )
        config = dict_apply(
            lambda v: fill_template_value(v, out_vals) if isinstance(v, str) else v,
            config,
        )
        return config
