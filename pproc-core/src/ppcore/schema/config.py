# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from earthkit.workflows.plugins.pproc.utils.metadata import fill_template_value
from ppcore.schema.base import BaseSchema
from ppcore.schema.base import dict_update
from ppcore.schema.filters import _selection
from ppcore.schema.filters import _steplength
from ppcore.schema.filters import _steptype
from ppcore.utils.dicts import dict_apply
from ppcore.utils.helpers import to_list
from ppcore.utils.requests import validate_request


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
        if output_request["type"] == "sot":
            sot_key = "number" if "number" in output_request else "quantile"
            if sot_key not in output_request:
                raise ValueError(
                    f"Output request of type 'sot' must contain either 'number' or 'quantile', but got: {output_request}"
                )
            config["sot"] = to_list(output_request[sot_key])
        elif "quantile" in output_request:
            out_quantiles = to_list(output_request["quantile"])
            numbers = np.zeros(len(out_quantiles))
            totals = np.zeros(len(out_quantiles))
            for index, quantile in enumerate(out_quantiles):
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
