# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from pproc.schema.base import BaseSchema
from pproc.schema.filters import _steplength, _selection, _steptype
from pproc.common.grib_helpers import fill_template_values


class ConfigSchema(BaseSchema):
    custom_filter = {
        "steplength": _steplength,
        "selection": _selection,
        "steptype": _steptype,
    }

    def config(self, output_request: dict) -> dict:
        config = self.traverse(output_request)
        if output_request["type"] in ["pb", "cd"]:
            numbers = np.zeros(len(output_request["quantile"]))
            totals = np.zeros(len(output_request["quantile"]))
            for index, quantile in enumerate(output_request["quantile"]):
                number, total = map(int, quantile.split(":"))
                numbers[index] = number
                totals[index] = total
            if np.all(totals == totals[0]) and np.all(np.diff(numbers) == 1):
                quantiles = int(totals[0])
            else:
                quantiles = numbers / totals
            config["quantiles"] = quantiles
        if output_request["type"] == "sot":
            config["sot"] = output_request["number"]

        config["metadata"] = fill_template_values(
            config.get("metadata", {}), output_request
        )
        return config
