# (C) Copyright 2021- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import yaml

from pproc.schema.base import BaseSchema


class ConfigSchema(BaseSchema):
    def config(self, output_request: dict) -> dict:
        config = self.traverse(output_request)
        if output_request["type"] in ["pb", "cd"]:
            quantiles = []
            for quantile in output_request["quantile"]:
                number, total = map(int, quantile.split(":"))
                quantiles.append(number / total)
            config["quantiles"] = quantiles
        if output_request["type"] == "sot":
            config["sot"] = output_request["number"]
        return config
