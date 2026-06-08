# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
from pathlib import Path

import pytest
import yaml

from earthkit.workflows.plugins.pproc import products

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "product, requests",
    [
        ["ensemble", os.path.join(ROOT_DIR, "templates", "prob.yaml")],
        ["anomaly", os.path.join(ROOT_DIR, "templates", "t850.yaml")],
        ["ensemble", os.path.join(ROOT_DIR, "templates", "ensms.yaml")],
        ["extreme", os.path.join(ROOT_DIR, "templates", "extreme.yaml")],
        ["ensemble", os.path.join(ROOT_DIR, "templates", "quantiles.yaml")],
    ],
    ids=["prob", "t850", "ensms", "extreme", "quantiles"],
)
def test_graph_construction(product, requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)
    schema_path = os.path.join(Path(ROOT_DIR).parent, "schema.yaml")
    getattr(products, product)(output_requests, schema_path)
