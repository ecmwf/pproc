# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os

import pytest
import yaml

from earthkit.workflows.graph import Graph

from earthkit.workflows.plugins.pproc.fluent import from_source
from earthkit.workflows.plugins.pproc.utils.request import Request
from ppcore.products import product_from_output, graph_from_outputs
from conftest import SCHEMA


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "requests",
    [
        os.path.join(ROOT_DIR, "templates", "ensms.yaml"),
    ],
    ids=["ensms"],
)
def test_graph_construction(requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)
    graph_from_outputs(output_requests, SCHEMA)


@pytest.mark.parametrize(
    "requests",
    [
        os.path.join(ROOT_DIR, "templates", "ensms.yaml"),
    ],
    ids=["ensms"],
)
def test_custom_source(requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)

    graph = Graph([])
    for req in output_requests:
        product = product_from_output(req, SCHEMA, metadata={"edition": 2})
        requests = []
        for request in product.config.inputs.fc.requests:
            req = Request(request)
            if "number" not in req:
                req.make_dim("number", 0)
            requests.append(req)
        sources = from_source(
            ["fdb"],
            requests,
            join_key="number",
        )
        new_action = product.action(
            ensemble_dim="number",
            forecast=sources,
        ).write([{"name": "null"}])
        graph += new_action.graph()
