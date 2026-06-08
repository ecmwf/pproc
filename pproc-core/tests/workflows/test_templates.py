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
from earthkit.workflows.graph import Graph
from earthkit.workflows.graph import deduplicate_nodes

from earthkit.workflows.plugins.pproc.fluent import from_source
from earthkit.workflows.plugins.pproc.templates import derive_template
from earthkit.workflows.plugins.pproc.templates import from_request
from earthkit.workflows.plugins.pproc.utils.request import Request
from ppcore.utils.requests import expand
from ppcore.utils.requests import squeeze


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sources = from_source(
    [
        Request(
            {
                "param": "130",
                "levtype": "pl",
                "levelist": [250, 850],
                "type": "pf",
                "number": list(range(0, 5)),
            }
        ),
        Request(
            {
                "param": "130",
                "levtype": "pl",
                "levelist": [250, 850],
                "type": "cf",
            }
        ),
    ],
    join_key="type",
    backend_kwargs={"stream": True},
)


@pytest.mark.parametrize(
    "requests",
    [
        os.path.join(ROOT_DIR, "templates", "prob.yaml"),
        os.path.join(ROOT_DIR, "templates", "ensms.yaml"),
        os.path.join(ROOT_DIR, "templates", "quantiles.yaml"),
    ],
    ids=["prob", "ensms", "quantiles"],
)
def test_from_request(requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)
    schema_path = os.path.join(Path(ROOT_DIR).parent, "schema.yaml")

    graph = Graph([])
    for req in output_requests:
        config = derive_template(req, schema_path)
        source = from_source(
            [
                Request(x, no_expand=("number",))
                for x in squeeze(
                    sum([list(expand(x)) for x in config.inputs], []),
                    ["step", "number", "param", "levelist"],
                )
            ],
            join_key="type",
            backend_kwargs={"stream": True},
        )
        new_action = from_request(
            req,
            schema_path,
            ensemble_dim="type",
            forecast=source,
            metadata={"edition": 2},
        ).write({"name": "null"})
        print("GRAPH", new_action.nodes)
        graph += new_action.graph()

    graph = deduplicate_nodes(graph)
