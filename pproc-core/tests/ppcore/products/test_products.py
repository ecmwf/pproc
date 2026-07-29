# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import yaml

from earthkit.workflows.graph import Graph
from earthkit.workflows.fluent import merge

from earthkit.workflows.plugins.pproc.fluent import from_source
from earthkit.workflows.plugins.pproc.utils.request import Request
from ppcore.schema.forecast import ForecastDefinition
from ppcore.products import product_from_output, graph_from_outputs, action_from_outputs
from ppcore.utils.requests import expand
from conftest import SCHEMA


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


@pytest.mark.parametrize(
    "requests",
    [
        os.path.join(ROOT_DIR, "templates", "ensms.yaml"),
        os.path.join(ROOT_DIR, "templates", "prob.yaml"),
        os.path.join(ROOT_DIR, "templates", "quantiles.yaml"),
        os.path.join(ROOT_DIR, "templates", "thermo.yaml"),
    ],
    ids=["ensms", "prob", "quantiles", "thermo"],
)
def test_graph_construction(requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)
    graph_from_outputs(output_requests, SCHEMA, forecast="enfo")


def test_action_construction():
    with open(os.path.join(ROOT_DIR, "templates", "ensms.yaml"), "r") as f:
        output_requests = yaml.safe_load(f)
    action_from_outputs(output_requests, SCHEMA, forecast="enfo")


@pytest.mark.parametrize(
    "requests",
    [
        os.path.join(ROOT_DIR, "templates", "ensms.yaml"),
        os.path.join(ROOT_DIR, "templates", "quantiles.yaml"),
    ],
    ids=["ensms", "quantiles"],
)
def test_custom_source(requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)

    graph = Graph([])
    defaults = {"class": "od", "date": "20230914", "time": "1200", "expver": "0001"}
    for req in expand(output_requests):
        inputs = [
            {
                **defaults,
                "stream": "oper",
                "type": "fc",
                "step": list(range(0, 24, 3)),
                "param": ["228", "167"],
                "levtype": "sfc",
            },
            {
                **defaults,
                "stream": "enfo",
                "type": "pf",
                "step": list(range(0, 24, 3)),
                "param": ["228", "167"],
                "number": list(range(1, 5)),
                "levtype": "sfc",
            },
            {
                **defaults,
                "stream": "oper",
                "type": "fc",
                "step": list(range(0, 24, 3)),
                "param": ["130"],
                "levtype": "pl",
                "levelist": [250, 500, 850],
            },
            {
                **defaults,
                "stream": "enfo",
                "type": "pf",
                "step": list(range(0, 24, 3)),
                "param": ["130"],
                "levtype": "pl",
                "levelist": [250, 500, 850],
                "number": list(range(1, 5)),
            },
        ]
        product = product_from_output(
            req,
            SCHEMA,
            forecast=ForecastDefinition(
                datacubes=inputs, unperturbed={"stream": "oper", "type": "fc"}
            ),
            metadata={"edition": 2},
        )
        source_actions = []
        for levtype in ["sfc", "pl"]:
            for request in inputs:
                if request["levtype"] != levtype:
                    continue
                req = Request(request)
                if "number" not in req:
                    req.make_dim("number", 0)
                new_action = from_source(
                    ["fdb"],
                    [req],
                    join_key="number",
                )
                new_action = new_action.set_path(f"/levtype={levtype}/{req['param']}")
                source_actions.append(new_action)
        sources = merge(*source_actions)
        new_action = product.action(
            ensemble_dim="number",
            forecast=sources,
        ).write([{"name": "null"}])
        graph += new_action.graph()
