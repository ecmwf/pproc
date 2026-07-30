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
from ppcore.schema.forecast import definition_to_dataset, ForecastDefinition
from ppcore.products import product_from_output, graph_from_outputs, action_from_outputs
from ppcore.utils.requests import expand
from conftest import SCHEMA


ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))


INPUTS = {
    "ensemble": [
        {
            "stream": "oper",
            "type": "fc",
            "step": list(range(0, 24, 3)),
            "param": ["228", "167"],
            "levtype": "sfc",
        },
        {
            "stream": "enfo",
            "type": "pf",
            "step": list(range(0, 24, 3)),
            "param": ["228", "167"],
            "number": list(range(1, 5)),
            "levtype": "sfc",
        },
        {
            "stream": "oper",
            "type": "fc",
            "step": list(range(0, 24, 3)),
            "param": ["130"],
            "levtype": "pl",
            "levelist": [250, 500, 850],
        },
        {
            "stream": "enfo",
            "type": "pf",
            "step": list(range(0, 24, 3)),
            "param": ["130"],
            "levtype": "pl",
            "levelist": [250, 500, 850],
            "number": list(range(1, 5)),
        },
    ],
    "deterministic": [
        {
            "stream": "oper",
            "type": "fc",
            "step": [0, 1, 2],
            "param": [
                "47",
                "165",
                "166",
                "167",
                "168",
                "169",
                "175",
                "176",
                "177",
                "228021",
            ],
            "levtype": "sfc",
            "date": "20260721",
            "time": "1200",
        },
        {
            "stream": "oper",
            "type": "fc",
            "step": list(range(0, 24, 3)),
            "param": ["130"],
            "levtype": "pl",
            "levelist": [250, 500, 850],
            "date": "20260721",
            "time": "1200",
        },
    ],
}


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
    "input_name, requests",
    [
        ["ensemble", os.path.join(ROOT_DIR, "templates", "ensms.yaml")],
        ["ensemble", os.path.join(ROOT_DIR, "templates", "quantiles.yaml")],
        ["deterministic", os.path.join(ROOT_DIR, "templates", "thermo.yaml")],
    ],
    ids=["ensms", "quantiles", "thermo"],
)
def test_custom_source(input_name, requests):
    with open(requests, "r") as f:
        output_requests = yaml.safe_load(f)

    graph = Graph([])
    defaults = {"class": "od", "date": "20230914", "time": "1200", "expver": "0001"}
    for req in expand(output_requests):
        inputs = [{**defaults, **inp} for inp in INPUTS[input_name]]
        forecast = ForecastDefinition(
            datacubes=inputs, unperturbed={"stream": "oper", "type": "fc"}
        )
        product = product_from_output(
            req,
            SCHEMA,
            forecast=forecast,
            metadata={"edition": 2},
        )
        source_actions = []
        is_ensemble = definition_to_dataset(forecast).is_ensemble
        for levtype in ["sfc", "pl"]:
            for request in inputs:
                if request["levtype"] != levtype:
                    continue
                req = Request(request)
                if is_ensemble and "number" not in req:
                    req.make_dim("number", 0)
                new_action = from_source(
                    ["fdb"],
                    [req],
                    join_key="number" if is_ensemble else "",
                )
                new_action = new_action.set_path(f"/levtype={levtype}/{req['param']}")
                source_actions.append(new_action)
        sources = merge(*source_actions)
        new_action = product.action(
            ensemble_dim="number",
            forecast=sources,
        ).write([{"name": "null"}])
        graph += new_action.graph()
