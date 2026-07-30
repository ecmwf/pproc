# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import schema

from ppcore.schema.input import InputSchema
from ppcore.schema.step import StepSchema
from ppcore.schema.forecast import (
    ForecastDefinition,
    ClimatologyDefinition,
    DatasetDefinitions,
    definition_to_dataset,
)
from ppcore.schema.schema import Schema
from ppcore.utils.requests import update_request
from ppcore.utils.mars import extract_mars
from ppcore.utils.qube import qube_from_datacubes

INPUTS = {
    "ensms": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "oper",
                    "levtype": "sfc",
                    "param": "167",
                    "step": 3,
                    "type": "fc",
                    "time": "0000",
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "levtype": "sfc",
                    "param": "167",
                    "step": 3,
                    "type": "pf",
                    "number": list(range(1, 51)),
                    "time": "0000",
                },
            ],
            unperturbed={"stream": "oper", "type": "fc"},
        ),
        None,
    ],
    "thermofeel": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "oper",
                    "levtype": "sfc",
                    "param": ["169", "175", "176", "177", "228021", "47"],
                    "step": [2, 3],
                    "type": "fc",
                    "time": "0000",
                },
                {
                    "class": "od",
                    "stream": "oper",
                    "levtype": "sfc",
                    "param": ["165", "166", "167", "168"],
                    "step": 3,
                    "type": "fc",
                    "time": "0000",
                },
            ],
            unperturbed={"stream": "oper", "type": "fc"},
        ),
        None,
    ],
    "thermo_pf": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "enfo",
                    "levtype": "sfc",
                    "param": ["169", "175", "176", "177", "228021", "47"],
                    "step": [2, 3],
                    "type": "pf",
                    "time": "0000",
                    "number": [1, 2, 3],
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "levtype": "sfc",
                    "param": ["165", "166", "167", "168"],
                    "step": 3,
                    "type": "pf",
                    "time": "0000",
                    "number": [1, 2, 3],
                },
            ],
            unperturbed={"stream": "oper", "type": "fc"},
        ),
        None,
    ],
    "t850": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "oper",
                    "param": "130",
                    "step": list(range(120, 169, 6)),
                    "type": "fc",
                    "date": "20250314",
                    "time": "1200",
                    "levtype": "pl",
                    "levelist": 250,
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "param": "130",
                    "step": list(range(120, 169, 6)),
                    "type": "pf",
                    "number": list(range(1, 51)),
                    "date": "20250314",
                    "time": "1200",
                    "levtype": "pl",
                    "levelist": 250,
                },
            ],
            unperturbed={"stream": "oper", "type": "fc"},
        ),
        ClimatologyDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "efhs",
                    "param": "130",
                    "step": list(range(132, 181, 6)),
                    "type": "em",
                    "date": "20250313",
                    "time": "0000",
                    "levtype": "pl",
                    "levelist": 250,
                },
                {
                    "class": "od",
                    "stream": "efhs",
                    "param": "130",
                    "step": list(range(132, 181, 6)),
                    "type": "es",
                    "date": "20250313",
                    "time": "0000",
                    "levtype": "pl",
                    "levelist": 250,
                },
            ],
            scheme="ecmwf-4days",
        ),
    ],
    "efi": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "eefo",
                    "levtype": "sfc",
                    "param": "167",
                    "step": "0-168",
                    "type": "fcmean",
                    "date": "20250315",
                    "number": list(range(0, 101)),
                    "time": "0000",
                },
            ],
            unperturbed={"stream": "eefo", "type": "cf"},
        ),
        ClimatologyDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "eehs",
                    "levtype": "sfc",
                    "param": "228004",
                    "step": "0-168",
                    "type": "cd",
                    "date": "20250315",
                    "time": "0000",
                    "quantile": [f"{x}:100" for x in range(0, 101)],
                },
            ],
            scheme="ecmwf-2days",
        ),
    ],
    "monthly": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "mmsf",
                    "levtype": "sfc",
                    "param": ["165", "166"],
                    "step": list(range(6, 745, 6)),
                    "type": "fc",
                    "number": list(range(0, 51)),
                    "date": "20241001",
                    "time": "0000",
                },
            ],
        ),
        None,
    ],
    "prob": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "oper",
                    "levtype": "sfc",
                    "param": "228",
                    "step": [0, 24],
                    "type": "fc",
                    "time": "0000",
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "levtype": "sfc",
                    "param": "228",
                    "step": [0, 24],
                    "type": "pf",
                    "number": list(range(1, 51)),
                    "time": "0000",
                },
            ],
            unperturbed={"stream": "oper", "type": "fc"},
        ),
        None,
    ],
    "sfc-pl": [
        ForecastDefinition(
            datacubes=[
                {
                    "class": "od",
                    "stream": "oper",
                    "levtype": "pl",
                    "levelist": [250, 850],
                    "param": "130",
                    "step": 6,
                    "type": "fc",
                    "time": "0000",
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "levtype": "pl",
                    "levelist": [250, 850],
                    "param": "130",
                    "step": 6,
                    "type": "pf",
                    "number": list(range(1, 51)),
                    "time": "0000",
                },
                {
                    "class": "od",
                    "stream": "oper",
                    "levtype": "sfc",
                    "param": "167",
                    "step": 6,
                    "type": "fc",
                    "time": "0000",
                },
                {
                    "class": "od",
                    "stream": "enfo",
                    "levtype": "sfc",
                    "param": "167",
                    "step": 6,
                    "type": "pf",
                    "number": list(range(1, 51)),
                    "time": "0000",
                },
            ],
            unperturbed={"stream": "oper", "type": "fc"},
        ),
        None,
    ],
}


@pytest.mark.parametrize(
    "output",
    [
        {
            "class": "od",
            "stream": "enfo",
            "type": "em",
            "time": "0000",
            "param": "167",
            "step": 3,
            "levtype": "sfc",
        },
        {
            "class": "od",
            "stream": "oper",
            "type": "fc",
            "param": "261001",
            "step": 3,
            "levtype": "sfc",
            "time": "0000",
        },
        {
            "class": "od",
            "stream": "enfo",
            "type": "pf",
            "param": "261001",
            "step": 3,
            "levtype": "sfc",
            "time": "0000",
            "number": [1, 2, 3],
        },
        {
            "class": "od",
            "stream": "enfo",
            "type": "ep",
            "param": "131020",
            "step": "120-168",
            "date": "20250314",
            "time": "1200",
            "levtype": "pl",
            "levelist": 250,
            "selection": "default",
        },
        {
            "class": "od",
            "stream": "eefo",
            "levtype": "sfc",
            "type": "efi",
            "param": "132167",
            "step": "0-168",
            "date": "20250315",
            "time": "0000",
        },
        {
            "class": "od",
            "stream": "msmm",
            "levtype": "sfc",
            "type": "fcmean",
            "param": "207",
            "fcmonth": 1,
            "date": "20241001",
            "number": list(range(0, 51)),
            "time": "0000",
        },
        {
            "class": "od",
            "stream": "enfo",
            "levtype": "sfc",
            "type": "ep",
            "param": "131060",
            "step": "0-24",
            "time": "0000",
            "selection": "default",
        },
    ],
    ids=["ensms", "thermofeel", "thermo_pf", "t850", "efi", "monthly", "prob"],
)
def test_inputs(request, output):
    forecast, climatology = INPUTS[request.node.callspec.id]
    input_schema = InputSchema(schema("inputs"))
    step_schema = StepSchema(schema("windows"))
    all_inputs = list(input_schema.inputs(output, forecast, climatology))
    for dataset, clim in [(forecast, False), (climatology, True)]:
        if dataset is None:
            continue
        generated_inputs = list(
            qube_from_datacubes(
                [
                    extract_mars(x)
                    for x in all_inputs
                    if x.get("climatology", False) == clim
                ]
            ).datacubes()
        )
        expected_inputs = list(definition_to_dataset(dataset).qube.datacubes())
        assert expected_inputs == generated_inputs

    generated = list(
        input_schema.outputs(
            forecast, climatology, step_schema=step_schema, output_template=output
        )
    )
    assert len(generated) == 1
    assert generated[0][0] == extract_mars(output)
    for dataset, clim in [(forecast, False), (climatology, True)]:
        if dataset is None:
            continue
        generated_inputs = list(
            qube_from_datacubes(
                [
                    extract_mars(x)
                    for x in generated[0][1]
                    if x.get("climatology", False) == clim
                ]
            ).datacubes()
        )
        expected_inputs = list(definition_to_dataset(dataset).qube.datacubes())
        assert expected_inputs == generated_inputs


@pytest.mark.parametrize(
    "template, num_outputs",
    [
        [{"stream": "enfo", "type": ["em", "es"]}, 2],
        [
            {
                "stream": "oper",
                "type": "fc",
                "param": [
                    261002,
                    261001,
                    260004,
                    260005,
                    260255,
                    260242,
                    261016,
                    261018,
                    261015,
                    261014,
                    261023,
                    207,
                ],
            },
            12,
        ],
        [{"stream": "enfo", "type": "ep", "selection": "default"}, 52],
        [{"stream": "eefo", "type": "efi"}, 1],
        [{"stream": "msmm", "type": ["fcmean", "fcmax"]}, 3],
        [{"stream": "enfo", "type": "ep", "selection": "default"}, 7],
    ],
    ids=["ensms", "thermofeel", "t850", "efi", "monthly", "prob"],
)
def test_outputs(request, template, num_outputs):
    test_schema = Schema(**schema())
    forecast, climatology = INPUTS[request.node.callspec.id]
    generated = list(
        test_schema.outputs_from_inputs(forecast, climatology, output_template=template)
    )
    assert len(generated) == num_outputs


def test_input_format():
    input_schema = InputSchema(schema("inputs"))
    step_schema = StepSchema(schema("windows"))
    dataset, _ = INPUTS["sfc-pl"]
    generated = list(
        input_schema.outputs(
            dataset,
            step_schema=step_schema,
            output_template={"stream": "enfo", "type": "em"},
            enable_cache=False,
        )
    )
    expected_outputs = [
        {
            "class": "od",
            "stream": "enfo",
            "levtype": "pl",
            "levelist": [250, 850],
            "param": "130",
            "step": 6,
            "type": "em",
            "time": "0000",
        },
        {
            "class": "od",
            "stream": "enfo",
            "levtype": "sfc",
            "param": "167",
            "step": 6,
            "type": "em",
            "time": "0000",
        },
    ]
    for out, inputs in generated:
        assert out in expected_outputs
        for inp in inputs:
            selection, _ = definition_to_dataset(dataset).select(inp)
            assert selection.n_leaves > 0


@pytest.mark.parametrize(
    "inputs, template, num_outputs",
    [
        [INPUTS["t850"][0], {"stream": "enfo", "type": "em", "step": 120}, 1],
        [
            ForecastDefinition(
                datacubes=[
                    {
                        "class": "od",
                        "stream": "oper",
                        "param": [
                            "165",
                            "166",
                            "167",
                            "168",
                            "169",
                            "175",
                            "176",
                            "177",
                            "228021",
                            "47",
                            "228",
                        ],
                        "step": [2, 3],
                        "type": "fc",
                        "levtype": "sfc",
                        "time": "0000",
                    }
                ],
            ),
            {"stream": "oper", "type": "fc"},
            20,
        ],
        [
            ForecastDefinition(
                datacubes=[
                    {
                        "class": "od",
                        "stream": "oper",
                        "param": ["228246", "228247"],
                        "type": "fc",
                        "levtype": "sfc",
                        "time": "0000",
                        "step": 3,
                    }
                ],
            ),
            {"stream": "oper", "type": "fc"},
            0,
        ],
        [
            ForecastDefinition(
                datacubes=[
                    {
                        "class": "od",
                        "stream": "oper",
                        "param": "129",
                        "type": "fc",
                        "levtype": "sfc",
                        "step": 3,
                        "time": "0000",
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "param": "129",
                        "type": "pf",
                        "levtype": "sfc",
                        "number": list(range(1, 51)),
                        "step": 3,
                        "time": "0000",
                    },
                    {
                        "class": "od",
                        "stream": "oper",
                        "param": "129",
                        "type": "fc",
                        "levtype": "pl",
                        "levelist": [50, 100],
                        "step": 3,
                        "time": "0000",
                    },
                    {
                        "class": "od",
                        "stream": "enfo",
                        "param": "129",
                        "type": "pf",
                        "levtype": "pl",
                        "number": list(range(1, 51)),
                        "levelist": [50, 100],
                        "step": 3,
                        "time": "0000",
                    },
                ],
                unperturbed={"stream": "oper", "type": "fc"},
            ),
            {"stream": "enfo", "levtype": "pl", "levelist": 50, "type": "em"},
            1,
        ],
    ],
    ids=["redundant-steps", "redundant-params", "not-from-inputs", "levels"],
)
def test_redundant_inputs(inputs, template, num_outputs):
    input_schema = InputSchema(schema("inputs"))
    step_schema = StepSchema(schema("windows"))
    generated = list(
        input_schema.outputs(inputs, step_schema=step_schema, output_template=template)
    )
    assert len(generated) == num_outputs


@pytest.mark.parametrize(
    "number, updates",
    [
        [0, {"type": "cf"}],
        [[0, 1], [{"type": "cf"}, {"type": "pf", "number": 1}]],
        [[1, 2], {"type": "pf", "number": [1, 2]}],
    ],
    ids=["cf", "cf-and-pf", "pf"],
)
def test_fcstat_inputs(number, updates):
    input_schema = InputSchema(schema("inputs"))
    datasets = DatasetDefinitions(definitions=schema("datasets"))
    output = {
        "class": "od",
        "stream": "eefo",
        "type": "fcmean",
        "number": number,
        "param": "167",
        "step": "0-168",
        "time": "0000",
        "levtype": "sfc",
        "domain": "g",
    }
    inputs = input_schema.inputs(output, datasets.definition("eefo"))  # type: ignore
    base_input = output.copy()
    base_input.pop("number")
    expected = update_request({**base_input, "step": list(range(6, 169, 6))}, updates)
    assert list(inputs) == expected
