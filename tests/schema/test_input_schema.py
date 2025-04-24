import pytest

from pproc.config.utils import expand
from pproc.schema.input import InputSchema
from pproc.schema.step import StepSchema

from conftest import schema

INPUTS = {
    "ensms": [
        {
            "stream": "enfo",
            "levtype": "sfc",
            "type": "em",
            "param": "167",
            "step": 3,
            "type": "cf",
        },
        {
            "stream": "enfo",
            "levtype": "sfc",
            "type": "em",
            "param": "167",
            "step": 3,
            "type": "pf",
            "number": list(range(1, 51)),
        },
    ],
    "thermofeel": [
        {
            "stream": "enfo",
            "levtype": "sfc",
            "param": ["169", "175", "176", "177", "228021", "47"],
            "step": [2, 3],
            "type": "cf",
        },
        {
            "stream": "enfo",
            "levtype": "sfc",
            "param": ["165", "166", "167", "168"],
            "step": 3,
            "type": "cf",
        },
    ],
    "t850": [
        {
            "stream": "enfo",
            "param": "130",
            "step": list(range(120, 169, 12)),
            "type": "cf",
            "date": "20250314",
            "time": 12,
            "levtype": "pl",
            "levelist": 250,
        },
        {
            "stream": "enfo",
            "param": "130",
            "step": list(range(120, 169, 12)),
            "type": "pf",
            "number": list(range(1, 51)),
            "date": "20250314",
            "time": 12,
            "levtype": "pl",
            "levelist": 250,
        },
        {
            "stream": "efhs",
            "param": "130",
            "step": list(range(132, 181, 12)),
            "type": "em",
            "date": "20250313",
            "time": "00",
            "levtype": "pl",
            "levelist": 250,
        },
        {
            "stream": "efhs",
            "param": "130",
            "step": list(range(132, 181, 12)),
            "type": "es",
            "date": "20250313",
            "time": "00",
            "levtype": "pl",
            "levelist": 250,
        },
    ],
    "efi": [
        {
            "stream": "eefo",
            "levtype": "sfc",
            "param": "167",
            "step": "0-168",
            "type": "fcmean",
            "date": "20250315",
            "number": list(range(0, 101)),
            "time": 0,
        },
        {
            "stream": "eehs",
            "levtype": "sfc",
            "param": "228004",
            "step": "0-168",
            "type": "cd",
            "date": "20250315",
            "time": "00",
            "quantile": [f"{x}:100" for x in range(0, 101)],
        },
    ],
    "monthly": [
        {
            "stream": "mmsf",
            "levtype": "sfc",
            "param": ["165", "166"],
            "step": list(range(0, 745, 6)),
            "type": "fc",
            "number": list(range(0, 51)),
            "date": "20241001",
        },
    ],
    "prob": [
        {
            "stream": "enfo",
            "levtype": "sfc",
            "param": "228",
            "step": [0, 24],
            "type": "pf",
            "number": list(range(1, 51)),
        },
    ],
}


@pytest.mark.parametrize(
    "output",
    [
        {"stream": "enfo", "type": "em", "param": "167", "step": 3, "levtype": "sfc"},
        {
            "stream": "enfo",
            "type": "cf",
            "param": "261001",
            "step": 3,
            "levtype": "sfc",
        },
        {
            "stream": "enfo",
            "type": "ep",
            "param": "131020",
            "step": "120-168",
            "date": "20250314",
            "time": 12,
            "levtype": "pl",
            "levelist": 250,
        },
        {
            "stream": "eefo",
            "levtype": "sfc",
            "type": "efi",
            "param": "132167",
            "step": "0-168",
            "date": "20250315",
            "time": 0,
        },
        {
            "stream": "msmm",
            "levtype": "sfc",
            "type": "fcmean",
            "param": "207",
            "fcmonth": 1,
            "date": "20241001",
            "number": list(range(0, 51)),
        },
        {
            "stream": "enfo",
            "levtype": "sfc",
            "type": "ep",
            "param": "131060",
            "step": "0-24",
        },
    ],
    ids=["ensms", "thermofeel", "t850", "efi", "monthly", "prob"],
)
def test_inputs(request, output):
    expected_inputs = INPUTS[request.node.callspec.id]
    input_schema = InputSchema(schema("inputs"))
    step_schema = StepSchema(schema("windows"))
    inputs = input_schema.inputs(output, step_schema)
    assert list(inputs) == expected_inputs

    expanded_inputs = sum([list(expand(x)) for x in expected_inputs], [])
    generated = list(
        input_schema.outputs(expanded_inputs, step_schema, output_template=output)
    )
    assert len(generated) == 1
    assert generated[0][0] == output
    assert generated[0][1] == expected_inputs


@pytest.mark.parametrize(
    "out_type, num_outputs",
    [["em", 1], ["cf", 11], ["ep", 52], ["efi", 1], ["fcmean", 3], ["ep", 7]],
    ids=["ensms", "thermofeel", "t850", "efi", "monthly", "prob"],
)
def test_outputs(request, out_type, num_outputs):
    expanded_inputs = sum(
        [list(expand(x)) for x in INPUTS[request.node.callspec.id]], []
    )
    input_schema = InputSchema(schema("inputs"))
    step_schema = StepSchema(schema("windows"))
    generated = list(
        input_schema.outputs(
            expanded_inputs, step_schema, output_template={"type": out_type}
        )
    )
    assert len(generated) == num_outputs


@pytest.mark.parametrize(
    "inputs, template, num_outputs",
    [
        [INPUTS["t850"], {"type": "em", "step": 120}, 1],
        [
            [
                {
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
                }
            ],
            {"type": "fc"},
            18,
        ],
        [
            [
                {
                    "stream": "enfo",
                    "param": ["228246", "228247"],
                    "type": "cf",
                    "levtype": "sfc",
                }
            ],
            {"type": "cf"},
            0,
        ],
        [
            [
                {
                    "stream": "enfo",
                    "param": "129",
                    "type": "cf",
                    "levtype": "sfc",
                    "step": 3,
                },
                {
                    "stream": "enfo",
                    "param": "129",
                    "type": "pf",
                    "levtype": "sfc",
                    "number": list(range(1, 51)),
                    "step": 3,
                },
                {
                    "stream": "enfo",
                    "param": "129",
                    "type": "cf",
                    "levtype": "pl",
                    "levelist": [50, 100],
                    "step": 3,
                },
                {
                    "stream": "enfo",
                    "param": "129",
                    "type": "pf",
                    "levtype": "pl",
                    "number": list(range(1, 51)),
                    "levelist": [50, 100],
                    "step": 3,
                },
            ],
            {"levtype": "pl", "levelist": 50, "type": "em"},
            1,
        ],
    ],
    ids=["redundant-steps", "redundant-params", "not-from-inputs", "levels"],
)
def test_redundant_inputs(inputs, template, num_outputs):
    expanded_inputs = sum([list(expand(x)) for x in inputs], [])
    input_schema = InputSchema(schema("inputs"))
    step_schema = StepSchema(schema("windows"))
    generated = list(
        input_schema.outputs(expanded_inputs, step_schema, output_template=template)
    )
    assert len(generated) == num_outputs
