import pytest

from pproc.config.utils import expand
from pproc.schema.input import InputSchema
from pproc.schema.step import StepSchema

from conftest import schema

INPUTS = {
    "ensms": [
        {
            "stream": "enfo",
            "type": "em",
            "param": "167",
            "step": 3,
            "type": "cf",
        },
        {
            "stream": "enfo",
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
            "param": ["169", "175", "176", "177", "228021", "47"],
            "step": [2, 3],
            "type": "cf",
        },
        {
            "stream": "enfo",
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
        },
        {
            "stream": "enfo",
            "param": "130",
            "step": list(range(120, 169, 12)),
            "type": "pf",
            "number": list(range(1, 51)),
            "date": "20250314",
            "time": 12,
        },
        {
            "stream": "efhs",
            "param": "130",
            "step": list(range(132, 181, 12)),
            "type": "em",
            "date": "20250313",
            "time": "00",
        },
        {
            "stream": "efhs",
            "param": "130",
            "step": list(range(132, 181, 12)),
            "type": "es",
            "date": "20250313",
            "time": "00",
        },
    ],
    "efi": [
        {
            "stream": "eefo",
            "param": "167",
            "step": "0-168",
            "type": "fcmean",
            "date": "20250315",
            "number": list(range(0, 101)),
            "time": 0,
        },
        {
            "stream": "eehs",
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
        {"stream": "enfo", "type": "em", "param": "167", "step": 3},
        {"stream": "enfo", "type": "cf", "param": "261001", "step": 3},
        {
            "stream": "enfo",
            "type": "ep",
            "param": "131020",
            "step": "120-168",
            "date": "20250314",
            "time": 12,
        },
        {
            "stream": "eefo",
            "type": "efi",
            "param": "132167",
            "step": "0-168",
            "date": "20250315",
            "time": 0,
        },
        {
            "stream": "msmm",
            "type": "fcmean",
            "param": "207",
            "fcmonth": 1,
            "date": "20241001",
            "number": list(range(0, 51)),
        },
        {
            "stream": "enfo",
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


def test_redundant_inputs():
    expanded_inputs = sum([list(expand(x)) for x in INPUTS["t850"]], [])
    input_schema = InputSchema(schema("inputs"))
    generated = list(
        input_schema.outputs(
            expanded_inputs, None, output_template={"type": "em", "step": 120}
        )
    )
    assert len(generated) == 1
