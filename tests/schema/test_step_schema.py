import pytest

from pproc.schema.step import StepSchema

from conftest import schema


def test_in_steps():
    test_schema = StepSchema(schema("windows"))
    in_steps = test_schema.in_steps({"stream": "enfo", "type": "em", "param": "167"})
    assert in_steps == (
        list(range(0, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6))
    )


@pytest.mark.parametrize(
    "out, expected, in_steps",
    [
        [
            {"stream": "enfo", "type": "em", "param": "167"},
            list(range(0, 145, 3)) + list(range(150, 361, 6)),
            None,
        ],
        [
            {"stream": "enfo", "type": "em", "param": "167"},
            list(range(0, 361, 12)),
            list(range(0, 361, 12)),
        ],
        [
            {"stream": "enfo", "type": "cf", "param": "261001"},
            list(range(1, 91)) + list(range(93, 145, 3)) + list(range(150, 361, 6)),
            None,
        ],
        [
            {"stream": "eefo", "type": "fcmean", "param": "167"},
            [f"{x}-{x+168}" for x in list(range(0, 1104 - 168 + 1, 24))],
            None,
        ],
        [
            {"stream": "msmm", "type": "fcmean", "param": "167", "date": "20241001"},
            list(range(1, 8)),
            None,
        ],
        [
            {"stream": "enfo", "type": "ep", "param": "131064"},
            ["120-240", "240-360", "120-168", "168-240"],
            None,
        ],
        [
            {"stream": "oper", "type": "fc", "param": "207"},
            list(range(1, 91)),
            list(range(1, 91)),
        ],
    ],
    ids=[
        "ensms_default",
        "ensms_insteps",
        "deaccumulate",
        "weekly",
        "monthly",
        "prob",
        "fc",
    ],
)
def test_out_steps(out, expected, in_steps):
    test_schema = StepSchema(schema("windows"))
    _, out_steps = test_schema.out_steps(out, in_steps)
    assert out_steps == expected
