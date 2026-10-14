import pytest

from earthkit.workflows.plugins.pproc.fluent import from_source
from earthkit.workflows.nodetree import datacubes


class TestThermalIndex:
    @pytest.mark.parametrize(
        "steps",
        [
            [0],
            [0, 1],
        ],
        ids=["single-step", "multiple-steps"],
    )
    @pytest.mark.parametrize("join", [True, False], ids=["join", "no-join"])
    def test_instantaneous(self, steps: list[int], join: bool):
        input_action = from_source(
            ["fdb"], [{"param": ["167", "168", "169"], "step": steps}]
        )
        result = input_action.thermal_index(
            function="ppruntime.thermal_indices.calc_hmdx",
            params=["167", "168"],
            metadata={"paramId": 261016},
            join=join,
        )
        cubes = datacubes(result.nodes)
        if not join:
            assert len(cubes) == 1
            assert cubes[0] == {"param": ["261016"], "step": steps}

    @pytest.mark.parametrize(
        "input_req, expected_steps",
        [
            [
                [
                    {"param": ["228021"], "step": [0, 1]},
                    {"param": ["167"], "step": [1]},
                ],
                [1],
            ],
            [
                [
                    {"param": ["228021"], "step": [0, 1, 2, 3]},
                    {"param": ["167"], "step": [1, 2, 3]},
                ],
                [1, 2, 3],
            ],
            [
                [{"param": ["228021"], "step": [1]}, {"param": ["167"], "step": [1]}],
                [1],
            ],
        ],
        ids=["single-step", "multiple-steps", "missing-zero"],
    )
    @pytest.mark.parametrize("join", [True, False], ids=["join", "no-join"])
    def test_accum_exact_steps(
        self, input_req: list[dict], expected_steps: list[int], join: bool
    ):
        input_action = from_source(["fdb"], input_req)
        result = input_action.thermal_index(
            function="ppruntime.thermal_indices.calc_cossza",
            params=["167", "228021"],
            deaccumulate=["228021"],
            metadata={"paramId": 214001},
            join=join,
        )
        cubes = datacubes(result.nodes)
        if not join:
            assert len(cubes) == 1
            assert cubes[0] == {"param": ["214001"], "step": expected_steps}

    @pytest.mark.parametrize(
        "steps, expected_steps",
        [
            [[0, 1], [1]],
            [[0, 1, 2, 3], [1, 2, 3]],
            [[1], [1]],
        ],
        ids=["single-step", "multiple-steps", "missing-zero"],
    )
    @pytest.mark.parametrize("join", [True, False], ids=["join", "no-join"])
    def test_accum_extra_steps(
        self, steps: list[int], expected_steps: list[int], join: bool
    ):
        input_action = from_source(
            ["fdb"], [{"param": ["228021", "167"], "step": steps}]
        )
        result = input_action.thermal_index(
            function="ppruntime.thermal_indices.calc_cossza",
            params=["167", "228021"],
            deaccumulate=["228021"],
            metadata={"paramId": 214001},
            join=join,
        )
        cubes = datacubes(result.nodes)
        if not join:
            assert len(cubes) == 1
            assert cubes[0] == {"param": ["214001"], "step": expected_steps}

    @pytest.mark.parametrize(
        "input_reqs, expected_steps",
        [
            [
                [
                    {
                        "param": [
                            "167",
                            "214001",
                            "47",
                            "169",
                            "228021",
                            "175",
                            "177",
                            "176",
                        ],
                        "step": [0, 1],
                    }
                ],
                [1],
            ],
            [
                [
                    {
                        "param": [
                            "167",
                            "214001",
                            "47",
                            "169",
                            "228021",
                            "175",
                            "177",
                            "176",
                        ],
                        "step": [0, 1, 2, 3],
                    }
                ],
                [1, 2, 3],
            ],
            [
                [
                    {
                        "param": [
                            "167",
                            "214001",
                            "47",
                            "169",
                            "228021",
                            "175",
                            "177",
                            "176",
                        ],
                        "step": [1],
                    }
                ],
                [1],
            ],
            [
                [
                    {
                        "param": ["47", "169", "228021", "175", "177", "176"],
                        "step": [0, 1],
                    },
                    {"param": ["167", "214001"], "step": [1]},
                ],
                [1],
            ],
            [
                [
                    {
                        "param": ["47", "169", "228021", "175", "177", "176"],
                        "step": [0, 1, 2, 3],
                    },
                    {"param": ["167", "214001"], "step": [1, 2, 3]},
                ],
                [1, 2, 3],
            ],
            [
                [
                    {
                        "param": ["47", "169", "228021", "175", "177", "176"],
                        "step": [1],
                    },
                    {"param": ["167", "214001"], "step": [1]},
                ],
                [1],
            ],
        ],
    )
    @pytest.mark.parametrize("join", [True, False], ids=["join", "no-join"])
    def test_utci(self, input_reqs: list[dict], expected_steps: list[int], join: bool):
        input_action = from_source(["fdb"], input_reqs)
        result = input_action.thermal_index(
            function="ppruntime.thermal_indices.calc_utci",
            params=["167", "214001", "47", "169", "228021", "175", "177", "176"],
            deaccumulate=["47", "169", "228021", "175", "177", "176"],
            metadata={"paramId": 261002},
            join=join,
        )
        cubes = datacubes(result.nodes)
        if not join:
            assert len(cubes) == 1
            assert cubes[0] == {"param": ["261002"], "step": expected_steps}


class TestAccumulation:
    @pytest.mark.parametrize(
        "input_req, accum_kwargs, expected",
        [
            (
                {"step": [0, 1]},
                {"operation": None, "coords": [[0, 1]], "deaccumulate": True},
                ["0-1"],
            ),
            (
                {"step": [0, 6, 12, 18, 24]},
                {"operation": "mean", "coords": [[0, 6, 12], [12, 18, 24]]},
                ["0-12", "12-24"],
            ),
            (
                {"step": 6},
                {"operation": None, "coords": [[6]]},
                [6],
            ),
            (
                {"step": [6, 12]},
                {"operation": None, "coords": [[6], [12]]},
                [6, 12],
            ),
            (
                {"date": ["20240101", "20240102", "20240103"]},
                {
                    "operation": "concatenate",
                    "coords": [["20240101", "20240102", "20240103"]],
                },
                ["20240101-20240103"],
            ),
        ],
    )
    def test_accumulation(self, input_req: dict, accum_kwargs, expected):
        input_action = from_source(["fdb"], [input_req])
        dim = list(input_req.keys())[0]
        accum_action = input_action.accumulation(
            **accum_kwargs,
            dim=dim,
        )
        cubes = datacubes(accum_action.nodes)
        assert len(cubes) == 1
        assert cubes[0][dim] == expected

    @pytest.mark.parametrize(
        "input_reqs, accum_kwargs, expected",
        [
            (
                [{"step": [1]}],
                {"operation": None, "coords": [[1]], "name": {"type": "end"}},
                [1],
            ),
            (
                [{"step": [6, 12, 18, 24]}],
                {
                    "operation": "mean",
                    "coords": [[6, 12], [12, 18, 24]],
                    "name": {"type": "default", "length": 12},
                },
                ["0-12", "12-24"],
            ),
        ],
        ids=["single-step", "range"],
    )
    def test_deaccumulate_missing_zero(
        self, input_reqs: list[dict], accum_kwargs, expected
    ):
        input_action = from_source(["fdb"], input_reqs)
        accum_action = input_action.accumulation(
            **accum_kwargs,
            deaccumulate=True,
            dim="step",
        )
        cubes = datacubes(accum_action.nodes)
        assert len(cubes) == 1
        assert cubes[0]["step"] == expected


# TODO:
# Add more acccumulation tests
# Add tests for from_source
# Add tests for preprocessing
# Add tests for ensemble statistics
# Fix the difference with select with expand=True/False, and allow concatenation
# inside merge instead of combine_by_coords? (to allow for unordered coords)
# Sorting is useful for dimensions like step, but not suitable for param
