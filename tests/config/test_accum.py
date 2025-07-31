import pytest

from pproc.config.accumulation import LegacyWindowConfig


@pytest.mark.parametrize(
    "config1, config2, merged, expected",
    [
        [
            {"operation": "difference", "coords": [[0, 6]]},
            {"operation": "minimum", "coords": [[0, 6]]},
            False,
            None,
        ],
        [
            {"operation": "difference", "coords": [[0, 6]]},
            {"operation": "difference", "coords": [[12, 18]]},
            True,
            {"operation": "difference", "coords": [[0, 6], [12, 18]]},
        ],
        [
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 6}]},
            {
                "operation": "minimum",
                "coords": [[12, 18]],
                "thresholds": [{"value": 6}],
            },
            True,
            {
                "operation": "minimum",
                "coords": [[0, 6], [12, 18]],
                "thresholds": [{"value": 6}],
            },
        ],
        [
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 6}]},
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 12}]},
            True,
            {
                "operation": "minimum",
                "coords": [[0, 6]],
                "thresholds": [{"value": 6}, {"value": 12}],
            },
        ],
        [
            {"operation": "minimum", "coords": [[0, 6]], "thresholds": [{"value": 6}]},
            {
                "operation": "minimum",
                "coords": [[12, 18]],
                "thresholds": [{"value": 12}],
            },
            False,
            None,
        ],
    ],
    ids=[
        "diff-op",
        "merge-steps",
        "threshold-merge-steps",
        "threshold-merge",
        "no-merge",
    ],
)
def test_legacy_merge(config1, config2, merged, expected):
    accum1 = LegacyWindowConfig(**config1)
    accum2 = LegacyWindowConfig(**config2)
    assert accum1.merge(accum2) == merged
    if merged:
        assert accum1 == LegacyWindowConfig(**expected)
