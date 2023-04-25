import pytest

from pproc.prob.window_manager import ThresholdWindowManager, AnomalyWindowManager

param_config = {
    "steps": [
        {
            "interval": 12,
            "start_step": 12,
            "end_step": 240,
        },
        {"interval": 3, "start_step": 123, "end_step": 144},
    ],
    "windows": [
        {
            "periods": [{"range": [12, 12]}, {"range": [168, 240]}],
            "thresholds": [
                {
                    "out_paramid": 131073,
                    "comparison": "<=",
                    "value": 273.15,
                }
            ],
        },
        {
            "window_operation": "diff",
            "periods": [
                {"range": [120, 168]},
            ],
            "thresholds": [
                {
                    "out_paramid": 131073,
                    "comparison": "<=",
                    "value": 273.15,
                }
            ],
        },
        {
            "include_start_step": True,
            "periods": [
                {"range": [24, 48]},
            ],
            "thresholds": [
                {
                    "out_paramid": 131073,
                    "comparison": "<=",
                    "value": 273.15,
                }
            ],
        },
    ],
}

anomaly_windows = [
    {
        "periods": [{"range": [96, 132]}],
        "thresholds": [
            {
                "out_paramid": 131073,
                "comparison": "<=",
                "value": 273.15,
            }
        ],
    }
]


@pytest.mark.parametrize(
    "checkpoint_step, start_step, num_windows",
    [
        pytest.param(0, 12, 4, id="checkpoint-before-start"),
        pytest.param(12, 24, 3, id="instantaneous"),
        pytest.param(36, 24, 3, id="period-include-init"),
        pytest.param(120, 120, 2, id="diff-start"),
        pytest.param(132, 120, 2, id="diff-middle"),
        pytest.param(168, 180, 1, id="exclude-init-start"),
        pytest.param(228, 180, 1, id="exclude-init-middle"),
    ],
)
def test_threshold_recovery(checkpoint_step: int, start_step: int, num_windows: int):
    window_manager = ThresholdWindowManager(param_config, {})

    window_manager.update_from_checkpoint(checkpoint_step)
    assert window_manager.unique_steps[0] == start_step
    assert len(window_manager.windows) == num_windows
    assert len(window_manager.window_thresholds) == num_windows


@pytest.mark.parametrize(
    "checkpoint_step, start_step, num_windows, num_anomaly_windows",
    [
        pytest.param(120, 108, 2, 1, id="anomaly-before-standard"),
        pytest.param(132, 120, 2, 0, id="anomaly-complete"),
    ],
)
def test_anomaly_recovery(
    checkpoint_step: int, start_step: int, num_windows: int, num_anomaly_windows: int
):
    param_config["std_anomaly_windows"] = anomaly_windows
    window_manager = AnomalyWindowManager(param_config, {})

    window_manager.update_from_checkpoint(checkpoint_step)
    assert window_manager.unique_steps[0] == start_step
    assert len(window_manager.windows) == num_windows + num_anomaly_windows
    assert len([x for x in window_manager.windows.keys() if 'std' in x]) == num_anomaly_windows
    assert len(window_manager.window_thresholds) == num_windows + num_anomaly_windows


@pytest.mark.parametrize(
    "window_manager, window_sets",
    [
        (ThresholdWindowManager, ["windows"]),
        (AnomalyWindowManager, ["windows"]),
    ],
)
def test_checkpoint_past_end(window_manager, window_sets):
    window_manager = window_manager(param_config, {})

    window_manager.update_from_checkpoint(240)
    assert len(window_manager.unique_steps) == 0
    for window_set in window_sets:
        assert len(getattr(window_manager, window_set)) == 0
    assert len(window_manager.window_thresholds) == 0
