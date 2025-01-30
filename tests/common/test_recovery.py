import pytest

from pproc.common.recovery import Recovery


def test_config_uniqueness(tmp_path):
    config1 = {"param": "2t"}
    config2 = {"param": "swh"}
    recover1 = Recovery(
        tmp_path,
        config1,
        recover=False,
    )
    recover2 = Recovery(
        tmp_path,
        config2,
        recover=False,
    )
    assert recover1.filename != recover2.filename


@pytest.mark.parametrize("recover, num_checkpoints", [(True, 1), (False, 0)])
def test_recovery(tmp_path, recover: bool, num_checkpoints: int):
    config = {"param": "2t"}
    recover1 = Recovery(
        tmp_path,
        config,
        recover=False,
    )

    # Should take arguments of different types
    recover1.add_checkpoint(param="2t", window="10-20", step=10)
    assert len(recover1.checkpoints) == 1
    computed = recover1.computed(param="2t")
    assert len(computed) == 1
    assert computed[0] == {"param": "2t", "window": "10-20", "step": "10"}

    recover1.add_checkpoint(param="2t", window="10-20", step=10)
    assert len(recover1.checkpoints) == 1

    recover2 = Recovery(
        tmp_path,
        config,
        recover=recover,
    )
    assert len(recover2.checkpoints) == num_checkpoints
    if num_checkpoints == 1:
        assert recover2.checkpoints == recover1.checkpoints
