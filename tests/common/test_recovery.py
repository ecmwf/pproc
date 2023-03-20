import tempfile
import datetime
import pathlib
import yaml
import pytest

from pproc.common import Recovery


@pytest.fixture
def temp_path():
    temp_dir = tempfile.TemporaryDirectory()
    path = pathlib.Path(temp_dir.name)
    with open(f"{path}/config1.yaml", "w") as f1:
        f1.write(yaml.dump({"param": "2t"}))

    yield path

    temp_dir.cleanup()


def test_config_uniqueness(temp_path):
    with open(f"{temp_path}/config2.yaml", "w") as f1:
        f1.write(yaml.dump({"param": "swh"}))
    recover1 = Recovery(
        temp_path,
        f"{temp_path}/config1.yaml",
        datetime.datetime(2023, 1, 1),
        recover=False,
    )
    recover2 = Recovery(
        temp_path,
        f"{temp_path}/config2.yaml",
        datetime.datetime(2023, 1, 1),
        recover=False,
    )
    assert recover1.filename != recover2.filename


@pytest.mark.parametrize("recover, num_checkpoints", [(True, 1), (False, 0)])
def test_recovery(temp_path, recover: bool, num_checkpoints: int):
    recover1 = Recovery(
        temp_path,
        f"{temp_path}/config1.yaml",
        datetime.datetime(2023, 1, 1),
        recover=False,
    )

    # Should take arguments of different types
    recover1.add_checkpoint("2t", "10-20", 10)
    assert len(recover1.checkpoints) == 1

    recover1.add_checkpoint("2t", "10-20", 10)
    assert len(recover1.checkpoints) == 1

    recover2 = Recovery(
        temp_path,
        f"{temp_path}/config1.yaml",
        datetime.datetime(2023, 1, 1),
        recover=recover,
    )
    assert len(recover2.checkpoints) == num_checkpoints
    if num_checkpoints == 1:
        assert recover2.checkpoints == recover1.checkpoints
