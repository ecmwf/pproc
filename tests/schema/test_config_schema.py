from pproc.schema.config import ConfigSchema

from conftest import schema


def test_reconstruct():
    config_schema = ConfigSchema(schema("config"))
    cfgs = list(config_schema.reconstruct(entrypoint="pproc-ensms"))
    assert len(cfgs) != 0
    for out, cfg in cfgs:
        assert cfg["entrypoint"] == "pproc-ensms"
        assert out["type"] in ["em", "es"]
    assert len([x for x, _ in cfgs if x["type"] == "em"]) == len(
        [x for x, _ in cfgs if x["type"] == "es"]
    )
