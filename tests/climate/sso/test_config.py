"""Tests for pproc.climate.sso.config.SSOConfig.

Verifies that the Pydantic model correctly mirrors the ksh script's env-var
table (see ``.weave/learnings/sso-migration.md``) and that ``resolve()``:

- infers ``model_grid_type`` / ``model_resolution`` from ``target_grid``
  when not explicitly supplied,
- defaults ``output_grid`` to ``target_grid`` when empty,
- computes ``effective_resolution`` via Unit C
  (``compute_effective_resolution``),
- honours user overrides of the model grid/resolution (the ksh test-run case
  where target=N256 but model=O80, yielding eres=N48),
- is idempotent,
- preserves the value via YAML/dict round-trip,
- rejects malformed and extra fields,
- does not mutate the caller's input dict.
"""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from pproc.climate.sso.config import SSOConfig


@pytest.fixture
def minimal_kwargs(tmp_path):
    orog = tmp_path / "orog"
    orog.touch()
    lsm = tmp_path / "lsm"
    lsm.touch()
    return {
        "orography": str(orog),
        "land_mask": str(lsm),
        "target_grid": "N256",
    }


class TestResolveInference:
    def test_target_grid_only_N256(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs).resolve()
        assert cfg.model_grid_type == "N"
        assert cfg.model_resolution == 256
        assert cfg.output_grid == "N256"
        assert cfg.effective_resolution == "N256"

    def test_target_grid_only_O80(self, minimal_kwargs):
        minimal_kwargs["target_grid"] = "O80"
        cfg = SSOConfig(**minimal_kwargs).resolve()
        assert cfg.model_grid_type == "O"
        assert cfg.model_resolution == 80
        assert cfg.output_grid == "O80"
        assert cfg.effective_resolution == "N48"

    def test_target_grid_F128(self, minimal_kwargs):
        minimal_kwargs["target_grid"] = "F128"
        cfg = SSOConfig(**minimal_kwargs).resolve()
        assert cfg.model_grid_type == "F"
        assert cfg.model_resolution == 128
        assert cfg.output_grid == "F128"
        assert cfg.effective_resolution == "F128"

    def test_explicit_overrides_inference_ksh_test_run(self, minimal_kwargs):
        """Reproduces the ksh test-run config: target=N256, but model=O80.

        From sso-migration.md: ``MIR_GTYPE_SET=N256`` (output) but
        ``GTYPE_SET=O`` and ``ORES=80`` (model). The script computes the
        effective resolution from the model -- not the output.
        """
        cfg = SSOConfig(
            **minimal_kwargs,
            model_grid_type="O",
            model_resolution=80,
        ).resolve()
        assert cfg.model_grid_type == "O"
        assert cfg.model_resolution == 80
        assert cfg.target_grid == "N256"
        assert cfg.output_grid == "N256"  # defaults to target
        assert cfg.effective_resolution == "N48"  # eres from model O80

    def test_output_grid_defaults_to_target_grid_when_empty(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs).resolve()
        assert cfg.output_grid == cfg.target_grid

    def test_explicit_output_grid_preserved(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs, output_grid="O1280").resolve()
        assert cfg.output_grid == "O1280"
        assert cfg.target_grid == "N256"

    def test_partial_explicit_only_grid_type_falls_back_to_inference(
        self, minimal_kwargs
    ):
        # Only model_grid_type given (no model_resolution): user did NOT
        # provide BOTH explicitly, so the inference path must run.
        cfg = SSOConfig(**minimal_kwargs, model_grid_type="O").resolve()
        # Inference from target_grid="N256" wins.
        assert cfg.model_grid_type == "N"
        assert cfg.model_resolution == 256

    def test_partial_explicit_only_resolution_falls_back_to_inference(
        self, minimal_kwargs
    ):
        cfg = SSOConfig(**minimal_kwargs, model_resolution=80).resolve()
        assert cfg.model_grid_type == "N"
        assert cfg.model_resolution == 256


class TestIdempotency:
    def test_resolve_is_idempotent(self, minimal_kwargs):
        cfg1 = SSOConfig(**minimal_kwargs).resolve()
        cfg2 = cfg1.resolve()
        assert cfg1 == cfg2

    def test_resolve_thrice_is_idempotent(self, minimal_kwargs):
        cfg1 = SSOConfig(**minimal_kwargs).resolve()
        cfg2 = cfg1.resolve().resolve().resolve()
        assert cfg1 == cfg2

    def test_resolve_idempotent_with_explicit_override(self, minimal_kwargs):
        cfg1 = SSOConfig(
            **minimal_kwargs, model_grid_type="O", model_resolution=80
        ).resolve()
        cfg2 = cfg1.resolve()
        assert cfg1 == cfg2
        assert cfg1.effective_resolution == "N48"


class TestNoMutation:
    def test_resolve_does_not_mutate_input_dict(self, minimal_kwargs):
        snapshot = copy.deepcopy(minimal_kwargs)
        SSOConfig(**minimal_kwargs).resolve()
        assert minimal_kwargs == snapshot


class TestSerialization:
    def test_yaml_round_trip(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs).resolve()
        dumped = yaml.safe_dump(cfg.model_dump(mode="json"))
        loaded = yaml.safe_load(dumped)
        cfg2 = SSOConfig(**loaded).resolve()
        assert cfg.model_dump(mode="json") == cfg2.model_dump(mode="json")

    def test_yaml_round_trip_ksh_test_run(self, minimal_kwargs):
        cfg = SSOConfig(
            **minimal_kwargs,
            model_grid_type="O",
            model_resolution=80,
        ).resolve()
        dumped = yaml.safe_dump(cfg.model_dump(mode="json"))
        loaded = yaml.safe_load(dumped)
        cfg2 = SSOConfig(**loaded).resolve()
        assert cfg == cfg2
        assert cfg2.effective_resolution == "N48"

    def test_dict_round_trip_equality(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs).resolve()
        cfg2 = SSOConfig(**cfg.model_dump()).resolve()
        assert cfg == cfg2


class TestEnvVarMapping:
    """Pin the field names against the env-var table from sso-migration.md."""

    def test_all_documented_fields_present(self):
        fields = SSOConfig.model_fields.keys()
        for required in (
            "orography",
            "land_mask",
            "source_orography",
            "target_grid",
            "model_grid_type",
            "model_resolution",
            "output_grid",
            "effective_resolution",
            "output_dir",
            "grib_roundtrip",
            "dump_intermediates",
        ):
            assert required in fields, f"missing field: {required}"

    def test_mir_version_not_exposed(self):
        # MIR_VERSION / MIR_COMPUTE_VERSION are tool config, not pipeline
        # config -- they must not be on the model.
        fields = SSOConfig.model_fields.keys()
        assert "mir_version" not in fields
        assert "mir_compute_version" not in fields


class TestErrors:
    def test_empty_target_grid_raises(self, minimal_kwargs):
        minimal_kwargs["target_grid"] = ""
        with pytest.raises(ValidationError):
            SSOConfig(**minimal_kwargs)

    def test_negative_resolution_raises(self, minimal_kwargs):
        minimal_kwargs["model_resolution"] = -10
        with pytest.raises(ValidationError):
            SSOConfig(**minimal_kwargs)

    def test_extra_field_rejected(self, minimal_kwargs):
        minimal_kwargs["unknown_field"] = "foo"
        with pytest.raises(ValidationError):
            SSOConfig(**minimal_kwargs)

    def test_malformed_target_grid_raises_at_resolve(self, minimal_kwargs):
        minimal_kwargs["target_grid"] = "X100"
        # Construction is fine (string is non-empty); resolve() must
        # propagate the ValueError from infer_grid_params.
        with pytest.raises(ValueError):
            SSOConfig(**minimal_kwargs).resolve()

    def test_missing_required_orography_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            SSOConfig(land_mask=str(tmp_path / "lsm"), target_grid="N256")

    def test_missing_required_land_mask_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            SSOConfig(orography=str(tmp_path / "orog"), target_grid="N256")

    def test_missing_required_target_grid_raises(self, tmp_path):
        with pytest.raises(ValidationError):
            SSOConfig(
                orography=str(tmp_path / "orog"),
                land_mask=str(tmp_path / "lsm"),
            )


class TestDefaults:
    def test_default_toggles_are_false(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs)
        assert cfg.grib_roundtrip is False
        assert cfg.dump_intermediates is False

    def test_source_orography_defaults_to_none(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs)
        assert cfg.source_orography is None

    def test_resolve_returns_sso_config_instance(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs).resolve()
        assert isinstance(cfg, SSOConfig)


class TestBitsPerValue:
    """The ``bits_per_value`` knob controls GRIB ``bitsPerValue`` on the
    four output fields. ``None`` (default) means "don't write it" so
    eccodes inherits/defaults the value from the packing in use."""

    def test_default_is_none(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs)
        assert cfg.bits_per_value is None

    def test_explicit_32_constructs(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs, bits_per_value=32)
        assert cfg.bits_per_value == 32

    def test_explicit_16_constructs(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs, bits_per_value=16)
        assert cfg.bits_per_value == 16

    def test_zero_rejected(self, minimal_kwargs):
        with pytest.raises(ValidationError):
            SSOConfig(**minimal_kwargs, bits_per_value=0)

    def test_negative_rejected(self, minimal_kwargs):
        with pytest.raises(ValidationError):
            SSOConfig(**minimal_kwargs, bits_per_value=-1)

    def test_yaml_round_trip_with_explicit_value(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs, bits_per_value=32).resolve()
        dumped = yaml.safe_dump(cfg.model_dump(mode="json"))
        loaded = yaml.safe_load(dumped)
        cfg2 = SSOConfig(**loaded).resolve()
        assert cfg2.bits_per_value == 32
        assert cfg == cfg2

    def test_yaml_round_trip_with_default_none(self, minimal_kwargs):
        cfg = SSOConfig(**minimal_kwargs).resolve()
        dumped = yaml.safe_dump(cfg.model_dump(mode="json"))
        loaded = yaml.safe_load(dumped)
        cfg2 = SSOConfig(**loaded).resolve()
        assert cfg2.bits_per_value is None
        assert cfg == cfg2
