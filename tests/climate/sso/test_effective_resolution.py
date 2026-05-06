"""Tests for pproc.climate.sso.effective_resolution.

Mirrors the ksh logic at lines 73-90 of generate_subgrid_orography_sso.ksh:

    ERES = ORES / 2                  # truncating integer division
    ERES = ERES - (ERES % 2)         # round DOWN to even
    MIR_ERES_SET = N${ERES}
    # special-case overrides:
    #   40   -> N48
    #   100  -> N128
    #   1000 -> N1024
    # for non-octahedral GTYPE_SET, MIR_ERES_SET = MIR_GTYPE_SET (passthrough).
"""

import pytest

from pproc.climate.sso.effective_resolution import (
    compute_effective_resolution,
    infer_grid_params,
)


class TestInferGridParams:
    def test_octahedral_O1280(self):
        assert infer_grid_params("O1280") == ("O", 1280)

    def test_octahedral_O80(self):
        assert infer_grid_params("O80") == ("O", 80)

    def test_reduced_gaussian_N256(self):
        assert infer_grid_params("N256") == ("N", 256)

    def test_full_gaussian_F128(self):
        assert infer_grid_params("F128") == ("F", 128)

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "O",
            "OO80",
            "X100",
            "O-1",
            "1280",
            "o80",
            "O0",
            "O 80",
            "O80 ",
            " O80",
            "O80x",
            "ON80",
        ],
    )
    def test_rejects_malformed(self, bad):
        with pytest.raises(ValueError):
            infer_grid_params(bad)

    def test_error_message_mentions_input(self):
        with pytest.raises(ValueError, match="X100"):
            infer_grid_params("X100")


class TestComputeEffectiveResolution:
    # --- special cases (verbatim from ksh lines 80-86) ---

    def test_O80_returns_N48_special(self):
        # ORES=80 -> 80//2=40 -> 40-(40%2)=40 -> special: 40 -> N48
        # This is the operational test-run default (per sso-migration.md).
        assert compute_effective_resolution("O", 80) == "N48"

    def test_O200_returns_N128_special(self):
        # ORES=200 -> 100 -> 100 -> special: N128
        assert compute_effective_resolution("O", 200) == "N128"

    def test_O2000_returns_N1024_special(self):
        # ORES=2000 -> 1000 -> 1000 -> special: N1024
        assert compute_effective_resolution("O", 2000) == "N1024"

    @pytest.mark.parametrize(
        "ores,expected",
        [
            (80, "N48"),
            (81, "N48"),  # 81//2=40, even -> 40 -> N48
            (82, "N48"),  # 82//2=41, 41-1=40 -> N48
            (200, "N128"),
            (201, "N128"),  # 201//2=100, even -> N128
            (2000, "N1024"),
            (2001, "N1024"),  # 2001//2=1000 -> N1024
        ],
    )
    def test_special_cases_parametrized(self, ores, expected):
        assert compute_effective_resolution("O", ores) == expected

    # --- general rule ---

    def test_O1280_returns_N640(self):
        assert compute_effective_resolution("O", 1280) == "N640"

    def test_O160_returns_N80(self):
        # ORES=160 -> 80, even, no specials
        assert compute_effective_resolution("O", 160) == "N80"

    def test_O400_returns_N200(self):
        assert compute_effective_resolution("O", 400) == "N200"

    # --- round-DOWN-to-even (ksh semantics, NOT banker's rounding) ---

    def test_round_down_to_even_odd_input_O79(self):
        # ORES=79 -> 79//2=39 -> 39-(39%2)=38 -> N38 (general rule)
        assert compute_effective_resolution("O", 79) == "N38"

    def test_round_down_to_even_odd_input_O83(self):
        # ORES=83 -> 83//2=41 -> 41-1=40 -> special N48
        assert compute_effective_resolution("O", 83) == "N48"

    def test_round_down_to_even_O6(self):
        # ORES=6 -> 6//2=3 -> 3-1=2 -> N2
        assert compute_effective_resolution("O", 6) == "N2"

    def test_round_down_to_even_O5(self):
        # ORES=5 -> 5//2=2 -> 2-0=2 -> N2
        assert compute_effective_resolution("O", 5) == "N2"

    def test_round_down_distinct_from_bankers_rounding(self):
        # If we used round(x/2) (banker's rounding) we'd get different
        # results on a half-integer boundary. Pin the ksh behaviour.
        # ORES=83: ksh gives 40 -> N48; banker's rounding of 41.5 -> 42 (no special)
        assert compute_effective_resolution("O", 83) == "N48"

    # --- passthrough for non-octahedral grids ---

    def test_N256_returns_N256(self):
        assert compute_effective_resolution("N", 256) == "N256"

    def test_F128_returns_F128(self):
        assert compute_effective_resolution("F", 128) == "F128"

    # --- error cases ---

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError):
            compute_effective_resolution("O", 0)

    def test_negative_resolution_raises(self):
        with pytest.raises(ValueError):
            compute_effective_resolution("O", -10)

    def test_negative_resolution_non_octahedral_raises(self):
        with pytest.raises(ValueError):
            compute_effective_resolution("N", -1)


class TestOperationalDefault:
    """Pin the operational test-run config from sso-migration.md."""

    def test_test_run_defaults(self):
        # grid_type="O", nominal_resolution=80 -> effective_grid="N48"
        grid_type, resolution = infer_grid_params("O80")
        assert grid_type == "O"
        assert resolution == 80
        assert compute_effective_resolution(grid_type, resolution) == "N48"
