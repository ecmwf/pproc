# (C) Copyright 2025- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the ``write_outputs`` template fallback and the
``sea-surface`` product's template validator.

Both pieces implement the "single templated flag replaces N per-output
flags" mechanism. The tests cover:

- The template-fallback path in :func:`pproc.climate.generate.io.write_outputs`
  writes each ``<prefix>_NN`` logical output to a path derived from
  ``<prefix>_out_template`` with ``{month}`` substituted (accepting both
  ``{month}`` and ``{month:02d}`` placeholder styles).
- Missing ``{month}`` placeholder in the sea-surface config is rejected
  at construction time by the Pydantic field validator.
- The per-output ``<name>_out`` field still wins when both mechanisms
  could apply (a config carrying both must prefer the explicit field —
  covered here by a synthetic model, since no shipped product mixes them).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pytest
from conflator import CLIArg, ConfigModel
from pydantic import ConfigDict, Field

from pproc.climate.generate.io import write_outputs


# ---------------------------------------------------------------------------
# Template fallback in write_outputs
# ---------------------------------------------------------------------------


class _TemplateOnlyConfig(ConfigModel):
    """Synthetic config exposing only a ``sst_out_template`` string field."""

    model_config = ConfigDict(extra="forbid")

    sst_out_template: Annotated[
        str, CLIArg("--sst-out-template", default=None), Field(description="")
    ] = "./will-be-overridden.{month:02d}"


def test_template_fallback_writes_each_output_with_substituted_month(
    tmp_path: Path,
) -> None:
    cfg = _TemplateOnlyConfig(sst_out_template=str(tmp_path / "sst.{month:02d}"))
    results = {f"sst_{m:02d}": f"payload-{m:02d}".encode() for m in range(1, 13)}

    write_outputs(results, cfg)

    for m in range(1, 13):
        p = tmp_path / f"sst.{m:02d}"
        assert p.is_file(), f"missing {p}"
        assert p.read_bytes() == f"payload-{m:02d}".encode()


def test_template_fallback_accepts_unpadded_month_placeholder(
    tmp_path: Path,
) -> None:
    """The template's format spec is honoured; ``{month}`` without ``:02d``
    yields unpadded month numbers."""
    cfg = _TemplateOnlyConfig(sst_out_template=str(tmp_path / "sst_m{month}.grb"))
    results = {"sst_01": b"jan", "sst_11": b"nov"}

    write_outputs(results, cfg)

    assert (tmp_path / "sst_m1.grb").read_bytes() == b"jan"
    assert (tmp_path / "sst_m11.grb").read_bytes() == b"nov"


def test_template_fallback_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    cfg = _TemplateOnlyConfig(sst_out_template=str(nested / "sst.{month:02d}"))
    write_outputs({"sst_07": b"july"}, cfg)
    assert (nested / "sst.07").read_bytes() == b"july"


class _PerOutputAndTemplateConfig(ConfigModel):
    """Synthetic config with BOTH the explicit per-output field and the
    template — the explicit field must win."""

    model_config = ConfigDict(extra="forbid")

    sst_01_out: Annotated[
        Path, CLIArg("--sst-01-out", default=None), Field(description="")
    ] = Path("./sst.01")
    sst_out_template: Annotated[
        str, CLIArg("--sst-out-template", default=None), Field(description="")
    ] = "./from-template.{month:02d}"


def test_per_output_field_takes_precedence_over_template(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-jan"
    templated = tmp_path / "templated.{month:02d}"
    cfg = _PerOutputAndTemplateConfig(
        sst_01_out=explicit,
        sst_out_template=str(templated),
    )

    write_outputs({"sst_01": b"jan"}, cfg)

    assert explicit.read_bytes() == b"jan", "explicit path was not used"
    assert not (tmp_path / "templated.01").exists(), (
        "template path was written even though the per-output field existed"
    )


def test_missing_field_and_no_template_raises_attribute_error(tmp_path: Path) -> None:
    cfg = _TemplateOnlyConfig()  # only has sst_out_template
    # A returned name that neither matches <name>_out nor '<prefix>_out_template':
    with pytest.raises(AttributeError, match="programming error"):
        write_outputs({"lakedl": b"nope"}, cfg)


def test_template_only_matched_for_numeric_suffix(tmp_path: Path) -> None:
    """The template fallback only fires when the trailing part is all digits."""
    cfg = _TemplateOnlyConfig(sst_out_template=str(tmp_path / "sst.{month:02d}"))
    # 'sst_january' has a non-digit suffix; the template path must NOT fire,
    # and since there is no 'sst_january_out' field either, this raises.
    with pytest.raises(AttributeError):
        write_outputs({"sst_january": b"x"}, cfg)


# ---------------------------------------------------------------------------
# sea-surface config template validator
# ---------------------------------------------------------------------------


def _sea_surface_config():
    """Import lazily so a broken sea_surface import doesn't blow up
    the whole test module during collection."""
    from pproc.climate.generate.products.sea_surface import SeaSurfaceConfig

    return SeaSurfaceConfig


def test_sea_surface_template_default_contains_month_placeholder() -> None:
    SeaSurfaceConfig = _sea_surface_config()
    cfg = SeaSurfaceConfig(target_grid="N48")
    assert "{month" in cfg.sst_out_template


def test_sea_surface_template_missing_placeholder_rejected() -> None:
    SeaSurfaceConfig = _sea_surface_config()
    with pytest.raises(ValueError, match=r"\{month\}"):
        SeaSurfaceConfig(
            target_grid="N48",
            sst_out_template="./no-placeholder-here.grb",
        )


def test_sea_surface_template_padded_placeholder_accepted() -> None:
    SeaSurfaceConfig = _sea_surface_config()
    cfg = SeaSurfaceConfig(
        target_grid="N48",
        sst_out_template="./sst_{month:02d}.grb",
    )
    assert cfg.sst_out_template == "./sst_{month:02d}.grb"


def test_sea_surface_template_bare_placeholder_accepted() -> None:
    SeaSurfaceConfig = _sea_surface_config()
    cfg = SeaSurfaceConfig(
        target_grid="N48",
        sst_out_template="./sst_{month}.grb",
    )
    assert cfg.sst_out_template == "./sst_{month}.grb"
