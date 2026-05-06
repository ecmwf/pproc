"""Effective-resolution grid mapping for the SSO pipeline.

Direct migration of ``generate_subgrid_orography_sso.ksh`` lines 73-90:

.. code-block:: bash

    if [ "$GTYPE_SET" = "O" ] ; then
       ERES=$(( $ORES / 2 ))                      # truncating integer division
       ERES=$(( $ERES - ("$ERES % 2") ))          # round DOWN to even
       MIR_ERES_SET=N${ERES}
       if   [ $ERES = 40   ] ; then MIR_ERES_SET=N48
       elif [ $ERES = 100  ] ; then MIR_ERES_SET=N128
       elif [ $ERES = 1000 ] ; then MIR_ERES_SET=N1024
       fi
    else
       MIR_ERES_SET=${MIR_GTYPE_SET}
    fi

The two public helpers below preserve that behaviour byte-for-byte so the
operational pipeline can swap out the shell logic without functional drift.
"""

from __future__ import annotations

import re

__all__ = ["infer_grid_params", "compute_effective_resolution"]


# Strict matcher: single uppercase letter prefix + one-or-more digits, no
# whitespace, no sign. The ksh script always uses uppercase, so we mirror that.
_GRID_SPEC_RE = re.compile(r"^([A-Z])([0-9]+)$")

# Recognised grid family codes:
#   O = octahedral reduced Gaussian
#   N = (non-octahedral) reduced Gaussian
#   F = full Gaussian
_KNOWN_GRID_TYPES = frozenset({"O", "N", "F"})

# Special-case overrides from generate_subgrid_orography_sso.ksh lines 80-86:
# when ERES lands on 40, 100, or 1000, the operational pipeline uses
# N48/N128/N1024 rather than the literal N40/N100/N1000. These compensate
# for grid-mesh quality at those exact spectral truncations and are not
# derivable from a clean formula -- they are operational constants.
_ERES_SPECIAL_CASES: dict[int, str] = {
    40: "N48",
    100: "N128",
    1000: "N1024",
}


def infer_grid_params(target_grid: str) -> tuple[str, int]:
    """Extract ``(grid_type, resolution)`` from a grid-spec string.

    Examples
    --------
    >>> infer_grid_params("O1280")
    ('O', 1280)
    >>> infer_grid_params("N256")
    ('N', 256)
    >>> infer_grid_params("F128")
    ('F', 128)

    Parameters
    ----------
    target_grid:
        A grid spec of the form ``<LETTER><DIGITS>`` (e.g. ``"O1280"``).
        The letter must be one of ``O``, ``N`` or ``F`` (uppercase).
        The digits must form a strictly-positive integer.

    Returns
    -------
    tuple[str, int]
        The grid family code and the integer resolution.

    Raises
    ------
    ValueError
        If ``target_grid`` is empty, malformed (e.g. ``"O"``, ``"OO80"``,
        ``"X100"``, ``"O-1"``, ``"1280"``), uses a lowercase prefix
        (``"o80"``), or has a non-positive resolution (``"O0"``).
        The exception message includes the offending input.
    """
    if not isinstance(target_grid, str):
        raise ValueError(
            f"target_grid must be a string, got {type(target_grid).__name__}"
        )
    match = _GRID_SPEC_RE.match(target_grid)
    if match is None:
        raise ValueError(
            f"Malformed grid spec {target_grid!r}: expected '<LETTER><DIGITS>' "
            f"with an uppercase letter prefix (e.g. 'O1280', 'N256', 'F128')."
        )
    letter, digits = match.group(1), match.group(2)
    if letter not in _KNOWN_GRID_TYPES:
        raise ValueError(
            f"Unknown grid type {letter!r} in {target_grid!r}: "
            f"expected one of {sorted(_KNOWN_GRID_TYPES)}."
        )
    resolution = int(digits)
    if resolution <= 0:
        raise ValueError(
            f"Grid resolution must be positive, got {resolution} in {target_grid!r}."
        )
    return letter, resolution


def compute_effective_resolution(grid_type: str, resolution: int) -> str:
    """Compute the effective-resolution grid for the SSO pipeline.

    For octahedral grids (``grid_type == "O"``) this reproduces the ksh
    arithmetic exactly:

    .. code-block:: text

        eres = (resolution // 2)            # truncating integer division
        eres = eres - (eres % 2)            # round DOWN to even
        result = 'N{eres}'                  # with special-case overrides

    Special-case overrides (from the ksh script):

    =====  ========
    eres   result
    =====  ========
    40     ``N48``
    100    ``N128``
    1000   ``N1024``
    =====  ========

    For non-octahedral grids the model grid is returned unchanged, e.g.
    ``("N", 256) -> "N256"``. This matches the ksh ``else`` branch
    (``MIR_ERES_SET = MIR_GTYPE_SET``).

    Parameters
    ----------
    grid_type:
        Grid family code (``"O"``, ``"N"`` or ``"F"`` typically).
    resolution:
        Strictly-positive integer resolution.

    Returns
    -------
    str
        The effective-resolution grid spec (e.g. ``"N48"``).

    Raises
    ------
    ValueError
        If ``resolution`` is not strictly positive.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be a positive integer, got {resolution}.")

    if grid_type != "O":
        # Non-octahedral: ERES stage runs but the grid is the model grid.
        return f"{grid_type}{resolution}"

    # Octahedral path: match ksh `$(( $ORES / 2 ))` (truncating int division)
    # and `$(( $ERES - ("$ERES % 2") ))` (round DOWN to even).
    eres = resolution // 2
    eres -= eres % 2

    return _ERES_SPECIAL_CASES.get(eres, f"N{eres}")
