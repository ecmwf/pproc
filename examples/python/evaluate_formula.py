"""Direct use of pproc.climate.field_calc.evaluate_formula on numpy arrays.

Use this pattern when:
- You already have numpy arrays in hand (no GRIB I/O needed).
- You want a string-based formula interface for ad-hoc arithmetic.
- You need to support user-supplied formulae (e.g. in a config file).

For GRIB I/O wrapping, use the ``pproc-field-calc`` CLI (see
``docs/climate/src/cli/field-calc.md``).
"""

import numpy as np

from pproc.climate.field_calc import evaluate_formula, parse_variables


def main() -> None:
    # Two synthetic fields
    rng = np.random.default_rng(42)
    a = rng.standard_normal(100).astype(np.float64)
    b = rng.standard_normal(100).astype(np.float64)

    # Single formula
    diff = evaluate_formula("a - b", {"a": a, "b": b})
    print(f"a - b: shape={diff.shape}, dtype={diff.dtype}")

    # Comparison returns 0/1 float
    mask = evaluate_formula("a > 0", {"a": a})
    print(f"a > 0: dtype={mask.dtype}, values are {sorted(set(mask.tolist()))}")

    # Composite expression
    hypot = evaluate_formula("sqrt(a^2 + b^2)", {"a": a, "b": b})
    print(f"sqrt(a^2+b^2) max={hypot.max():.4f}")

    # Variable list parsing
    names = parse_variables("a;b;c")
    print(f"parse_variables('a;b;c') -> {names}")


if __name__ == "__main__":
    main()
