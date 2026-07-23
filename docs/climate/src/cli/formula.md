# pproc-formula

> **Transitional.** `pproc-formula` is a standalone CLI that predates
> [`pproc-climate-fields`](climate-fields.md). It remains installed for
> callers that invoke it directly (a few legacy wrappers and ad-hoc
> shell pipelines). New code should not depend on this CLI; product
> modules under `pproc.climate.generate.products.*` call
> `pproc.formula.evaluate_formula` directly, and any new formula-based
> field belongs in a product module. This CLI will be removed once the
> remaining callers migrate.

`pproc-formula` evaluates arithmetic formulae over GRIB fields. It reads
one or more GRIB inputs, evaluates a formula (or a semicolon-separated
list of formulae), and writes a single output GRIB file with one message
per sub-formula. The implementation lives in
`pproc/src/pproc/formula_cli.py` and delegates to
`pproc.formula.evaluate_formula` for parsing and evaluation.

## Synopsis

```
pproc-formula --formula EXPR [--variables a;b;...] [--multi-dimensional N]
                 [--metadata KEY=VAL]...
                 INPUT [INPUT ...] OUTPUT
```

## Flags

| Flag | Argument | Description |
|------|----------|-------------|
| `--formula` | `EXPR` | **Required.** Formula expression. May contain `;` to separate multiple sub-formulae; each sub-formula produces one output GRIB message. |
| `--variables` | `a;b;...` | Semicolon-separated variable names in input order. Defaults to `f1, f2, ... fN`. |
| `--multi-dimensional` | `N` | Treat the (single) input file as `N` consecutive GRIB messages. Incompatible with multiple input files. |
| `--metadata` | `KEY=VAL` | A single `KEY=VAL` GRIB metadata override applied to the output. Repeat the flag for multiple overrides, e.g. `--metadata shortName=sdor --metadata packingType=grid_simple` (idiom follows `git -c`, `docker -e`). |
| `paths` | `INPUT_OR_OUTPUT [INPUT_OR_OUTPUT ...]` | One or more INPUT GRIB files followed by a single OUTPUT GRIB file. With `--multi-dimensional N`, exactly one INPUT is allowed. |
| `-v`, `--verbose` | — | Count flag. Absent: silent (WARNING). `-v`: INFO logging to stdout (per-sub-formula evaluation and start/end summary). `-vv`: DEBUG. See [Logging and verbosity](#logging-and-verbosity). |
| `-h`, `--help` | — | Show argparse help and exit. |

## Formula grammar

Lowest to highest precedence:

| Level | Operator(s) | Associativity | Notes |
|-------|-------------|---------------|-------|
| compare | `>` `<` `>=` `<=` `==` | non-chaining | Returns `float64` 0.0 / 1.0, **not** bool, so the result composes with subsequent arithmetic. |
| addsub | `+` `-` | left | |
| muldiv | `*` `/` | left | |
| power | `^` | right | `^` is **power** (`numpy.power`); never XOR. |
| unary | `-` | — | Binds tighter than `^`. Unary `+` is accepted as a no-op. |

**Functions**: `sqrt`, `abs`, `atan2(x, y)`, `min(x, y)`, `max(x, y)`.

**Constants**: `pi` evaluates to `numpy.pi`. Numeric literals support
integer, decimal, and exponent (`1.0e-8`) forms.

**Grouping**: parentheses `(...)`.

**Comparisons return float64 0/1.** This is deliberate; the legacy ksh
script relies on it (e.g. `(K - Lprime) > 0` is multiplied into a later
expression).

**Chained comparison** (e.g. `a < b < c`) is rejected as a parse error.

The parser is recursive-descent and lives in
`pproc.formula`. No `eval()` or `exec()` is used.

## Examples

The following invocations are taken from the test suite (see
`pproc/tests/test_formula_cli.py`):

### Two-input subtraction

```bash
pproc-formula --formula "f1 - f2" a.grib b.grib diff.grib
```

The default variable names are `f1, f2, ...` so two positional inputs
bind to `f1` and `f2`.

### With named variables

```bash
pproc-formula --variables "a;b" --formula "sqrt(a^2 + b^2)" \
                 a.grib b.grib hypot.grib
```

### Multi-dimensional input

A single GRIB file containing two consecutive messages (e.g. the
gradient output from `pproc-gradient`):

```bash
pproc-formula --variables "gx;gy" --formula "gx*gy" \
                 --multi-dimensional 2 grad.grib cross.grib
```

### Metadata override

```bash
pproc-formula --variables "var;lsm" --formula "sqrt(var) * lsm" \
                 --metadata shortName=sdor --metadata packingType=grid_simple \
                 variance.grib mask.grib stdgwd.grib
```

### Multiple outputs from one invocation

```bash
pproc-formula --formula "f1-f2;f1+f2" a.grib b.grib both.grib
```

The output file `both.grib` contains two GRIB messages, one per
sub-formula.

## Logging and verbosity

`pproc-formula` is silent by default; pass `-v` (or `--verbose`) to
surface per-sub-formula evaluation on **stdout**, or `-vv` for
timestamped DEBUG lines. A representative `-v` excerpt:

```text
[pproc.formula] pproc-formula start formula='f1 - f2' variables=['f1', 'f2'] inputs=['a.grib', 'b.grib'] output=diff.grib multi_dimensional=None
[pproc.formula] evaluating formula 1/1: f1 - f2
[pproc.formula] pproc-formula done elapsed=0.034
```

Verbosity is driven by argparse's `action='count'`, so `-vvv` and
above clamp to DEBUG.
