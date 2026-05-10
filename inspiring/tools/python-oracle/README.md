# `inspiring-oracle`

A small, deliberately slow, deliberately readable Python implementation of
**Algorithm 1 (`InspiRING.Pack`)** from the InsPIRe paper
([eprint 2025/1352](https://eprint.iacr.org/2025/1352)). Its only job is to be
*obviously correct* at tiny parameters (`d in {8, 16}`).

The Rust crate (built in Phases 4-9) loads JSON fixtures generated here and
asserts byte-equality on every captured intermediate. This package is therefore
the **byte-equal correctness ground truth** for the Rust implementation.

> See [../../SPEC.md](../../SPEC.md) for the mathematical contract this code
> implements verbatim. Every function, variable, and loop bound mirrors the
> notation used there.

## Build status

Implemented stage by stage; see
[`.cursor/plans/phase_2_python_oracle_*.plan.md`](../../../.cursor/plans/) for
the per-stage breakdown.

| Stage | Concept                              | Status |
| ----- | ------------------------------------ | ------ |
| 0     | Scaffold + `RlweParams`              | done   |
| 1     | Ring arithmetic `R_q`                | done   |
| 2     | Galois automorphisms `tau_g, tau_h`  | done   |
| 3     | Lemma 1: trace `Tr(p) = d * c_0`     | done   |
| 4     | LWE                                  | done   |
| 5     | RLWE under `s_tilde`                 | done   |
| 6     | Gadget decomposition                 | done   |
| 7     | Key switching                        | done   |
| 8     | `TRANSFORM` (LWE -> IRCtx)           | done   |
| 9     | `aggregate`                          | done   |
| 10    | `collapse_one`                       | done   |
| 11    | `collapse_half`                      | done   |
| 12    | `collapse` (full)                    | done   |
| 13    | `pack` (Algorithm 1)                 | done   |
| 14    | Empirical Theorem 2 noise check      | done   |
| 15    | JSON fixture generation              | done   |

**Phase 2 status**: complete. 534 tests pass under `make oracle-test`.
Generated fixtures live at [`../../fixtures/`](../../fixtures/) and are
checked into the repo as the cross-language contract for the future
Rust crate.

## Quickstart

This package is managed by [uv](https://docs.astral.sh/uv/). Install uv first
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
cd inspiring/tools/python-oracle
uv sync                                      # creates .venv, installs deps
uv run ruff check .                         # lint Python sources and tests
uv run pytest -v                             # run all tests
uv run pytest -v tests/test_params.py        # one stage at a time
```

To regenerate the JSON fixtures consumed by the Rust crate (Stage 15):

```bash
make oracle-fixtures               # writes to ../../fixtures/ from repo root
# or directly:
uv run python scripts/generate_fixtures.py --output ../../fixtures/
```

The generated set lives at [`inspiring/fixtures/`](../../fixtures/) and
contains a `MANIFEST.json` plus one self-contained JSON per fixture
(see `inspiring_oracle/fixtures.py` for the schema). Each fixture
captures every intermediate of one full `pack` execution so the Rust
crate can assert byte-equality on every stage boundary.

## Parameter presets

Defined in [`inspiring_oracle/params.py`](inspiring_oracle/params.py):

| preset           | `d` | `q`     | `p` | `sigma` | `z` | `ell` | `delta` | `sigma_pack` bound |
| ---------------- | --- | ------- | --- | ------- | --- | ----- | ------- | ------------------ |
| `ORACLE_TINY`    | 8   | 12289   | 4   | 3.2     | 8   | 5     | 3072    | ~72                |
| `ORACLE_SMALL`   | 16  | 65537   | 4   | 3.2     | 16  | 5     | 16384   | ~916               |

Both presets satisfy `6 * sigma_pack < delta / 2`, leaving an ample
correctness margin (per-coefficient decryption-failure probability under 1e-9).
`q = 12289 = 3 * 2^12 + 1` and `q = 65537 = 2^16 + 1` are standard
NTT-friendly primes used widely in lattice-based cryptography testing.

## Design rules (do not break)

The plan in [`.cursor/plans/phase_2_python_oracle_*.plan.md`](../../../.cursor/plans/)
spells these out, but the short version:

1. **One sympy idiom only.** Polynomials are stored as `list[int]` of length
   `d` with coefficients in `[0, q)`. Multiplication is naive `O(d^2)`
   schoolbook plus negacyclic reduction. No NTT, no `sympy.Poly`.
2. **Verbatim mapping to SPEC.md.** Every function name and variable name
   matches the symbol table in SPEC.md section 10.
3. **No premature noise hiding.** Every step exposes its noise vector so the
   noise test (Stage 14) and the eventual Rust noise test can both consume it.
4. **Deterministic randomness.** All randomness flows through one
   `random.Random(seed)` instance per fixture; the seed is recorded in JSON.

## Layout

```
tools/python-oracle/
|-- pyproject.toml
|-- README.md
|-- inspiring_oracle/
|   |-- __init__.py
|   |-- params.py          # Stage 0
|   |-- ring.py            # Stage 1
|   |-- automorph.py       # Stages 2, 3
|   |-- lwe.py             # Stage 4
|   |-- rlwe.py            # Stage 5
|   |-- gadget.py          # Stage 6
|   |-- key_switching.py   # Stage 7
|   |-- intermediate.py    # Stages 8, 9
|   |-- collapse.py        # Stages 10, 11, 12
|   |-- pack.py            # Stage 13
|   |-- decrypt_under_s_hat.py
|   `-- fixtures.py        # Stage 15
|-- tests/
|   |-- test_params.py     # Stage 0
|   |-- ...
|   `-- test_noise.py      # Stage 14
`-- scripts/
    `-- generate_fixtures.py # Stage 15

inspiring/fixtures/                  # generated JSON fixtures (committed)
|-- MANIFEST.json
|-- tiny_random_seed_42.json
|-- tiny_all_zero.json
|-- tiny_all_max.json
|-- tiny_single_one_at_3.json
|-- small_random_seed_42.json
|-- small_random_seed_1337.json
|-- small_all_zero.json
|-- small_all_max.json
`-- small_single_one_at_15.json
```
