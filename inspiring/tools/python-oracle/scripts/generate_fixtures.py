"""Regenerate the JSON fixtures consumed by the future Rust crate.

Run from the project root::

    make oracle-fixtures
    # or, equivalently:
    cd tools/python-oracle
    uv run python scripts/generate_fixtures.py --output ../../fixtures/

Each invocation produces a deterministic set of fixtures (the RNG seeds
are baked into the manifest below). Running again should produce
byte-identical output -- the script aborts with a non-zero exit code if
any fixture fails its self-check.

The fixtures landing zone lives **outside** ``tools/python-oracle/`` so
the future Rust crate (``inspiring/`` proper) can read them via a
relative path (``../fixtures/`` from the crate root). The default
``--output`` argument matches that layout.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Ensure the package is importable when run as a script (uv handles this
# automatically through the venv but a defensive sys.path tweak makes
# direct ``python scripts/generate_fixtures.py`` work too).
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from inspiring_oracle import key_switching, lwe, rlwe  # noqa: E402
from inspiring_oracle.automorph import G, h, tau  # noqa: E402
from inspiring_oracle.fixtures import (  # noqa: E402
    Manifest,
    ManifestEntry,
    SCHEMA_VERSION,
    build_fixture,
)
from inspiring_oracle.params import (  # noqa: E402
    ORACLE_SMALL,
    ORACLE_TINY,
    RlweParams,
)


@dataclass(frozen=True)
class _FixtureRecipe:
    """Declarative recipe -- specifies what to generate, not how."""

    name: str
    description: str
    preset_name: str
    params: RlweParams
    rng_seed: int
    messages_kind: str  # "random" | "all_zero" | "all_max" | "single_one"
    single_one_index: int = 0  # only used when messages_kind == "single_one"


# The canonical fixture set. Bumping this list means bumping the manifest
# the Rust crate iterates over. New entries should pick fresh, distinctive
# seeds so independence between fixtures is obvious from a glance.
_RECIPES: list[_FixtureRecipe] = [
    _FixtureRecipe(
        name="tiny_random_seed_42",
        description="ORACLE_TINY (d=8) with random messages and seed 42.",
        preset_name="ORACLE_TINY",
        params=ORACLE_TINY,
        rng_seed=42,
        messages_kind="random",
    ),
    _FixtureRecipe(
        name="tiny_all_zero",
        description="ORACLE_TINY with all-zero messages -- noise-only output.",
        preset_name="ORACLE_TINY",
        params=ORACLE_TINY,
        rng_seed=0xA110_2E20,
        messages_kind="all_zero",
    ),
    _FixtureRecipe(
        name="tiny_all_max",
        description="ORACLE_TINY with messages = p - 1 in every slot.",
        preset_name="ORACLE_TINY",
        params=ORACLE_TINY,
        rng_seed=0xA110_AA01,
        messages_kind="all_max",
    ),
    _FixtureRecipe(
        name="tiny_single_one_at_3",
        description="ORACLE_TINY with only slot 3 nonzero (p - 1).",
        preset_name="ORACLE_TINY",
        params=ORACLE_TINY,
        rng_seed=0xA110_5103,
        messages_kind="single_one",
        single_one_index=3,
    ),
    _FixtureRecipe(
        name="small_random_seed_42",
        description="ORACLE_SMALL (d=16) with random messages and seed 42.",
        preset_name="ORACLE_SMALL",
        params=ORACLE_SMALL,
        rng_seed=42,
        messages_kind="random",
    ),
    _FixtureRecipe(
        name="small_random_seed_1337",
        description="ORACLE_SMALL with a different random message set.",
        preset_name="ORACLE_SMALL",
        params=ORACLE_SMALL,
        rng_seed=1337,
        messages_kind="random",
    ),
    _FixtureRecipe(
        name="small_all_zero",
        description="ORACLE_SMALL with all-zero messages.",
        preset_name="ORACLE_SMALL",
        params=ORACLE_SMALL,
        rng_seed=0xB220_2E20,
        messages_kind="all_zero",
    ),
    _FixtureRecipe(
        name="small_all_max",
        description="ORACLE_SMALL with messages = p - 1 in every slot.",
        preset_name="ORACLE_SMALL",
        params=ORACLE_SMALL,
        rng_seed=0xB220_AA01,
        messages_kind="all_max",
    ),
    _FixtureRecipe(
        name="small_single_one_at_15",
        description="ORACLE_SMALL with only the highest slot nonzero.",
        preset_name="ORACLE_SMALL",
        params=ORACLE_SMALL,
        rng_seed=0xB220_510F,
        messages_kind="single_one",
        single_one_index=15,
    ),
]


def _build_messages(recipe: _FixtureRecipe, rng: random.Random) -> list[int]:
    p, d = recipe.params.p, recipe.params.d
    if recipe.messages_kind == "random":
        return [rng.randrange(p) for _ in range(d)]
    if recipe.messages_kind == "all_zero":
        return [0] * d
    if recipe.messages_kind == "all_max":
        return [p - 1] * d
    if recipe.messages_kind == "single_one":
        msgs = [0] * d
        msgs[recipe.single_one_index % d] = p - 1
        return msgs
    raise ValueError(f"unknown messages_kind: {recipe.messages_kind!r}")


def _generate_one(recipe: _FixtureRecipe) -> tuple[ManifestEntry, str]:
    """Build one fixture, return its manifest entry and serialised JSON."""
    params = recipe.params
    rng = random.Random(recipe.rng_seed)

    s = lwe.keygen(params, rng)
    s_tilde = rlwe.s_tilde_from_s(s, params)
    K_g = key_switching.setup(tau(s_tilde, G, params.q), s_tilde, params, rng)
    K_h = key_switching.setup(
        tau(s_tilde, h(params.d), params.q), s_tilde, params, rng
    )

    messages = _build_messages(recipe, rng)
    lwes = [lwe.encrypt(s, m, params, rng) for m in messages]

    fixture = build_fixture(
        name=recipe.name,
        description=recipe.description,
        s=s,
        s_tilde=s_tilde,
        K_g=K_g,
        K_h=K_h,
        lwes=lwes,
        messages=messages,
        rng_seed=recipe.rng_seed,
        params=params,
    )

    entry = ManifestEntry(
        file=f"{recipe.name}.json",
        name=recipe.name,
        description=recipe.description,
        params_preset=recipe.preset_name,
    )
    return entry, fixture.to_json()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate JSON fixtures for the inspiring Rust crate."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "fixtures",
        help=(
            "Destination directory (default: <repo>/inspiring/fixtures/). "
            "Will be created if missing."
        ),
    )
    args = parser.parse_args()

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing fixtures to {out_dir}")
    entries: list[ManifestEntry] = []
    for recipe in _RECIPES:
        entry, json_text = _generate_one(recipe)
        (out_dir / entry.file).write_text(json_text)
        entries.append(entry)
        print(f"  wrote {entry.file}  ({recipe.preset_name})")

    manifest = Manifest(schema_version=SCHEMA_VERSION, fixtures=entries)
    (out_dir / "MANIFEST.json").write_text(manifest.to_json())
    print(f"  wrote MANIFEST.json with {len(entries)} entries")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
