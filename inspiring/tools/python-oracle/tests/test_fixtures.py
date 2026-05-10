"""Tests for the JSON fixture format and self-check infrastructure.

These tests guard the contract between the Python oracle and the future
Rust crate: every fixture round-trips losslessly through JSON, every
fixture's captured intermediates re-derive on replay, and the recipe
script produces a valid manifest.

Test groups
-----------

1. **TestFixtureRoundtrip** -- a freshly built fixture serializes to
   JSON, parses back, and re-verifies. Pins down the schema and the
   ``Fixture.from_dict`` round-trip.

2. **TestFixtureCorruptionDetection** -- mutating any captured
   intermediate after construction makes ``verify`` raise, so the Rust
   crate is guaranteed to reject silently-corrupted fixtures.

3. **TestGenerateScriptEndToEnd** -- runs the actual fixture generator
   into a tmp_path and asserts every produced file parses, verifies, and
   appears in MANIFEST.json.

4. **TestManifestRoundtrip** -- serialize / parse the manifest itself.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from inspiring_oracle import key_switching, lwe, rlwe
from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.fixtures import (
    SCHEMA_VERSION,
    Fixture,
    Manifest,
    ManifestEntry,
    build_fixture,
)
from inspiring_oracle.params import ORACLE_TINY, RlweParams


def _build_minimal_fixture(rng_seed: int = 42) -> Fixture:
    """Quickly build a fixture at ORACLE_TINY -- shared by several tests."""
    params: RlweParams = ORACLE_TINY
    rng = random.Random(rng_seed)
    s = lwe.keygen(params, rng)
    s_tilde = rlwe.s_tilde_from_s(s, params)
    K_g = key_switching.setup(tau(s_tilde, G, params.q), s_tilde, params, rng)
    K_h = key_switching.setup(
        tau(s_tilde, h(params.d), params.q), s_tilde, params, rng
    )
    messages = [rng.randrange(params.p) for _ in range(params.d)]
    lwes = [lwe.encrypt(s, m, params, rng) for m in messages]
    return build_fixture(
        name="test_fixture",
        description="In-process fixture used by test_fixtures.py.",
        s=s,
        s_tilde=s_tilde,
        K_g=K_g,
        K_h=K_h,
        lwes=lwes,
        messages=messages,
        rng_seed=rng_seed,
        params=params,
    )


# ---------------------------------------------------------------------------
# 1. Fixture round-trip through JSON
# ---------------------------------------------------------------------------


class TestFixtureRoundtrip:
    def test_to_json_parses_back(self) -> None:
        fixture = _build_minimal_fixture()
        json_text = fixture.to_json()
        roundtripped = Fixture.from_json(json_text)
        assert roundtripped == fixture

    def test_roundtripped_fixture_self_verifies(self) -> None:
        fixture = _build_minimal_fixture()
        roundtripped = Fixture.from_json(fixture.to_json())
        roundtripped.verify()

    def test_write_then_read(self, tmp_path: Path) -> None:
        fixture = _build_minimal_fixture()
        path = tmp_path / "fixture.json"
        fixture.write(path)
        loaded = Fixture.read(path)
        assert loaded == fixture
        loaded.verify()

    def test_schema_version_present(self) -> None:
        fixture = _build_minimal_fixture()
        as_dict = json.loads(fixture.to_json())
        assert as_dict["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. Verify catches corruption
# ---------------------------------------------------------------------------


class TestFixtureCorruptionDetection:
    """Mutating any captured intermediate must make ``verify`` raise.

    This is the property that makes JSON fixtures a usable
    cross-language contract: a Rust crate that loads a fixture and
    diffs against its own re-derivation is guaranteed to detect any
    mutation, however small.
    """

    def _mutate_first_coeff(
        self, fixture: Fixture, field_path: str
    ) -> Fixture:
        """Return a new fixture with one coefficient flipped along field_path."""
        data = json.loads(fixture.to_json())
        # Walk the path -- field_path uses dot notation for nested lookup.
        parts = field_path.split(".")
        node = data
        for part in parts[:-1]:
            node = node[part]
        # The final part should be a list-of-lists; bump its [0][0] entry.
        last = parts[-1]
        target = node[last]
        # Mutation strategy depends on the shape.
        if isinstance(target, list) and target and isinstance(target[0], list):
            target[0][0] = (target[0][0] + 1) % fixture.params.q
        elif isinstance(target, list):
            target[0] = (target[0] + 1) % fixture.params.q
        else:
            raise AssertionError(f"don't know how to mutate {field_path}")
        return Fixture.from_dict(data)

    def test_corrupt_aggregate_a_hat_detected(self) -> None:
        fixture = _build_minimal_fixture()
        bad = self._mutate_first_coeff(fixture, "aggregate_output.a_hat")
        with pytest.raises(AssertionError, match="aggregate"):
            bad.verify()

    def test_corrupt_aggregate_b_tilde_detected(self) -> None:
        fixture = _build_minimal_fixture()
        bad = self._mutate_first_coeff(fixture, "aggregate_output.b_tilde")
        with pytest.raises(AssertionError, match="aggregate"):
            bad.verify()

    def test_corrupt_packed_c1_detected(self) -> None:
        fixture = _build_minimal_fixture()
        bad = self._mutate_first_coeff(fixture, "packed.c1")
        with pytest.raises(AssertionError, match="packed.c1"):
            bad.verify()

    def test_corrupt_packed_c2_detected(self) -> None:
        fixture = _build_minimal_fixture()
        bad = self._mutate_first_coeff(fixture, "packed.c2")
        with pytest.raises(AssertionError, match="packed.c2"):
            bad.verify()

    def test_wrong_schema_version_detected(self) -> None:
        fixture = _build_minimal_fixture()
        data = json.loads(fixture.to_json())
        data["schema_version"] = SCHEMA_VERSION + 1
        bad = Fixture.from_dict(data)
        with pytest.raises(AssertionError, match="schema_version"):
            bad.verify()


# ---------------------------------------------------------------------------
# 3. End-to-end script run
# ---------------------------------------------------------------------------


class TestGenerateScriptEndToEnd:
    """Run the actual generate_fixtures.py script and validate its output."""

    def test_script_produces_valid_manifest_and_fixtures(
        self, tmp_path: Path
    ) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "generate_fixtures.py"
        )
        out = tmp_path / "fixtures"
        result = subprocess.run(
            [sys.executable, str(script), "--output", str(out)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"script failed: stderr=\n{result.stderr}\nstdout=\n{result.stdout}"
        )

        manifest = Manifest.read(out / "MANIFEST.json")
        assert manifest.schema_version == SCHEMA_VERSION
        assert len(manifest.fixtures) >= 4, "expected at least 4 fixtures"

        for entry in manifest.fixtures:
            path = out / entry.file
            assert path.exists(), f"manifest references missing file {entry.file}"
            fixture = Fixture.read(path)
            assert fixture.name == entry.name
            fixture.verify()  # guaranteed to pass since build_fixture ran it

    def test_script_is_deterministic(self, tmp_path: Path) -> None:
        """Running twice must produce byte-identical fixtures."""
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "generate_fixtures.py"
        )
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        for out in (out_a, out_b):
            result = subprocess.run(
                [sys.executable, str(script), "--output", str(out)],
                check=True,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
        files_a = sorted(out_a.glob("*.json"))
        files_b = sorted(out_b.glob("*.json"))
        assert [p.name for p in files_a] == [p.name for p in files_b]
        for fa, fb in zip(files_a, files_b, strict=True):
            assert fa.read_text() == fb.read_text(), (
                f"fixture {fa.name} differs between runs"
            )


# ---------------------------------------------------------------------------
# 4. Manifest round-trip
# ---------------------------------------------------------------------------


class TestManifestRoundtrip:
    def test_to_json_parses_back(self) -> None:
        manifest = Manifest(
            schema_version=SCHEMA_VERSION,
            fixtures=[
                ManifestEntry(
                    file="a.json",
                    name="a",
                    description="x",
                    params_preset="ORACLE_TINY",
                ),
                ManifestEntry(
                    file="b.json",
                    name="b",
                    description="y",
                    params_preset="ORACLE_SMALL",
                ),
            ],
        )
        roundtripped = Manifest.from_json(manifest.to_json())
        assert roundtripped == manifest

    def test_write_then_read(self, tmp_path: Path) -> None:
        manifest = Manifest(schema_version=SCHEMA_VERSION, fixtures=[])
        path = tmp_path / "MANIFEST.json"
        manifest.write(path)
        assert Manifest.read(path) == manifest
