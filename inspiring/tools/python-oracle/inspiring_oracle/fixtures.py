"""JSON fixture format consumed by the future Rust crate.

The Rust crate (built in Phases 4-9 per `SPEC.md` and the project plan)
needs to assert byte-equality against this oracle on every captured
intermediate. This module defines:

* The fixture file format (a single ``Fixture`` dataclass with strict
  field ordering and a ``schema_version``).
* (De)serialization to / from JSON.
* Self-check: ``fixture.verify()`` re-runs the algorithm at decode time
  and asserts the captured intermediates and final result are
  internally consistent.

Every fixture file captures one full ``pack`` execution at one of the
oracle parameter presets (``ORACLE_TINY`` or ``ORACLE_SMALL``). The
captured artefacts are everything a Rust implementation needs to:

1. Reconstruct the inputs (LWE secret, LWE ciphertexts, K_g, K_h).
2. Replay the algorithm step by step.
3. Assert byte-level match against the Python implementation at every
   stage boundary (transform, aggregate, collapse).
4. Verify the final RLWE ciphertext decrypts to the expected messages.

Format choice: human-readable JSON with explicit nested structure rather
than a binary or base64-encoded blob. The fixtures are small enough
(``d <= 16`` so each polynomial is at most 16 ints) that legibility
trumps size.

Versioning: the ``schema_version`` field protects against silent
breakage if the format ever changes. The current revision is ``1``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inspiring_oracle.intermediate import IRCtx, aggregate, transform
from inspiring_oracle.key_switching import KeySwitchingMatrix
from inspiring_oracle.lwe import LweCiphertext
from inspiring_oracle.params import RlweParams
from inspiring_oracle.pack import pack
from inspiring_oracle.rlwe import RlweCiphertext, decrypt

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Wire format dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureParams:
    """Mirror of ``RlweParams`` for serialization (sigma is float, the rest int)."""

    d: int
    q: int
    p: int
    sigma: float
    z: int
    ell: int

    @classmethod
    def from_params(cls, params: RlweParams) -> "FixtureParams":
        return cls(
            d=params.d,
            q=params.q,
            p=params.p,
            sigma=params.sigma,
            z=params.z,
            ell=params.ell,
        )

    def to_params(self) -> RlweParams:
        return RlweParams(
            d=self.d, q=self.q, p=self.p,
            sigma=self.sigma, z=self.z, ell=self.ell,
        )


@dataclass(frozen=True)
class FixtureKsMatrix:
    """Serializable form of ``KeySwitchingMatrix``. Includes ``noise``
    so the Rust crate can validate Stage 7's per-row noise extraction
    against the Python sample."""

    w: list[list[int]]
    y: list[list[int]]
    noise: list[list[int]]

    @classmethod
    def from_ks(cls, K: KeySwitchingMatrix) -> "FixtureKsMatrix":
        return cls(w=K.w, y=K.y, noise=K.noise)

    def to_ks(self) -> KeySwitchingMatrix:
        return KeySwitchingMatrix(w=self.w, y=self.y, noise=self.noise)


@dataclass(frozen=True)
class FixtureLwe:
    a: list[int]
    b: int

    @classmethod
    def from_lwe(cls, ct: LweCiphertext) -> "FixtureLwe":
        return cls(a=ct.a, b=ct.b)

    def to_lwe(self) -> LweCiphertext:
        return LweCiphertext(a=self.a, b=self.b)


@dataclass(frozen=True)
class FixtureIRCtx:
    a_hat: list[list[int]]
    b_tilde: list[int]

    @classmethod
    def from_irctx(cls, ictx: IRCtx) -> "FixtureIRCtx":
        return cls(a_hat=ictx.a_hat, b_tilde=ictx.b_tilde)

    def to_irctx(self) -> IRCtx:
        return IRCtx(a_hat=self.a_hat, b_tilde=self.b_tilde)


@dataclass(frozen=True)
class FixtureRlwe:
    c1: list[int]
    c2: list[int]

    @classmethod
    def from_rlwe(cls, ct: RlweCiphertext) -> "FixtureRlwe":
        return cls(c1=ct.c1, c2=ct.c2)

    def to_rlwe(self) -> RlweCiphertext:
        return RlweCiphertext(c1=self.c1, c2=self.c2)


@dataclass(frozen=True)
class Fixture:
    """One full captured ``pack`` execution.

    Field ordering is **stable** across schema_version 1 -- the Rust
    crate parses by field name, not position, but humans reading the
    JSON benefit from a predictable layout.
    """

    schema_version: int
    name: str
    description: str
    rng_seed: int

    params: FixtureParams

    # Inputs.
    s: list[int]                 # LWE secret (length d, ternary)
    s_tilde: list[int]           # RLWE secret (= s mod q, length d)
    K_g: FixtureKsMatrix
    K_h: FixtureKsMatrix
    messages: list[int]          # plaintext messages, length d
    lwes: list[FixtureLwe]       # input LWE ciphertexts, length d

    # Stage 1 outputs (one IRCtx per input LWE).
    transform_outputs: list[FixtureIRCtx]

    # Stage 2 output.
    aggregate_output: FixtureIRCtx

    # Stage 3 output -- the final packed RLWE ciphertext.
    packed: FixtureRlwe

    # Expected decryption result. Always equal to ``messages`` for a
    # well-formed fixture, but stored explicitly so a Rust crate can
    # assert directly against this without reconstructing.
    expected_decrypted: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Self-check -- re-runs the algorithm and verifies internal consistency
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Replay the algorithm and assert every captured artefact matches.

        Raises ``AssertionError`` on the first mismatch. Used both at
        fixture-generation time (to catch internal bugs early) and at
        load time (to detect corruption / schema drift).
        """
        if self.schema_version != SCHEMA_VERSION:
            raise AssertionError(
                f"fixture {self.name!r}: schema_version {self.schema_version} "
                f"!= current SCHEMA_VERSION {SCHEMA_VERSION}"
            )

        params = self.params.to_params()

        # Replay transform.
        lwes = [fl.to_lwe() for fl in self.lwes]
        replay_irctxs = [transform(ct, params) for ct in lwes]
        for k, (rp, captured) in enumerate(
            zip(replay_irctxs, self.transform_outputs, strict=True)
        ):
            assert rp.a_hat == captured.a_hat, (
                f"fixture {self.name!r}: transform[{k}].a_hat mismatch"
            )
            assert rp.b_tilde == captured.b_tilde, (
                f"fixture {self.name!r}: transform[{k}].b_tilde mismatch"
            )

        # Replay aggregate.
        replay_agg = aggregate(replay_irctxs, params)
        assert replay_agg.a_hat == self.aggregate_output.a_hat, (
            f"fixture {self.name!r}: aggregate.a_hat mismatch"
        )
        assert replay_agg.b_tilde == self.aggregate_output.b_tilde, (
            f"fixture {self.name!r}: aggregate.b_tilde mismatch"
        )

        # Replay full pack.
        K_g = self.K_g.to_ks()
        K_h = self.K_h.to_ks()
        replay_packed = pack(lwes, K_g, K_h, params)
        assert replay_packed.c1 == self.packed.c1, (
            f"fixture {self.name!r}: packed.c1 mismatch"
        )
        assert replay_packed.c2 == self.packed.c2, (
            f"fixture {self.name!r}: packed.c2 mismatch"
        )

        # Decryption sanity.
        decrypted = decrypt(self.s_tilde, replay_packed, params)
        assert decrypted == self.messages, (
            f"fixture {self.name!r}: decryption {decrypted} != messages "
            f"{self.messages}"
        )
        assert decrypted == self.expected_decrypted, (
            f"fixture {self.name!r}: expected_decrypted "
            f"{self.expected_decrypted} != actual {decrypted}"
        )

    # ------------------------------------------------------------------
    # JSON I/O
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialize to pretty-printed JSON (stable, sorted-key-free)."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Fixture":
        return cls.from_dict(json.loads(s))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Fixture":
        """Reconstruct from a parsed JSON dict.

        Strict: every nested dict is re-instantiated through its dataclass
        constructor, so missing or extra fields are caught by Python.
        """
        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            description=data["description"],
            rng_seed=data["rng_seed"],
            params=FixtureParams(**data["params"]),
            s=data["s"],
            s_tilde=data["s_tilde"],
            K_g=FixtureKsMatrix(**data["K_g"]),
            K_h=FixtureKsMatrix(**data["K_h"]),
            messages=data["messages"],
            lwes=[FixtureLwe(**lwe) for lwe in data["lwes"]],
            transform_outputs=[
                FixtureIRCtx(**ictx) for ictx in data["transform_outputs"]
            ],
            aggregate_output=FixtureIRCtx(**data["aggregate_output"]),
            packed=FixtureRlwe(**data["packed"]),
            expected_decrypted=data["expected_decrypted"],
        )

    def write(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> "Fixture":
        return cls.from_json(path.read_text())


# ---------------------------------------------------------------------------
# Manifest -- index of all fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    file: str
    name: str
    description: str
    params_preset: str  # "ORACLE_TINY" or "ORACLE_SMALL"


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    fixtures: list[ManifestEntry]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Manifest":
        data = json.loads(s)
        return cls(
            schema_version=data["schema_version"],
            fixtures=[ManifestEntry(**e) for e in data["fixtures"]],
        )

    def write(self, path: Path) -> None:
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        return cls.from_json(path.read_text())


# ---------------------------------------------------------------------------
# Builder -- run pack and capture every intermediate
# ---------------------------------------------------------------------------


def build_fixture(
    name: str,
    description: str,
    s: list[int],
    s_tilde: list[int],
    K_g: KeySwitchingMatrix,
    K_h: KeySwitchingMatrix,
    lwes: list[LweCiphertext],
    messages: list[int],
    rng_seed: int,
    params: RlweParams,
) -> Fixture:
    """Run ``pack`` end-to-end on the given inputs and capture every step.

    Internally re-runs ``transform`` and ``aggregate`` (alongside ``pack``)
    so each intermediate is captured for byte-equality assertion by the
    Rust crate. Validates the result via ``Fixture.verify`` before
    returning.
    """
    transform_outputs = [transform(ct, params) for ct in lwes]
    aggregate_output = aggregate(transform_outputs, params)
    packed = pack(lwes, K_g, K_h, params)
    decrypted = decrypt(s_tilde, packed, params)

    fixture = Fixture(
        schema_version=SCHEMA_VERSION,
        name=name,
        description=description,
        rng_seed=rng_seed,
        params=FixtureParams.from_params(params),
        s=s,
        s_tilde=s_tilde,
        K_g=FixtureKsMatrix.from_ks(K_g),
        K_h=FixtureKsMatrix.from_ks(K_h),
        messages=messages,
        lwes=[FixtureLwe.from_lwe(ct) for ct in lwes],
        transform_outputs=[
            FixtureIRCtx.from_irctx(ictx) for ictx in transform_outputs
        ],
        aggregate_output=FixtureIRCtx.from_irctx(aggregate_output),
        packed=FixtureRlwe.from_rlwe(packed),
        expected_decrypted=decrypted,
    )
    fixture.verify()
    return fixture
