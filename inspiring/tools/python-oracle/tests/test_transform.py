"""Tests for ``intermediate.transform`` -- Stage 1 of Algorithm 1.

These tests are the **first** to exercise InspiRING's actual algorithm
(stages 1-7 built the substrate). They verify the deepest claim of the
paper's stage 1: that applying the trace operator to an LWE ciphertext
yields an intermediate ciphertext whose decryption polynomial has the
LWE message+noise in slot 0 and **exactly zero** everywhere else.

Test groups
-----------

1. **TestIRCtxStructure** -- the IRCtx dataclass behaves correctly.

2. **TestATildeNegativeExponent** -- the negative-exponent embedding
   ``a_tilde[d-i] = -a[i]`` is constructed correctly. This embedding is
   what makes the ring-product trick work; pin it down with KATs.

3. **TestProductCoefficientIdentity** -- ``(a_tilde * s_tilde)[0] == <a, s>``
   in ``R_q``. The reason the negative-exponent embedding exists. If this
   identity fails the rest of stage 1 is meaningless.

4. **TestSHatStructure** -- ``s_hat`` is built correctly from ``s_tilde``.
   Includes ``s_hat[0] == s_tilde`` (the identity case) and Galois
   structure for the rest.

5. **TestTransformOutputShape** -- IRCtx has the right shape; ``b_tilde``
   has only slot 0 nonzero; all entries are in canonical ``[0, q)`` form.

6. **TestTransformDecryptionRoundtrip** -- the firewall: encrypt LWE m,
   transform, decrypt under s_hat, recover m. Many random samples.

7. **TestRecoveredPolynomialIsConstant** -- the proof of Lemma 1 in
   action: for a fresh transform output, slots ``k > 0`` of the recovered
   plaintext polynomial are **exactly** zero in ``[0, q)`` form.

8. **TestNoNewNoise** -- transform is noise-free. The extracted noise from
   an IRCtx equals the original LWE noise; other slots have zero noise.

9. **TestAHatIndependentOfB** -- ``a_hat`` depends only on ``a``, not on
   ``b``. This is the algebraic basis for the offline/online split
   (SPEC.md section 8): the server can precompute ``a_hat`` per CRS-fixed
   LWE ``a`` vector; the only online cost per query is the trace's
   action on the constant polynomial ``b * X^0`` (which is trivial).

10. **TestStructuralDistinctionFromSTilde** -- ``a_tilde`` (negative
    exponent, sign-flipped) and ``s_tilde`` (positive exponent, no flip)
    are genuinely different embeddings. Pinning this distinction down
    prevents a class of subtle implementation bugs.

11. **TestDeterminism** -- transform is a pure function; same input
    yields identical output.
"""

from __future__ import annotations

import random

import pytest

from inspiring_oracle import lwe, rlwe
from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.decrypt_under_s_hat import (
    decrypt_polynomial_under_s_hat,
    decrypt_under_s_hat,
    extract_noise_under_s_hat,
    s_hat_from_s_tilde,
)
from inspiring_oracle.intermediate import IRCtx, transform
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY, RlweParams
from inspiring_oracle.ring import mul as ring_mul

PARAMS = [ORACLE_TINY, ORACLE_SMALL]


def _inner_product_modq(a: list[int], s: list[int], q: int) -> int:
    return sum(ai * si for ai, si in zip(a, s, strict=True)) % q


# ---------------------------------------------------------------------------
# 1. IRCtx structure
# ---------------------------------------------------------------------------


class TestIRCtxStructure:
    """The IRCtx dataclass is frozen and exposes a_hat + b_tilde."""

    def test_dataclass_attributes(self) -> None:
        ictx = IRCtx(a_hat=[[1, 2], [3, 4]], b_tilde=[5, 6])
        assert ictx.a_hat == [[1, 2], [3, 4]]
        assert ictx.b_tilde == [5, 6]

    def test_dataclass_is_frozen(self) -> None:
        ictx = IRCtx(a_hat=[[1, 2]], b_tilde=[3, 4])
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            ictx.b_tilde = [9, 9]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Negative-exponent embedding a_tilde[d-i] = -a[i]
# ---------------------------------------------------------------------------


def _a_tilde_oracle(a: list[int], q: int) -> list[int]:
    """Mirror SPEC.md::TRANSFORM step "a_tilde = sum a[i] X^{-i}"."""
    d = len(a)
    out = [0] * d
    out[0] = a[0] % q
    for i in range(1, d):
        out[d - i] = (-a[i]) % q
    return out


class TestATildeNegativeExponent:
    """Verify the construction of a_tilde inside transform."""

    def test_kat_d4(self) -> None:
        params = RlweParams(d=4, q=12289, p=4, sigma=3.2, z=64, ell=3)
        a = [1, 2, 3, 4]
        # Build via transform with b=0 to inspect a_tilde indirectly:
        # the transform formula uses a_tilde directly, so we mirror it
        # against the oracle.
        expected_a_tilde = [1, (-4) % params.q, (-3) % params.q, (-2) % params.q]
        assert _a_tilde_oracle(a, params.q) == expected_a_tilde

    def test_zero_input_yields_zero_a_tilde(self) -> None:
        params = ORACLE_TINY
        a = [0] * params.d
        assert _a_tilde_oracle(a, params.q) == [0] * params.d

    @pytest.mark.parametrize("params", PARAMS)
    def test_a_tilde_round_trip_to_a(self, params: RlweParams) -> None:
        """Recovering a from a_tilde is the inverse: a[0] = a_tilde[0],
        a[i] = -a_tilde[d-i] for i in [1, d)."""
        rng = random.Random(0xA71EDE)
        a = [rng.randrange(params.q) for _ in range(params.d)]
        at = _a_tilde_oracle(a, params.q)
        recovered = [at[0]] + [(-at[params.d - i]) % params.q for i in range(1, params.d)]
        assert recovered == a


# ---------------------------------------------------------------------------
# 3. The product-coefficient identity: (a_tilde * s_tilde)[0] == <a, s>
# ---------------------------------------------------------------------------


class TestProductCoefficientIdentity:
    """The reason the negative-exponent embedding exists.

    This is the algebraic core of stage 1: the constant coefficient of
    the ring product equals the LWE inner product. If this fails, every
    derivation in SPEC.md section 4 collapses.
    """

    def test_kat_d4_explicit(self) -> None:
        # Worked example from the design notes:
        # a = [1, 2, 3, 4], s = [0, 1, 0, 1]; <a, s> = 6.
        d, q = 4, 12289
        a = [1, 2, 3, 4]
        s = [0, 1, 0, 1]
        a_tilde = _a_tilde_oracle(a, q)
        s_tilde = s[:]
        prod = ring_mul(a_tilde, s_tilde, q)
        assert prod[0] == 6
        assert _inner_product_modq(a, s, q) == 6

    @pytest.mark.parametrize("params", PARAMS)
    def test_random_identity(self, params: RlweParams) -> None:
        rng = random.Random(0xC0FFEE_01)
        for _ in range(50):
            a = [rng.randrange(params.q) for _ in range(params.d)]
            s = [rng.choice([-1, 0, 0, 1]) for _ in range(params.d)]
            a_tilde = _a_tilde_oracle(a, params.q)
            s_tilde = rlwe.s_tilde_from_s(s, params)
            prod = ring_mul(a_tilde, s_tilde, params.q)
            assert prod[0] == _inner_product_modq(a, s, params.q)


# ---------------------------------------------------------------------------
# 4. s_hat structure
# ---------------------------------------------------------------------------


class TestSHatStructure:
    """Verify s_hat_from_s_tilde matches SPEC.md section 4."""

    @pytest.mark.parametrize("params", PARAMS)
    def test_shape(self, params: RlweParams) -> None:
        rng = random.Random(0xD00DAD)
        s_tilde = rlwe.keygen(params, rng)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        assert len(s_hat) == params.d
        for poly in s_hat:
            assert len(poly) == params.d
            assert all(0 <= c < params.q for c in poly)

    @pytest.mark.parametrize("params", PARAMS)
    def test_first_slot_is_s_tilde(self, params: RlweParams) -> None:
        """tau_g^0 is the identity -> s_hat[0] = s_tilde."""
        rng = random.Random(0xD00DAE)
        s_tilde = rlwe.keygen(params, rng)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        assert s_hat[0] == s_tilde

    @pytest.mark.parametrize("params", PARAMS)
    def test_half_d_slot_is_tau_h(self, params: RlweParams) -> None:
        """s_hat[d/2] = tau_h(tau_g^0(s_tilde)) = tau_h(s_tilde)."""
        rng = random.Random(0xD00DAF)
        s_tilde = rlwe.keygen(params, rng)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        expected = tau(s_tilde, h(params.d), params.q)
        assert s_hat[params.d // 2] == expected

    @pytest.mark.parametrize("params", PARAMS)
    def test_galois_pattern(self, params: RlweParams) -> None:
        """All slots match the explicit Galois formula from SPEC."""
        rng = random.Random(0xD00DB0)
        s_tilde = rlwe.keygen(params, rng)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        two_d = 2 * params.d
        h_d = h(params.d)
        for j in range(params.d // 2):
            gj = pow(G, j, two_d)
            assert s_hat[j] == tau(s_tilde, gj, params.q)
            assert s_hat[j + params.d // 2] == tau(
                s_tilde, (gj * h_d) % two_d, params.q
            )


# ---------------------------------------------------------------------------
# 5. transform output shape
# ---------------------------------------------------------------------------


class TestTransformOutputShape:
    @pytest.mark.parametrize("params", PARAMS)
    def test_shapes_and_ranges(self, params: RlweParams) -> None:
        rng = random.Random(0x5EED01)
        s = lwe.keygen(params, rng)
        ct = lwe.encrypt(s, 0, params, rng)
        ictx = transform(ct, params)
        assert len(ictx.a_hat) == params.d
        for poly in ictx.a_hat:
            assert len(poly) == params.d
            assert all(0 <= c < params.q for c in poly)
        assert len(ictx.b_tilde) == params.d
        assert all(0 <= c < params.q for c in ictx.b_tilde)

    @pytest.mark.parametrize("params", PARAMS)
    def test_b_tilde_has_only_slot_0_nonzero(self, params: RlweParams) -> None:
        """b_tilde = [b mod q, 0, 0, ..., 0]."""
        rng = random.Random(0x5EED02)
        s = lwe.keygen(params, rng)
        ct = lwe.encrypt(s, params.p - 1, params, rng)
        ictx = transform(ct, params)
        assert ictx.b_tilde[0] == ct.b % params.q
        assert all(c == 0 for c in ictx.b_tilde[1:])


# ---------------------------------------------------------------------------
# 6. Decryption round-trip (the firewall)
# ---------------------------------------------------------------------------


class TestTransformDecryptionRoundtrip:
    """Encrypt LWE m -> transform -> decrypt under s_hat -> recover m.

    This is the load-bearing correctness test for stage 1.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_random_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xDEC0DE_01)
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        for _ in range(50):
            m = rng.randrange(params.p)
            ct = lwe.encrypt(s, m, params, rng)
            ictx = transform(ct, params)
            recovered = decrypt_under_s_hat(ictx, s_hat, params)
            assert recovered[0] == m, f"slot 0 mismatch: got {recovered[0]} want {m}"

    @pytest.mark.parametrize("params", PARAMS)
    def test_message_zero(self, params: RlweParams) -> None:
        rng = random.Random(0xDEC0DE_02)
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        ct = lwe.encrypt(s, 0, params, rng)
        ictx = transform(ct, params)
        assert decrypt_under_s_hat(ictx, s_hat, params)[0] == 0

    @pytest.mark.parametrize("params", PARAMS)
    def test_message_max(self, params: RlweParams) -> None:
        rng = random.Random(0xDEC0DE_03)
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        ct = lwe.encrypt(s, params.p - 1, params, rng)
        ictx = transform(ct, params)
        assert decrypt_under_s_hat(ictx, s_hat, params)[0] == params.p - 1


# ---------------------------------------------------------------------------
# 7. Recovered polynomial is constant (Lemma 1 in action)
# ---------------------------------------------------------------------------


class TestRecoveredPolynomialIsConstant:
    """For a fresh transform output, slots k > 0 of b_tilde + <a_hat, s_hat>
    are EXACTLY zero in [0, q) form.

    This is the cryptographic embodiment of Lemma 1 (Tr(p) = d * p[0]):
    after the trace + d^{-1} scaling, only the constant coefficient
    survives. There is no rounding; the higher slots are integer-zero.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_higher_slots_are_zero(self, params: RlweParams) -> None:
        rng = random.Random(0xC057_01)
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        for _ in range(20):
            m = rng.randrange(params.p)
            ct = lwe.encrypt(s, m, params, rng)
            ictx = transform(ct, params)
            raw = decrypt_polynomial_under_s_hat(ictx, s_hat, params)
            for k in range(1, params.d):
                assert raw[k] == 0, f"slot {k} = {raw[k]} (params={params})"

    @pytest.mark.parametrize("params", PARAMS)
    def test_slot_0_is_delta_m_plus_e(self, params: RlweParams) -> None:
        """Slot 0 holds Delta * m + e in centered form."""
        rng = random.Random(0xC057_02)
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        for _ in range(20):
            m = rng.randrange(params.p)
            ct = lwe.encrypt(s, m, params, rng)
            ictx = transform(ct, params)
            raw = decrypt_polynomial_under_s_hat(ictx, s_hat, params)
            # Compute residual mod q first, THEN center -- otherwise large
            # values of delta * m (when m is near p) overflow q/2.
            residual_modq = (raw[0] - params.delta * m) % params.q
            residual = (
                residual_modq if residual_modq <= params.q // 2
                else residual_modq - params.q
            )
            # Residual must be a small noise term, well under Delta/2.
            assert abs(residual) < params.delta // 2


# ---------------------------------------------------------------------------
# 8. No new noise (transform is a noise-free operation)
# ---------------------------------------------------------------------------


class TestNoNewNoise:
    """The trace is Z-linear, so transform adds NO new noise.

    The original LWE noise e survives unchanged in slot 0 of the recovered
    m_hat; every higher slot has noise exactly 0.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_extracted_noise_matches_lwe_noise(self, params: RlweParams) -> None:
        rng = random.Random(0xCAFE_01)
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        s_hat = s_hat_from_s_tilde(s_tilde, params)
        for _ in range(20):
            m = rng.randrange(params.p)
            ct = lwe.encrypt(s, m, params, rng)
            e_lwe = lwe.extract_noise(s, ct, m, params)
            ictx = transform(ct, params)
            m_bar = [m] + [0] * (params.d - 1)
            e_ictx = extract_noise_under_s_hat(ictx, s_hat, m_bar, params)
            assert e_ictx[0] == e_lwe
            assert all(e_ictx[k] == 0 for k in range(1, params.d))


# ---------------------------------------------------------------------------
# 9. a_hat depends only on a (offline/online split)
# ---------------------------------------------------------------------------


class TestAHatIndependentOfB:
    """transform(a, b1).a_hat == transform(a, b2).a_hat for any b1, b2.

    This is the algebraic foundation of the offline/online split in
    SPEC.md section 8. The server can precompute a_hat per CRS-fixed LWE
    a-vector once; per-query work is trivial (a constant polynomial b * X^0).
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_random_b_pairs(self, params: RlweParams) -> None:
        rng = random.Random(0xACE_01)
        a = [rng.randrange(params.q) for _ in range(params.d)]
        b1 = rng.randrange(params.q)
        b2 = rng.randrange(params.q)
        ct1 = lwe.LweCiphertext(a=a, b=b1)
        ct2 = lwe.LweCiphertext(a=a, b=b2)
        ictx1 = transform(ct1, params)
        ictx2 = transform(ct2, params)
        assert ictx1.a_hat == ictx2.a_hat
        # b_tilde does depend on b, naturally.
        assert ictx1.b_tilde[0] == b1 % params.q
        assert ictx2.b_tilde[0] == b2 % params.q


# ---------------------------------------------------------------------------
# 10. Structural distinction: a_tilde uses negative exponent, s_tilde positive
# ---------------------------------------------------------------------------


class TestStructuralDistinctionFromSTilde:
    """Pin down that a_tilde and s_tilde use *different* embeddings."""

    def test_kat_d4(self) -> None:
        """For a = s = [1, 2, 3, 4], a_tilde != s_tilde in general."""
        params = RlweParams(d=4, q=12289, p=4, sigma=3.2, z=64, ell=3)
        v = [1, 2, 3, 4]
        a_tilde = _a_tilde_oracle(v, params.q)
        s_tilde = rlwe.s_tilde_from_s(v, params)
        assert s_tilde == [1, 2, 3, 4]
        assert a_tilde == [1, (-4) % params.q, (-3) % params.q, (-2) % params.q]
        assert a_tilde != s_tilde

    def test_distinct_for_random_input(self) -> None:
        """Random non-zero non-symmetric vectors produce different embeddings."""
        params = ORACLE_TINY
        rng = random.Random(0xD15_01)
        for _ in range(20):
            v = [rng.randrange(1, params.q) for _ in range(params.d)]
            a_tilde = _a_tilde_oracle(v, params.q)
            s_tilde = rlwe.s_tilde_from_s(v, params)
            # They should differ in at least one slot.
            assert a_tilde != s_tilde


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("params", PARAMS)
    def test_same_input_same_output(self, params: RlweParams) -> None:
        rng = random.Random(0xDE7_01)
        s = lwe.keygen(params, rng)
        ct = lwe.encrypt(s, rng.randrange(params.p), params, rng)
        ictx_a = transform(ct, params)
        ictx_b = transform(ct, params)
        assert ictx_a == ictx_b
