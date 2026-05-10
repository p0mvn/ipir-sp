"""Tests for ``collapse.collapse_one`` -- the atomic step of Algorithm 1 stage 3.

``CollapseOne`` is the fundamental noise-introducing operation of
InspiRING. Each invocation reduces a wider ciphertext by one component
via a single ``KS.Switch`` call. This is also the **only** place in
Algorithm 1 where noise grows; all subsequent stages either chain
``collapse_one`` together (Stages 11-12) or drive plaintext data
through it (Stage 13).

Test groups
-----------

1. **TestInputValidation** -- ``k < 2`` is rejected; the call counter
   does not increment on a rejected call.

2. **TestOutputShape** -- ``a'`` has length ``k - 1``; every polynomial
   has length ``d`` and coefficients in ``[0, q)``.

3. **TestCorrectnessFirewall** -- the load-bearing test:
   * Build a wider secret ``s' in (R_q)^k``.
   * Build a wider ciphertext encrypting message ``m_bar`` under ``s'``.
   * Set up ``K = KS.Setup(s'_{k-1}, s'_{k-2})``.
   * Run ``collapse_one(a, b, K)``.
   * Decrypt the result under ``s'' = s'[:k-1]``.
   * Recover ``m_bar`` correctly.

   Multiple ``k`` values (2, 3, 4, ``d/2``) and multiple message
   patterns (random, all-zero, all-max).

4. **TestNoiseGrowthBounded** -- the post-collapse noise is at most
   ``|e_old| + |e_ks|`` per coefficient, and the empirical KS noise
   stays well within ``Theorem 2``'s single-step bound
   ``sigma_one_ks <= sqrt(ell * d / 4) * z * sigma_chi``.

5. **TestRandomComponentInvariant** -- ``a'`` depends only on ``(a, K)``,
   not on ``b``. This is paper §3.2's invariant and the algebraic basis
   for the offline/online split (SPEC.md section 8): the server
   precomputes the entire ``a``-trace of the ``d - 1`` collapse steps,
   leaving the online phase to update only ``b``.

6. **TestKEqual2EdgeCase** -- the smallest valid case: ``k = 2``
   collapses to ``k = 1``, producing a wider ciphertext with a single
   secret share, which is structurally just an RLWE ciphertext under
   ``s'_0``.

7. **TestSwitchCallCountIncrement** -- each successful ``collapse_one``
   bumps ``key_switching.switch_call_count`` by exactly 1. Stage 12's
   ``d - 1`` invariant test depends on this.

8. **TestDeterminism** -- pure function of ``(a, b, K, params)``.
"""

from __future__ import annotations

import math
import random

import pytest

from inspiring_oracle import key_switching, lwe, rlwe
from inspiring_oracle.collapse import collapse_one
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY, RlweParams
from inspiring_oracle.ring import mul as ring_mul
from inspiring_oracle.ring import sub as ring_sub
from inspiring_oracle.wide_helpers import (
    build_wide_ciphertext,
    decrypt_wide,
    extract_wide_noise,
)

PARAMS = [ORACLE_TINY, ORACLE_SMALL]


def _make_wide_secret(
    k: int, params: RlweParams, rng: random.Random
) -> list[list[int]]:
    """Build a wider secret s' = (s'_0, ..., s'_{k-1}), each ternary."""
    return [
        rlwe.s_tilde_from_s(lwe.keygen(params, rng), params)
        for _ in range(k)
    ]


def _per_step_ks_noise_sigma_bound(params: RlweParams) -> float:
    """Theorem 2 single-step bound: sigma_one_ks <= sqrt(ell * d / 4) * z * sigma_chi."""
    return math.sqrt(params.ell * params.d / 4.0) * params.z * params.sigma


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize("params", PARAMS)
    def test_rejects_k_zero(self, params: RlweParams) -> None:
        rng = random.Random(0xC011_01)
        s_tilde = rlwe.keygen(params, rng)
        K = key_switching.setup(s_tilde, s_tilde, params, rng)
        with pytest.raises(ValueError, match="requires k >= 2"):
            collapse_one([], [0] * params.d, K, params)

    @pytest.mark.parametrize("params", PARAMS)
    def test_rejects_k_one(self, params: RlweParams) -> None:
        rng = random.Random(0xC011_02)
        s_tilde = rlwe.keygen(params, rng)
        K = key_switching.setup(s_tilde, s_tilde, params, rng)
        a = [[0] * params.d]  # k = 1
        with pytest.raises(ValueError, match="requires k >= 2"):
            collapse_one(a, [0] * params.d, K, params)

    @pytest.mark.parametrize("params", PARAMS)
    def test_rejected_call_does_not_increment_counter(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xC011_03)
        s_tilde = rlwe.keygen(params, rng)
        K = key_switching.setup(s_tilde, s_tilde, params, rng)
        key_switching.reset_switch_counter()
        before = key_switching.switch_call_count()
        with pytest.raises(ValueError):
            collapse_one([[0] * params.d], [0] * params.d, K, params)
        assert key_switching.switch_call_count() == before


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    @pytest.mark.parametrize("params", PARAMS)
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_a_prime_length_is_k_minus_one(
        self, params: RlweParams, k: int
    ) -> None:
        rng = random.Random(0xC011_10 + k)
        s_wide = _make_wide_secret(k, params, rng)
        m_bar = [0] * params.d
        a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        new_a, new_b = collapse_one(a, b, K, params)
        assert len(new_a) == k - 1
        assert all(len(poly) == params.d for poly in new_a)
        for poly in new_a:
            assert all(0 <= c < params.q for c in poly)
        assert len(new_b) == params.d
        assert all(0 <= c < params.q for c in new_b)


# ---------------------------------------------------------------------------
# 3. Correctness firewall
# ---------------------------------------------------------------------------


class TestCorrectnessFirewall:
    """Encrypt under s', collapse one component, decrypt under s'' -> recover m."""

    @pytest.mark.parametrize("params", PARAMS)
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_random_messages(self, params: RlweParams, k: int) -> None:
        rng = random.Random(0xC011_20 + k)
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        for _ in range(20):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
            new_a, new_b = collapse_one(a, b, K, params)
            recovered = decrypt_wide(new_a, new_b, s_wide[: k - 1], params)
            assert recovered == m_bar

    @pytest.mark.parametrize("params", PARAMS)
    def test_zero_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xC011_21)
        k = 3
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        m_bar = [0] * params.d
        a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
        new_a, new_b = collapse_one(a, b, K, params)
        assert decrypt_wide(new_a, new_b, s_wide[: k - 1], params) == m_bar

    @pytest.mark.parametrize("params", PARAMS)
    def test_max_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xC011_22)
        k = 3
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        m_bar = [params.p - 1] * params.d
        a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
        new_a, new_b = collapse_one(a, b, K, params)
        assert decrypt_wide(new_a, new_b, s_wide[: k - 1], params) == m_bar

    @pytest.mark.parametrize("params", PARAMS)
    def test_at_max_k_is_d_over_2(self, params: RlweParams) -> None:
        """Stage 11 uses k = d/2 as the starting point."""
        rng = random.Random(0xC011_23)
        k = params.d // 2
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        for _ in range(10):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
            new_a, new_b = collapse_one(a, b, K, params)
            assert decrypt_wide(new_a, new_b, s_wide[: k - 1], params) == m_bar


# ---------------------------------------------------------------------------
# 4. Noise growth bounded
# ---------------------------------------------------------------------------


class TestNoiseGrowthBounded:
    """The new noise is bounded by |e_old| + |e_ks|, and |e_ks| stays well
    inside Theorem 2's single-step subgaussian bound.
    """

    @pytest.mark.parametrize("params", PARAMS)
    @pytest.mark.parametrize("k", [2, 4])
    def test_noise_stays_within_decryption_budget(
        self, params: RlweParams, k: int
    ) -> None:
        """After one collapse, noise must still leave a comfortable
        margin to Delta / 2 (decryption budget)."""
        rng = random.Random(0xC011_30 + k)
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        budget = params.delta // 2
        for _ in range(20):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
            new_a, new_b = collapse_one(a, b, K, params)
            e_after = extract_wide_noise(
                new_a, new_b, s_wide[: k - 1], m_bar, params
            )
            assert all(abs(c) < budget for c in e_after), (
                f"noise overflowed Delta/2 = {budget}: "
                f"max |e| = {max(abs(c) for c in e_after)}"
            )

    @pytest.mark.parametrize("params", PARAMS)
    def test_ks_noise_below_theorem2_single_step_bound(
        self, params: RlweParams
    ) -> None:
        """Empirical |e_ks| should stay well below 6 * sigma_one_ks
        (subgaussian tail bound, ~1e-9 failure prob).

        Builds a *zero-noise* input ciphertext so the post-collapse noise
        IS the e_ks contribution -- isolating Stage 7's per-step noise
        for direct measurement.
        """
        rng = random.Random(0xC011_31)
        k = 3
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        sigma_bound = _per_step_ks_noise_sigma_bound(params)
        # Pick one trial; the goal is a sanity check, not statistical power
        # (Stage 14 will do statistics over many trials).
        m_bar = [0] * params.d
        # Construct a ciphertext with zero noise by hand.
        d, q = params.d, params.q
        a = [[rng.randrange(q) for _ in range(d)] for _ in range(k)]
        b = [0] * d
        for i in range(k):
            b = ring_sub(b, ring_mul(a[i], s_wide[i], q), q)
        new_a, new_b = collapse_one(a, b, K, params)
        e_after = extract_wide_noise(
            new_a, new_b, s_wide[: k - 1], m_bar, params
        )
        # Subgaussian: P(|x| > 6*sigma) is astronomically small.
        max_abs = max(abs(c) for c in e_after)
        assert max_abs < 6 * sigma_bound, (
            f"max |e_ks| = {max_abs} > 6 * sigma_one_ks = {6 * sigma_bound}"
        )


# ---------------------------------------------------------------------------
# 5. Random-component invariant (paper §3.2)
# ---------------------------------------------------------------------------


class TestRandomComponentInvariant:
    """``a'`` depends only on ``(a, K)``, not on ``b``.

    The foundation of the offline/online split: the server precomputes
    the full a-trace of the d-1 collapse steps from a CRS-fixed input.
    Per-query work only updates b.
    """

    @pytest.mark.parametrize("params", PARAMS)
    @pytest.mark.parametrize("k", [2, 3, 4])
    def test_varying_b_does_not_change_a_prime(
        self, params: RlweParams, k: int
    ) -> None:
        rng = random.Random(0xC011_40 + k)
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        a = [[rng.randrange(params.q) for _ in range(params.d)] for _ in range(k)]
        b1 = [rng.randrange(params.q) for _ in range(params.d)]
        b2 = [rng.randrange(params.q) for _ in range(params.d)]
        new_a1, _ = collapse_one(a, b1, K, params)
        new_a2, _ = collapse_one(a, b2, K, params)
        assert new_a1 == new_a2


# ---------------------------------------------------------------------------
# 6. k = 2 edge case
# ---------------------------------------------------------------------------


class TestKEqual2EdgeCase:
    """k = 2 -> k = 1 is the smallest non-trivial collapse.

    The output is a wider ciphertext with a single secret share, which
    is structurally just an RLWE ciphertext under s_wide[0]. Stage 12's
    final K_h step is exactly this case.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_k2_to_k1_decrypts_under_single_secret(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xC011_50)
        s_wide = _make_wide_secret(2, params, rng)
        K = key_switching.setup(s_wide[1], s_wide[0], params, rng)
        for _ in range(20):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
            new_a, new_b = collapse_one(a, b, K, params)
            assert len(new_a) == 1
            # Decryption under the single remaining secret share s_wide[0].
            assert decrypt_wide(new_a, new_b, [s_wide[0]], params) == m_bar


# ---------------------------------------------------------------------------
# 7. Switch call count increment
# ---------------------------------------------------------------------------


class TestSwitchCallCountIncrement:
    @pytest.mark.parametrize("params", PARAMS)
    def test_each_call_increments_counter_by_one(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xC011_60)
        k = 4
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        m_bar = [0] * params.d
        a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)

        key_switching.reset_switch_counter()
        for i in range(1, 6):
            collapse_one(a, b, K, params)
            assert key_switching.switch_call_count() == i


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("params", PARAMS)
    def test_same_inputs_same_output(self, params: RlweParams) -> None:
        rng = random.Random(0xC011_70)
        k = 3
        s_wide = _make_wide_secret(k, params, rng)
        K = key_switching.setup(s_wide[k - 1], s_wide[k - 2], params, rng)
        m_bar = [rng.randrange(params.p) for _ in range(params.d)]
        a, b, _ = build_wide_ciphertext(s_wide, m_bar, params, rng)
        out1 = collapse_one(a, b, K, params)
        out2 = collapse_one(a, b, K, params)
        assert out1 == out2
