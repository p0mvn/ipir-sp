"""Stage 7: tests for ``KS.Setup`` and ``KS.Switch`` (SPEC.md sections 6, 7).

Nine test groups, layered from primitive to InspiRING-specific:

1. ``TestSetupShape`` -- ``K = (w, y, noise)`` has the right shape and
   coefficient ranges.

2. ``TestSetupFormula`` -- the **defining equation**::

       y[i] == -s_out * w[i] + s_in * z^i + e[i]   (mod q)

   verified by re-running the formula against ``K.noise`` and asserting
   byte-equality with ``K.y``.

3. ``TestSwitchShape`` -- output is two length-``d`` polynomials in ``[0, q)``.

4. ``TestSwitchCorrectness`` -- the **headline contract**: encrypt under
   ``s_in``, switch with ``K = setup(s_in, s_out, ...)``, decrypt under
   ``s_out``, recover the same plaintext.

5. ``TestPerStepNoiseBound`` -- empirical: the noise added by one
   ``KS.Switch`` call has variance below Theorem 2's per-step bound
   ``ell * d * z^2 * sigma^2 / 4``.

6. ``TestSwitchCallCounter`` -- the call counter increments correctly
   and resets cleanly. **Required** for Stages 11/12's ``d - 1`` invariant.

7. ``TestKgKhUseCase`` -- the actual InspiRING idiom: build
   ``K_g = setup(tau_g(s_tilde), s_tilde)`` and ``K_h = setup(tau_h(s_tilde),
   s_tilde)``, then verify they switch correctly. Pins down the secret-pair
   convention used by Stages 10-12.

8. ``TestApplyAutomorph`` -- the **automorphism trick**:
   ``apply_automorph(K, g)`` is a valid KS matrix from ``tau_g(s_in)``
   to ``tau_g(s_out)``. Verified both by formula check and by full
   encrypt-switch-decrypt roundtrip. **This is the test that pins down
   InspiRING's "2 KS matrices instead of log d" advantage.**

9. ``TestSwitchChain`` -- two switches in sequence (``s_a -> s_b -> s_c``)
   work correctly. Defensive but cheap; rules out any state leakage in
   ``KS.Switch``.
"""

from __future__ import annotations

import random
import statistics

import pytest

from inspiring_oracle import lwe, rlwe
from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.key_switching import (
    KeySwitchingMatrix,
    apply_automorph,
    reset_switch_counter,
    setup,
    switch,
    switch_call_count,
)
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY
from inspiring_oracle.ring import add, mul, neg, scalar_mul


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0xCABBA6E)


@pytest.fixture(autouse=True)
def _reset_counter_per_test():
    """Every test in this file starts with the call counter at 0."""
    reset_switch_counter()
    yield


def _two_secrets(params, rng):
    """Generate two paired (s, s_tilde) pairs and return their RLWE forms."""
    s_in = lwe.keygen(params, rng)
    s_out = lwe.keygen(params, rng)
    return (
        rlwe.s_tilde_from_s(s_in, params),
        rlwe.s_tilde_from_s(s_out, params),
    )


# --------------------------------------------------------------------------
# 1. Setup shape
# --------------------------------------------------------------------------


class TestSetupShape:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_w_has_ell_rows_each_length_d(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        assert len(K.w) == params.ell
        for row in K.w:
            assert len(row) == params.d
            assert all(0 <= c < params.q for c in row)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_y_has_ell_rows_each_length_d(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        assert len(K.y) == params.ell
        for row in K.y:
            assert len(row) == params.d
            assert all(0 <= c < params.q for c in row)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_noise_has_ell_rows_each_length_d_signed(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        assert len(K.noise) == params.ell
        for row in K.noise:
            assert len(row) == params.d
            assert all(abs(c) < 6 * params.sigma + 5 for c in row)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_setup_rejects_wrong_secret_length(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        with pytest.raises(ValueError, match=r"length d ="):
            setup(s_in[:-1], s_out, params, rng)


# --------------------------------------------------------------------------
# 2. Setup formula (the defining equation)
# --------------------------------------------------------------------------


class TestSetupFormula:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_y_matches_minus_s_out_w_plus_s_in_zi_plus_e(
        self, params, rng
    ) -> None:
        """Re-derive ``y[i]`` from ``(w[i], noise[i], s_in, s_out)`` and
        compare byte-for-byte to ``K.y[i]``.
        """
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        for i in range(params.ell):
            e_i_modq = [v % params.q for v in K.noise[i]]
            z_pow_i = pow(params.z, i, params.q)
            s_in_zi = scalar_mul(s_in, z_pow_i, params.q)
            expected_y_i = add(
                add(
                    neg(mul(s_out, K.w[i], params.q), params.q),
                    s_in_zi,
                    params.q,
                ),
                e_i_modq,
                params.q,
            )
            assert K.y[i] == expected_y_i, f"mismatch at gadget level {i}"


# --------------------------------------------------------------------------
# 3. Switch shape
# --------------------------------------------------------------------------


class TestSwitchShape:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_outputs_are_length_d_polynomials_in_z_q(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        m_bar = [rng.randrange(params.p) for _ in range(params.d)]
        ct = rlwe.encrypt(s_in, m_bar, params, rng)
        a_new, b_new = switch(ct.c1, ct.c2, K, params)
        assert len(a_new) == params.d
        assert len(b_new) == params.d
        assert all(0 <= c < params.q for c in a_new)
        assert all(0 <= c < params.q for c in b_new)


# --------------------------------------------------------------------------
# 4. Switch correctness (the headline contract)
# --------------------------------------------------------------------------


class TestSwitchCorrectness:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_random_messages_survive_switching(self, params, rng) -> None:
        """50 random ``(s_in, s_out, m)``: encrypt under ``s_in``, switch,
        decrypt under ``s_out``, recover ``m``.
        """
        for _ in range(50):
            s_in, s_out = _two_secrets(params, rng)
            K = setup(s_in, s_out, params, rng)
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            ct = rlwe.encrypt(s_in, m_bar, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(s_out, new_ct, params) == m_bar

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_zero_message_after_switch(self, params, rng) -> None:
        """All-zero plaintext survives the switch."""
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        zero = [0] * params.d
        for _ in range(20):
            ct = rlwe.encrypt(s_in, zero, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(s_out, new_ct, params) == zero

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_max_message_after_switch(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        max_m = [params.p - 1] * params.d
        for _ in range(20):
            ct = rlwe.encrypt(s_in, max_m, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(s_out, new_ct, params) == max_m


# --------------------------------------------------------------------------
# 5. Per-step noise bound (Theorem 2 single-switch)
# --------------------------------------------------------------------------


class TestPerStepNoiseBound:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_empirical_variance_below_theorem2_per_step(
        self, params, rng
    ) -> None:
        """Run 200 fresh ``(K, ct)`` pairs encrypting all-zero, switch each
        once, extract the resulting noise. Empirical variance must be
        below ``ell * d * z^2 * sigma^2 / 4`` plus 5% slack.

        We use ``m_bar = 0`` so the residual ``raw mod q == e_old + e_ks``
        directly; ``rlwe.extract_noise(s_out, ct_new, 0, params)`` returns
        ``e_old + e_ks`` per coefficient.
        """
        per_step_var_bound = (
            params.ell * params.d * params.z ** 2 * params.sigma ** 2 / 4.0
        )
        zero = [0] * params.d
        coefs: list[int] = []
        for _ in range(200):
            s_in, s_out = _two_secrets(params, rng)
            K = setup(s_in, s_out, params, rng)
            ct = rlwe.encrypt(s_in, zero, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            coefs.extend(rlwe.extract_noise(s_out, new_ct, zero, params))
        empirical_var = statistics.fmean(c * c for c in coefs)
        assert empirical_var < per_step_var_bound * 1.05, (
            f"empirical variance {empirical_var:.1f} exceeds bound "
            f"{per_step_var_bound:.1f} (+5% slack)"
        )

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_max_abs_noise_within_decryption_budget(
        self, params, rng
    ) -> None:
        """After a single switch, ``|noise|_inf < Delta/2`` so decryption
        still works. This is the per-step correctness budget.
        """
        zero = [0] * params.d
        max_abs = 0
        for _ in range(100):
            s_in, s_out = _two_secrets(params, rng)
            K = setup(s_in, s_out, params, rng)
            ct = rlwe.encrypt(s_in, zero, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            coefs = rlwe.extract_noise(s_out, new_ct, zero, params)
            max_abs = max(max_abs, max(abs(c) for c in coefs))
        assert max_abs < params.delta // 2, (
            f"max |noise| {max_abs} exceeds budget {params.delta // 2}"
        )


# --------------------------------------------------------------------------
# 6. Switch call counter
# --------------------------------------------------------------------------


class TestSwitchCallCounter:
    def test_counter_starts_at_zero(self) -> None:
        assert switch_call_count() == 0

    def test_one_switch_increments_to_one(self, rng) -> None:
        params = ORACLE_TINY
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        ct = rlwe.encrypt(s_in, [0] * params.d, params, rng)
        switch(ct.c1, ct.c2, K, params)
        assert switch_call_count() == 1

    def test_multiple_switches_accumulate(self, rng) -> None:
        params = ORACLE_TINY
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        ct = rlwe.encrypt(s_in, [0] * params.d, params, rng)
        for _ in range(7):
            switch(ct.c1, ct.c2, K, params)
        assert switch_call_count() == 7

    def test_reset_clears_counter(self, rng) -> None:
        params = ORACLE_TINY
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        ct = rlwe.encrypt(s_in, [0] * params.d, params, rng)
        switch(ct.c1, ct.c2, K, params)
        switch(ct.c1, ct.c2, K, params)
        assert switch_call_count() == 2
        reset_switch_counter()
        assert switch_call_count() == 0


# --------------------------------------------------------------------------
# 7. K_g and K_h use cases (the InspiRING idiom)
# --------------------------------------------------------------------------


class TestKgKhUseCase:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_K_g_switches_tau_g_secret_back_to_base(self, params, rng) -> None:
        """``K_g = setup(tau_g(s_tilde), s_tilde)`` switches ciphertexts
        masked by ``tau_g(s_tilde)`` to ciphertexts under ``s_tilde``.
        """
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        tau_g_s_tilde = tau(s_tilde, G, params.q)
        K_g = setup(tau_g_s_tilde, s_tilde, params, rng)

        for _ in range(20):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            ct = rlwe.encrypt(tau_g_s_tilde, m_bar, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K_g, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(s_tilde, new_ct, params) == m_bar

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_K_h_switches_tau_h_secret_back_to_base(self, params, rng) -> None:
        """``K_h = setup(tau_h(s_tilde), s_tilde)`` switches ciphertexts
        masked by ``tau_h(s_tilde)`` to ciphertexts under ``s_tilde``.
        Used in the final step of Stage 12's ``Collapse``.
        """
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        tau_h_s_tilde = tau(s_tilde, h(params.d), params.q)
        K_h = setup(tau_h_s_tilde, s_tilde, params, rng)

        for _ in range(20):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            ct = rlwe.encrypt(tau_h_s_tilde, m_bar, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K_h, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(s_tilde, new_ct, params) == m_bar


# --------------------------------------------------------------------------
# 8. Apply automorphism (the "2 KS matrices instead of log d" trick)
# --------------------------------------------------------------------------


class TestApplyAutomorph:
    """The InspiRING-specific test group: ``apply_automorph(K, g, ...)``
    produces a valid KS matrix from ``tau_g(s_in)`` to ``tau_g(s_out)``.

    This is the property that makes ``CollapseHalf`` (Stage 11) work with
    just **one** base ``K_g`` matrix instead of ``log d`` distinct ones.
    If any test in this group fails, the entire ``Collapse`` cascade
    breaks.
    """

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_apply_automorph_preserves_shape(self, params, rng) -> None:
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        K_rho = apply_automorph(K, G, params)
        assert len(K_rho.w) == params.ell
        assert len(K_rho.y) == params.ell
        assert len(K_rho.noise) == params.ell
        for row in K_rho.w:
            assert len(row) == params.d
            assert all(0 <= c < params.q for c in row)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_apply_automorph_y_matches_formula(self, params, rng) -> None:
        """``rho(K).y[i] == -tau_g(s_out) * rho(K).w[i] + tau_g(s_in) * z^i + tau_g(e[i])``.

        Verifies the algebraic claim about ``rho(K)`` directly, separately
        from any encryption/decryption.
        """
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        K_rho = apply_automorph(K, G, params)

        tau_s_out = tau(s_out, G, params.q)
        tau_s_in = tau(s_in, G, params.q)
        for i in range(params.ell):
            tau_e_i_modq = [v % params.q for v in K_rho.noise[i]]
            z_pow_i = pow(params.z, i, params.q)
            expected = add(
                add(
                    neg(mul(tau_s_out, K_rho.w[i], params.q), params.q),
                    scalar_mul(tau_s_in, z_pow_i, params.q),
                    params.q,
                ),
                tau_e_i_modq,
                params.q,
            )
            assert K_rho.y[i] == expected, f"mismatch at gadget level {i}"

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_apply_automorph_switches_automorphic_secrets(
        self, params, rng
    ) -> None:
        """End-to-end: encrypt under ``tau_g(s_in)``, switch with
        ``apply_automorph(K, g)``, decrypt under ``tau_g(s_out)``,
        recover ``m``. The "headline test" for the automorphism trick.
        """
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        K_rho = apply_automorph(K, G, params)

        tau_s_in = tau(s_in, G, params.q)
        tau_s_out = tau(s_out, G, params.q)
        for _ in range(20):
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            ct = rlwe.encrypt(tau_s_in, m_bar, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K_rho, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(tau_s_out, new_ct, params) == m_bar

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_apply_automorph_powers_of_g_all_work(self, params, rng) -> None:
        """Iterate over a few powers ``g^k`` to confirm the trick stacks.

        ``Stage 11 needs apply_automorph(K_g, G^{k-1}, params)`` for
        ``k = 1, ..., d/2 - 1``. Spot-checking ``k = 1, 2, 3`` here gives
        that downstream test a stable foundation.
        """
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        for k in range(1, 4):
            g_k = pow(G, k, 2 * params.d)
            K_rho = apply_automorph(K, g_k, params)
            tau_s_in = tau(s_in, g_k, params.q)
            tau_s_out = tau(s_out, g_k, params.q)
            m_bar = [rng.randrange(params.p) for _ in range(params.d)]
            ct = rlwe.encrypt(tau_s_in, m_bar, params, rng)
            a_new, b_new = switch(ct.c1, ct.c2, K_rho, params)
            new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
            assert rlwe.decrypt(tau_s_out, new_ct, params) == m_bar, (
                f"apply_automorph failed at G^{k}"
            )

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_apply_automorph_with_h_works(self, params, rng) -> None:
        """``rho = tau_h`` is the other case used by Stage 11 (right half)."""
        s_in, s_out = _two_secrets(params, rng)
        K = setup(s_in, s_out, params, rng)
        K_rho = apply_automorph(K, h(params.d), params)
        tau_s_in = tau(s_in, h(params.d), params.q)
        tau_s_out = tau(s_out, h(params.d), params.q)
        m_bar = [rng.randrange(params.p) for _ in range(params.d)]
        ct = rlwe.encrypt(tau_s_in, m_bar, params, rng)
        a_new, b_new = switch(ct.c1, ct.c2, K_rho, params)
        new_ct = rlwe.RlweCiphertext(c1=a_new, c2=b_new)
        assert rlwe.decrypt(tau_s_out, new_ct, params) == m_bar


# --------------------------------------------------------------------------
# 9. Switch chain (sanity: state doesn't leak between switches)
# --------------------------------------------------------------------------


class TestSwitchChain:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_two_switches_in_sequence(self, params, rng) -> None:
        """``s_a -> s_b -> s_c`` round-trip recovers ``m``. Defensive.

        We use a low-noise message and verify the noise budget allows
        two switches; ``correctness_ok()`` checks the bound for d-1.
        """
        s_a, s_b = _two_secrets(params, rng)
        _, s_c = _two_secrets(params, rng)
        K_ab = setup(s_a, s_b, params, rng)
        K_bc = setup(s_b, s_c, params, rng)
        m_bar = [rng.randrange(params.p) for _ in range(params.d)]
        ct = rlwe.encrypt(s_a, m_bar, params, rng)
        a1, b1 = switch(ct.c1, ct.c2, K_ab, params)
        a2, b2 = switch(a1, b1, K_bc, params)
        new_ct = rlwe.RlweCiphertext(c1=a2, c2=b2)
        assert rlwe.decrypt(s_c, new_ct, params) == m_bar
