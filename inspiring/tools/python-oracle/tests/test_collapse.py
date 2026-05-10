"""Tests for ``collapse.collapse`` -- the top-level Stage 3 of Algorithm 1.

``collapse`` orchestrates both halves and the final ``K_h`` fold to turn
an aggregated IRCtx into an honest two-element RLWE ciphertext under
the base secret ``s_tilde``.

The single most important property tested here is the
**``d - 1`` ``KS.Switch`` invariant**: every well-formed call to
``collapse`` runs *exactly* ``d - 1`` key-switch operations. This is
the central design distinction of InspiRING vs CDKS (which would run
``(d - 1) * log d`` for the same packing).

Test groups
-----------

1. **TestOutputType** -- returns a proper ``RlweCiphertext`` with
   ``c1, c2`` of length ``d``, coefficients in ``[0, q)``.

2. **TestSwitchCallCountInvariant** -- the load-bearing structural test:
   exactly ``d - 1`` ``KS.Switch`` calls. This invariant prevents the
   most common implementation drift (accidental CDKS-style recursion).

3. **TestCorrectnessRoundtrip** -- the firewall: build an aggregated
   IRCtx from real LWEs, ``collapse``, decrypt with ``rlwe.decrypt``,
   recover the per-slot messages.

4. **TestNoiseGrowthBounded** -- end-to-end noise stays under
   ``Delta / 2``; empirical noise stays under ``6 * sigma_pack``
   (Theorem 2 single-sample tail bound).

5. **TestRandomComponentInvariant** -- ``c1`` depends only on
   ``(a_hat_agg, K_g, K_h)``, never on ``b_tilde_agg``. The basis for
   pre-computing the entire ``a``-trace of all ``d - 1`` collapse steps.

6. **TestDeterminism** -- pure function of inputs.

7. **TestEdgeCases** -- all-zero messages; all-max messages.

Stage 13's pack-roundtrip test layers on top of this with the full
LWE -> RLWE pipeline.
"""

from __future__ import annotations

import math
import random

import pytest

from inspiring_oracle import key_switching, lwe, rlwe
from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.collapse import collapse
from inspiring_oracle.intermediate import IRCtx, aggregate, transform
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY, RlweParams

PARAMS = [ORACLE_TINY, ORACLE_SMALL]


def _build_pipeline(
    params: RlweParams, rng: random.Random
) -> tuple[
    list[int],         # s
    list[int],         # s_tilde
    key_switching.KeySwitchingMatrix,  # K_g
    key_switching.KeySwitchingMatrix,  # K_h
]:
    """Generate a fresh secret key and the two CRS key-switching matrices."""
    s = lwe.keygen(params, rng)
    s_tilde = rlwe.s_tilde_from_s(s, params)
    tau_g_s = tau(s_tilde, G, params.q)
    tau_h_s = tau(s_tilde, h(params.d), params.q)
    K_g = key_switching.setup(tau_g_s, s_tilde, params, rng)
    K_h = key_switching.setup(tau_h_s, s_tilde, params, rng)
    return s, s_tilde, K_g, K_h


def _aggregate_d_messages(
    s: list[int],
    messages: list[int],
    params: RlweParams,
    rng: random.Random,
) -> IRCtx:
    """Encrypt ``d`` messages under ``s`` and aggregate the resulting IRCtxs."""
    cts = [lwe.encrypt(s, m, params, rng) for m in messages]
    irctxs = [transform(ct, params) for ct in cts]
    return aggregate(irctxs, params)


def _theorem2_sigma_pack(params: RlweParams) -> float:
    """Theorem 2 bound: sigma_pack <= sqrt(ell * d^2 / 4) * z * sigma."""
    return (
        math.sqrt(params.ell * params.d * params.d / 4.0)
        * params.z
        * params.sigma
    )


# ---------------------------------------------------------------------------
# 1. Output type
# ---------------------------------------------------------------------------


class TestOutputType:
    @pytest.mark.parametrize("params", PARAMS)
    def test_returns_rlwe_ciphertext(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_01)
        s, _s_tilde, K_g, K_h = _build_pipeline(params, rng)
        messages = [0] * params.d
        ictx_agg = _aggregate_d_messages(s, messages, params, rng)
        ct = collapse(ictx_agg, K_g, K_h, params)
        assert isinstance(ct, rlwe.RlweCiphertext)
        assert len(ct.c1) == params.d
        assert len(ct.c2) == params.d
        assert all(0 <= c < params.q for c in ct.c1)
        assert all(0 <= c < params.q for c in ct.c2)


# ---------------------------------------------------------------------------
# 2. Switch call count -- the d - 1 invariant
# ---------------------------------------------------------------------------


class TestSwitchCallCountInvariant:
    """Each ``collapse`` does **exactly** ``d - 1`` ``KS.Switch`` calls.

    This is the central design distinction of InspiRING vs CDKS, asserted
    at runtime as a structural firewall against accidental drift to
    CDKS-style ``(d - 1) * log d`` recursion.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_exactly_d_minus_1_switch_calls(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_10)
        s, _s_tilde, K_g, K_h = _build_pipeline(params, rng)
        messages = [rng.randrange(params.p) for _ in range(params.d)]
        ictx_agg = _aggregate_d_messages(s, messages, params, rng)
        key_switching.reset_switch_counter()
        collapse(ictx_agg, K_g, K_h, params)
        assert key_switching.switch_call_count() == params.d - 1

    @pytest.mark.parametrize("params", PARAMS)
    def test_count_is_invariant_across_inputs(
        self, params: RlweParams
    ) -> None:
        """Different inputs all produce d - 1 calls -- structural, not data-dependent."""
        rng = random.Random(0xC12_11)
        s, _s_tilde, K_g, K_h = _build_pipeline(params, rng)
        for trial in range(5):
            messages = [rng.randrange(params.p) for _ in range(params.d)]
            ictx_agg = _aggregate_d_messages(s, messages, params, rng)
            key_switching.reset_switch_counter()
            collapse(ictx_agg, K_g, K_h, params)
            assert key_switching.switch_call_count() == params.d - 1, (
                f"trial {trial}: {key_switching.switch_call_count()} != {params.d - 1}"
            )


# ---------------------------------------------------------------------------
# 3. Correctness roundtrip
# ---------------------------------------------------------------------------


class TestCorrectnessRoundtrip:
    """Aggregate IRCtx -> collapse -> rlwe.decrypt -> recover all d messages."""

    @pytest.mark.parametrize("params", PARAMS)
    def test_random_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_20)
        s, s_tilde, K_g, K_h = _build_pipeline(params, rng)
        for _ in range(20):
            messages = [rng.randrange(params.p) for _ in range(params.d)]
            ictx_agg = _aggregate_d_messages(s, messages, params, rng)
            ct = collapse(ictx_agg, K_g, K_h, params)
            recovered = rlwe.decrypt(s_tilde, ct, params)
            assert recovered == messages


# ---------------------------------------------------------------------------
# 4. Noise growth bounded
# ---------------------------------------------------------------------------


class TestNoiseGrowthBounded:
    @pytest.mark.parametrize("params", PARAMS)
    def test_noise_under_decryption_budget(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_30)
        s, s_tilde, K_g, K_h = _build_pipeline(params, rng)
        budget = params.delta // 2
        for _ in range(20):
            messages = [rng.randrange(params.p) for _ in range(params.d)]
            ictx_agg = _aggregate_d_messages(s, messages, params, rng)
            ct = collapse(ictx_agg, K_g, K_h, params)
            e = rlwe.extract_noise(s_tilde, ct, messages, params)
            max_abs = max(abs(c) for c in e)
            assert max_abs < budget, (
                f"noise overflowed Delta/2 = {budget}: max |e| = {max_abs}"
            )

    @pytest.mark.parametrize("params", PARAMS)
    def test_noise_under_6_sigma_pack(self, params: RlweParams) -> None:
        """Subgaussian tail bound: P(|e| > 6 sigma_pack) ~ 1e-9."""
        rng = random.Random(0xC12_31)
        s, s_tilde, K_g, K_h = _build_pipeline(params, rng)
        sigma_pack = _theorem2_sigma_pack(params)
        bound = 6.0 * sigma_pack
        # 3 trials -- a sanity check, not statistical power. Stage 14 does stats.
        for _ in range(3):
            messages = [rng.randrange(params.p) for _ in range(params.d)]
            ictx_agg = _aggregate_d_messages(s, messages, params, rng)
            ct = collapse(ictx_agg, K_g, K_h, params)
            e = rlwe.extract_noise(s_tilde, ct, messages, params)
            max_abs = max(abs(c) for c in e)
            assert max_abs < bound, (
                f"max |e_pack| = {max_abs} > 6 * sigma_pack = {bound:.0f}"
            )


# ---------------------------------------------------------------------------
# 5. Random-component invariant (the offline-precomputable a-trace)
# ---------------------------------------------------------------------------


class TestRandomComponentInvariant:
    """``c1`` depends only on ``(a_hat_agg, K_g, K_h)``, never on ``b_tilde_agg``."""

    @pytest.mark.parametrize("params", PARAMS)
    def test_varying_b_tilde_does_not_change_c1(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xC12_40)
        _s, _s_tilde, K_g, K_h = _build_pipeline(params, rng)
        # Build a synthetic IRCtx with chosen a_hat and two different b_tildes.
        a_hat = [
            [rng.randrange(params.q) for _ in range(params.d)]
            for _ in range(params.d)
        ]
        b_tilde_1 = [rng.randrange(params.q) for _ in range(params.d)]
        b_tilde_2 = [rng.randrange(params.q) for _ in range(params.d)]
        ct_1 = collapse(IRCtx(a_hat=a_hat, b_tilde=b_tilde_1), K_g, K_h, params)
        ct_2 = collapse(IRCtx(a_hat=a_hat, b_tilde=b_tilde_2), K_g, K_h, params)
        assert ct_1.c1 == ct_2.c1
        # c2 SHOULD differ -- it carries the message+noise.
        assert ct_1.c2 != ct_2.c2


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("params", PARAMS)
    def test_same_inputs_same_output(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_50)
        s, _s_tilde, K_g, K_h = _build_pipeline(params, rng)
        messages = [rng.randrange(params.p) for _ in range(params.d)]
        ictx_agg = _aggregate_d_messages(s, messages, params, rng)
        ct1 = collapse(ictx_agg, K_g, K_h, params)
        ct2 = collapse(ictx_agg, K_g, K_h, params)
        assert ct1 == ct2


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("params", PARAMS)
    def test_all_zero_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_60)
        s, s_tilde, K_g, K_h = _build_pipeline(params, rng)
        messages = [0] * params.d
        ictx_agg = _aggregate_d_messages(s, messages, params, rng)
        ct = collapse(ictx_agg, K_g, K_h, params)
        assert rlwe.decrypt(s_tilde, ct, params) == messages

    @pytest.mark.parametrize("params", PARAMS)
    def test_all_max_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xC12_61)
        s, s_tilde, K_g, K_h = _build_pipeline(params, rng)
        messages = [params.p - 1] * params.d
        ictx_agg = _aggregate_d_messages(s, messages, params, rng)
        ct = collapse(ictx_agg, K_g, K_h, params)
        assert rlwe.decrypt(s_tilde, ct, params) == messages
