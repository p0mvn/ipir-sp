"""End-to-end roundtrip tests for ``pack`` -- Algorithm 1 of the InsPIRe paper.

These are the **headline correctness tests** for the entire oracle:
encrypt ``d`` plaintexts as LWE ciphertexts, ``pack`` them with the
two CRS-fixed key-switching matrices, decrypt the resulting RLWE
ciphertext, and recover the original plaintexts as the coefficients of
the message polynomial.

Test groups
-----------

1. **TestInputValidation** -- rejects input lists of the wrong length.

2. **TestRoundtripFirewall** -- the load-bearing test. ``d`` random
   messages survive the entire LWE -> IRCtx -> aggregate -> collapse
   -> RLWE pipeline.

3. **TestSwitchCallCount** -- pack invokes ``KS.Switch`` exactly
   ``d - 1`` times (transform / aggregate add zero, collapse adds
   ``d - 1``).

4. **TestRandomComponentInvariant** -- ``c1`` of the output depends
   only on the LWE ``a``-vectors and the two key-switching matrices,
   never on the LWE ``b``-values. The algebraic basis for the
   ``PackPreprocessed`` cache (SPEC.md section 8).

5. **TestNoiseBoundedByDeltaHalf** -- end-to-end correctness margin.

6. **TestDeterminism** -- pure function of inputs.

7. **TestEdgeCases** -- all-zero messages; all-max messages; only one
   non-zero slot.

8. **TestMultipleSecretsIndependent** -- packing the same messages
   under different secrets produces ciphertexts decryptable only with
   their respective secret keys. Sanity check that secrets actually
   matter (catches the "I forgot to use the key" class of bug).
"""

from __future__ import annotations

import random

import pytest

from inspiring_oracle import key_switching, lwe, rlwe
from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.pack import pack
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY, RlweParams

PARAMS = [ORACLE_TINY, ORACLE_SMALL]


def _setup(
    params: RlweParams, rng: random.Random
) -> tuple[
    list[int],         # s
    list[int],         # s_tilde
    key_switching.KeySwitchingMatrix,  # K_g
    key_switching.KeySwitchingMatrix,  # K_h
]:
    s = lwe.keygen(params, rng)
    s_tilde = rlwe.s_tilde_from_s(s, params)
    K_g = key_switching.setup(tau(s_tilde, G, params.q), s_tilde, params, rng)
    K_h = key_switching.setup(
        tau(s_tilde, h(params.d), params.q), s_tilde, params, rng
    )
    return s, s_tilde, K_g, K_h


def _encrypt_d(
    s: list[int],
    messages: list[int],
    params: RlweParams,
    rng: random.Random,
) -> list[lwe.LweCiphertext]:
    return [lwe.encrypt(s, m, params, rng) for m in messages]


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize("params", PARAMS)
    def test_rejects_too_few_lwes(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_01)
        s, _s_tilde, K_g, K_h = _setup(params, rng)
        cts = _encrypt_d(s, [0] * (params.d - 1), params, rng)
        with pytest.raises(ValueError, match="exactly d"):
            pack(cts, K_g, K_h, params)

    @pytest.mark.parametrize("params", PARAMS)
    def test_rejects_too_many_lwes(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_02)
        s, _s_tilde, K_g, K_h = _setup(params, rng)
        cts = _encrypt_d(s, [0] * (params.d + 1), params, rng)
        with pytest.raises(ValueError, match="exactly d"):
            pack(cts, K_g, K_h, params)

    @pytest.mark.parametrize("params", PARAMS)
    def test_rejects_empty(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_03)
        _s, _s_tilde, K_g, K_h = _setup(params, rng)
        with pytest.raises(ValueError, match="exactly d"):
            pack([], K_g, K_h, params)


# ---------------------------------------------------------------------------
# 2. Roundtrip firewall -- the headline test
# ---------------------------------------------------------------------------


class TestRoundtripFirewall:
    """Encrypt d -> pack -> decrypt -> recover all d. Multiple runs."""

    @pytest.mark.parametrize("params", PARAMS)
    def test_random_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_10)
        s, s_tilde, K_g, K_h = _setup(params, rng)
        for _ in range(20):
            messages = [rng.randrange(params.p) for _ in range(params.d)]
            cts = _encrypt_d(s, messages, params, rng)
            packed = pack(cts, K_g, K_h, params)
            recovered = rlwe.decrypt(s_tilde, packed, params)
            assert recovered == messages


# ---------------------------------------------------------------------------
# 3. Switch call count -- d - 1 (the structural InspiRING property)
# ---------------------------------------------------------------------------


class TestSwitchCallCount:
    """``pack`` runs exactly ``d - 1`` ``KS.Switch`` calls.

    ``transform`` and ``aggregate`` add zero; ``collapse`` adds
    ``(d/2 - 1) + (d/2 - 1) + 1 = d - 1``. A regression here is the
    canonical "we accidentally re-implemented CDKS" symptom.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_pack_uses_exactly_d_minus_1_switches(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xCA0_20)
        s, _s_tilde, K_g, K_h = _setup(params, rng)
        messages = [rng.randrange(params.p) for _ in range(params.d)]
        cts = _encrypt_d(s, messages, params, rng)
        key_switching.reset_switch_counter()
        pack(cts, K_g, K_h, params)
        assert key_switching.switch_call_count() == params.d - 1


# ---------------------------------------------------------------------------
# 4. Random-component invariant (PackPreprocessed basis)
# ---------------------------------------------------------------------------


class TestRandomComponentInvariant:
    """``c1`` depends only on (a-vectors, K_g, K_h), not on b-values."""

    @pytest.mark.parametrize("params", PARAMS)
    def test_varying_b_does_not_change_c1(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_30)
        _s, _s_tilde, K_g, K_h = _setup(params, rng)
        # Build d LWE ciphertexts with shared a-vectors, different b-values.
        a_vecs = [
            [rng.randrange(params.q) for _ in range(params.d)]
            for _ in range(params.d)
        ]
        b1 = [rng.randrange(params.q) for _ in range(params.d)]
        b2 = [rng.randrange(params.q) for _ in range(params.d)]
        cts1 = [lwe.LweCiphertext(a=a, b=b) for a, b in zip(a_vecs, b1, strict=True)]
        cts2 = [lwe.LweCiphertext(a=a, b=b) for a, b in zip(a_vecs, b2, strict=True)]
        ct_packed_1 = pack(cts1, K_g, K_h, params)
        ct_packed_2 = pack(cts2, K_g, K_h, params)
        assert ct_packed_1.c1 == ct_packed_2.c1
        # c2 SHOULD differ -- it carries the message+noise.
        assert ct_packed_1.c2 != ct_packed_2.c2


# ---------------------------------------------------------------------------
# 5. Noise bounded by Delta / 2
# ---------------------------------------------------------------------------


class TestNoiseBoundedByDeltaHalf:
    @pytest.mark.parametrize("params", PARAMS)
    def test_per_coefficient_noise_under_budget(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xCA0_40)
        s, s_tilde, K_g, K_h = _setup(params, rng)
        budget = params.delta // 2
        for _ in range(20):
            messages = [rng.randrange(params.p) for _ in range(params.d)]
            cts = _encrypt_d(s, messages, params, rng)
            packed = pack(cts, K_g, K_h, params)
            e = rlwe.extract_noise(s_tilde, packed, messages, params)
            max_abs = max(abs(c) for c in e)
            assert max_abs < budget, (
                f"noise overflowed Delta/2 = {budget}: max |e| = {max_abs}"
            )


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("params", PARAMS)
    def test_same_inputs_same_output(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_50)
        s, _s_tilde, K_g, K_h = _setup(params, rng)
        messages = [rng.randrange(params.p) for _ in range(params.d)]
        cts = _encrypt_d(s, messages, params, rng)
        out1 = pack(cts, K_g, K_h, params)
        out2 = pack(cts, K_g, K_h, params)
        assert out1 == out2


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("params", PARAMS)
    def test_all_zero_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_60)
        s, s_tilde, K_g, K_h = _setup(params, rng)
        messages = [0] * params.d
        cts = _encrypt_d(s, messages, params, rng)
        packed = pack(cts, K_g, K_h, params)
        assert rlwe.decrypt(s_tilde, packed, params) == messages

    @pytest.mark.parametrize("params", PARAMS)
    def test_all_max_messages(self, params: RlweParams) -> None:
        rng = random.Random(0xCA0_61)
        s, s_tilde, K_g, K_h = _setup(params, rng)
        messages = [params.p - 1] * params.d
        cts = _encrypt_d(s, messages, params, rng)
        packed = pack(cts, K_g, K_h, params)
        assert rlwe.decrypt(s_tilde, packed, params) == messages

    @pytest.mark.parametrize("params", PARAMS)
    @pytest.mark.parametrize("nonzero_slot", [0, 1, 3, -1])
    def test_only_one_nonzero_slot(
        self, params: RlweParams, nonzero_slot: int
    ) -> None:
        nonzero_slot = nonzero_slot % params.d
        rng = random.Random(0xCA0_62 + nonzero_slot)
        s, s_tilde, K_g, K_h = _setup(params, rng)
        messages = [0] * params.d
        messages[nonzero_slot] = params.p - 1
        cts = _encrypt_d(s, messages, params, rng)
        packed = pack(cts, K_g, K_h, params)
        assert rlwe.decrypt(s_tilde, packed, params) == messages


# ---------------------------------------------------------------------------
# 8. Different secrets isolate different ciphertexts
# ---------------------------------------------------------------------------


class TestMultipleSecretsIndependent:
    """Same messages, different secrets -> different packed ciphertexts.

    A sanity check that the secret actually participates in the output --
    catches the class of bug where an implementation accidentally hardcodes
    or shares secret material across calls.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_two_secrets_yield_two_different_packed_ciphertexts(
        self, params: RlweParams
    ) -> None:
        rng = random.Random(0xCA0_70)
        messages = [rng.randrange(params.p) for _ in range(params.d)]
        # Two independent secrets and CRS material.
        s1, s_tilde_1, K_g_1, K_h_1 = _setup(params, rng)
        s2, s_tilde_2, K_g_2, K_h_2 = _setup(params, rng)
        cts1 = _encrypt_d(s1, messages, params, rng)
        cts2 = _encrypt_d(s2, messages, params, rng)
        packed1 = pack(cts1, K_g_1, K_h_1, params)
        packed2 = pack(cts2, K_g_2, K_h_2, params)
        # Both must decrypt under their own secret.
        assert rlwe.decrypt(s_tilde_1, packed1, params) == messages
        assert rlwe.decrypt(s_tilde_2, packed2, params) == messages
        # And NOT under each other's secret (overwhelmingly likely).
        # (Decryption may give garbage; we just require it differs.)
        assert rlwe.decrypt(s_tilde_2, packed1, params) != messages
        assert rlwe.decrypt(s_tilde_1, packed2, params) != messages
