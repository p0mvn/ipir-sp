"""Stage 4: tests for symmetric-key LWE (SPEC.md section 1).

Six test groups, each pinning down one specific property:

1. ``TestKeygen`` -- secret structure (length, ternary support, uniformity).

2. ``TestSampleNoise`` -- the discrete-Gaussian noise sampler is centered
   at 0 with the requested standard deviation. This is the firewall for
   "I subtly broke the chi distribution" regressions.

3. ``TestEncryptDecryptRoundtrip`` -- the headline contract: encrypt then
   decrypt recovers ``m`` for every valid ``m``, over many random
   ``(s, RNG-state)`` choices. 1000+ samples per parameter set.

4. ``TestSignConventionKAT`` -- the **most important** Stage 4 test.
   Constructs a ciphertext entirely by hand (no RNG, no noise) per the
   SPEC.md formula ``b = -<a, s> + Delta * m``, then verifies decryption.
   If the encryption formula uses ``+<a, s>`` instead of ``-<a, s>``, this
   test catches it immediately.

5. ``TestNoiseBudget`` -- ``extract_noise`` returns values that:
   - Have mean ~ 0 and std ~ sigma (chi distribution check).
   - Stay within ``6 * sigma`` for >99% of samples (sub-Gaussian tail).
   - Always satisfy ``|e| < Delta / 2`` so single-LWE decryption is
     correct (the "noise budget for one encryption" check).

6. ``TestDeterminism`` -- given the same RNG seed and the same secret,
   ``encrypt`` returns the same ciphertext byte-for-byte. Required for
   the JSON fixtures in Stage 15 to be reproducible.

7. ``TestInputValidation`` -- ``encrypt`` rejects messages outside
   ``[0, p)``. Defensive but cheap.
"""

from __future__ import annotations

import random
import statistics

import pytest

from inspiring_oracle.lwe import (
    LweCiphertext,
    decrypt,
    encrypt,
    extract_noise,
    keygen,
    sample_noise,
)
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY, RlweParams


@pytest.fixture
def rng() -> random.Random:
    return random.Random(0xCABBA6E)


# --------------------------------------------------------------------------
# 1. Keygen
# --------------------------------------------------------------------------


class TestKeygen:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_secret_length_equals_d(self, params, rng) -> None:
        s = keygen(params, rng)
        assert len(s) == params.d

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_secret_values_are_ternary(self, params, rng) -> None:
        for _ in range(50):
            s = keygen(params, rng)
            assert all(v in (-1, 0, 1) for v in s), s

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_secret_distribution_is_approximately_uniform(
        self, params, rng
    ) -> None:
        """Over 3000 secret-coefficient samples, each of {-1, 0, 1} should
        appear roughly 1000 times. We allow a wide chi-squared-style
        tolerance (+- 20%) -- this catches "secret is all zeros" or
        "secret is uniformly in {0, 1}" regressions, not subtle bias.
        """
        n = 3000
        samples: list[int] = []
        for _ in range(n // params.d):
            samples.extend(keygen(params, rng))
        counts = {v: samples.count(v) for v in (-1, 0, 1)}
        expected = len(samples) / 3.0
        for v, c in counts.items():
            assert abs(c - expected) < 0.2 * expected, (
                f"value {v} count {c} far from expected {expected:.0f}"
            )


# --------------------------------------------------------------------------
# 2. Noise sampler (chi distribution)
# --------------------------------------------------------------------------


class TestSampleNoise:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_noise_is_int(self, params, rng) -> None:
        for _ in range(20):
            assert isinstance(sample_noise(params, rng), int)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_noise_statistics_match_chi(self, params, rng) -> None:
        """10000 samples: mean within +-0.1*sigma of 0, std within +-10% of sigma."""
        samples = [sample_noise(params, rng) for _ in range(10_000)]
        mean = statistics.fmean(samples)
        std = statistics.stdev(samples)
        assert abs(mean) < 0.1 * params.sigma, f"mean {mean} too far from 0"
        assert 0.9 * params.sigma < std < 1.1 * params.sigma, (
            f"std {std:.3f} not within +-10% of sigma={params.sigma}"
        )


# --------------------------------------------------------------------------
# 3. Encrypt / Decrypt roundtrip
# --------------------------------------------------------------------------


class TestEncryptDecryptRoundtrip:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_random_messages_roundtrip(self, params, rng) -> None:
        """1000 random ``(s, m)`` pairs -- decrypt(encrypt(m), s) == m."""
        for _ in range(1000):
            s = keygen(params, rng)
            m = rng.randrange(params.p)
            ct = encrypt(s, m, params, rng)
            assert decrypt(s, ct, params) == m, (
                f"roundtrip failed: m={m}, ct.b={ct.b}, p={params.p}"
            )

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_every_message_roundtrips(self, params, rng) -> None:
        """For each ``m in [0, p)``, 50 fresh-randomness encryptions all decrypt
        back to the same ``m``. Catches "decrypts to one specific message
        mod some bug" regressions.
        """
        s = keygen(params, rng)
        for m in range(params.p):
            for _ in range(50):
                ct = encrypt(s, m, params, rng)
                assert decrypt(s, ct, params) == m, f"failed for m={m}"

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_zero_message_roundtrips(self, params, rng) -> None:
        s = keygen(params, rng)
        for _ in range(100):
            ct = encrypt(s, 0, params, rng)
            assert decrypt(s, ct, params) == 0

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_max_message_roundtrips(self, params, rng) -> None:
        s = keygen(params, rng)
        m = params.p - 1
        for _ in range(100):
            ct = encrypt(s, m, params, rng)
            assert decrypt(s, ct, params) == m


# --------------------------------------------------------------------------
# 4. Sign convention KAT (the firewall test)
# --------------------------------------------------------------------------


class TestSignConventionKAT:
    """Construct ciphertexts manually (no RNG, no noise) to pin down the
    encryption formula's sign convention. SPEC.md section 1 specifies::

        b = -<a, s> + e + Delta * m  (mod q)

    If anyone "fixes" this to ``+<a, s>``, every test in this group fails.
    """

    def test_manual_zero_noise_zero_message_at_d8(self) -> None:
        """``e = 0, m = 0``: then ``b = -<a, s> mod q``, decrypts to 0."""
        params = ORACLE_TINY
        s = [1, -1, 0, 1, 0, -1, 1, 0]
        a = [100, 200, 300, 400, 500, 600, 700, 800]
        inner = sum(ai * si for ai, si in zip(a, s))
        b = (-inner) % params.q
        ct = LweCiphertext(a=a, b=b)
        assert decrypt(s, ct, params) == 0

    def test_manual_zero_noise_nonzero_message_at_d8(self) -> None:
        """``e = 0, m = 2``: then ``b = -<a, s> + Delta * 2 mod q``, decrypts to 2."""
        params = ORACLE_TINY
        s = [1, -1, 0, 1, 0, -1, 1, 0]
        a = [100, 200, 300, 400, 500, 600, 700, 800]
        m = 2
        inner = sum(ai * si for ai, si in zip(a, s))
        b = (-inner + params.delta * m) % params.q
        ct = LweCiphertext(a=a, b=b)
        assert decrypt(s, ct, params) == m

    def test_manual_with_small_noise_decrypts_correctly(self) -> None:
        """``|e| << Delta/2``: noise doesn't push the rounding off the right slot."""
        params = ORACLE_TINY
        s = [-1, 1, 1, 0, -1, 0, 1, -1]
        a = [1234, 5678, 9012, 3456, 7890, 1357, 2468, 9999]
        m = 3
        e = 7
        inner = sum(ai * si for ai, si in zip(a, s))
        b = (-inner + e + params.delta * m) % params.q
        ct = LweCiphertext(a=a, b=b)
        assert decrypt(s, ct, params) == m

    def test_manual_with_negative_small_noise_decrypts_correctly(self) -> None:
        """``e < 0``, including the wraparound case (raw + delta//2 crosses q)."""
        params = ORACLE_TINY
        s = [-1, 1, 1, 0, -1, 0, 1, -1]
        a = [1234, 5678, 9012, 3456, 7890, 1357, 2468, 9999]
        m = 0
        e = -11
        inner = sum(ai * si for ai, si in zip(a, s))
        b = (-inner + e + params.delta * m) % params.q
        ct = LweCiphertext(a=a, b=b)
        assert decrypt(s, ct, params) == m

    def test_extract_noise_recovers_known_noise_value(self) -> None:
        """Manually construct a ciphertext with a known ``e``; ``extract_noise``
        returns exactly that ``e``.
        """
        params = ORACLE_TINY
        s = [1, -1, 0, 1, 0, -1, 1, 0]
        a = [100, 200, 300, 400, 500, 600, 700, 800]
        m = 1
        for e in (-15, -3, 0, 5, 12):
            inner = sum(ai * si for ai, si in zip(a, s))
            b = (-inner + e + params.delta * m) % params.q
            ct = LweCiphertext(a=a, b=b)
            assert extract_noise(s, ct, m, params) == e


# --------------------------------------------------------------------------
# 5. Noise budget (extracted noise distribution)
# --------------------------------------------------------------------------


class TestNoiseBudget:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_extracted_noise_statistics(self, params, rng) -> None:
        """Over 2000 fresh encryptions, extracted noise has mean ~ 0 and
        std ~ sigma. Verifies that ``encrypt`` and ``extract_noise``
        round-trip the chi distribution faithfully.
        """
        s = keygen(params, rng)
        noises: list[int] = []
        for _ in range(2000):
            m = rng.randrange(params.p)
            ct = encrypt(s, m, params, rng)
            noises.append(extract_noise(s, ct, m, params))
        mean = statistics.fmean(noises)
        std = statistics.stdev(noises)
        assert abs(mean) < 0.15 * params.sigma
        assert 0.9 * params.sigma < std < 1.1 * params.sigma

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_extracted_noise_within_six_sigma(self, params, rng) -> None:
        """Sub-Gaussian tail check: at most a handful of 1000 samples should
        exceed 6*sigma. We allow up to 5 outliers (2*N(0,1) two-sided
        ``Pr[|x| > 6] ~ 2e-9``, but rng.gauss is float-based and small-sigma
        rounding can produce occasional 5-6 sigma values; 0.5% slack).
        """
        s = keygen(params, rng)
        outliers = 0
        for _ in range(1000):
            m = rng.randrange(params.p)
            ct = encrypt(s, m, params, rng)
            e = extract_noise(s, ct, m, params)
            if abs(e) > 6 * params.sigma:
                outliers += 1
        assert outliers <= 5, f"got {outliers} > 6-sigma outliers (expected <= 5)"

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_single_lwe_noise_within_decryption_budget(self, params, rng) -> None:
        """The whole point of the noise budget: ``|e| < Delta/2`` so a fresh
        LWE ciphertext always decrypts. We sample 500 noises and assert
        the maximum is comfortably below the budget.
        """
        s = keygen(params, rng)
        noises: list[int] = []
        for _ in range(500):
            m = rng.randrange(params.p)
            ct = encrypt(s, m, params, rng)
            noises.append(extract_noise(s, ct, m, params))
        max_abs_noise = max(abs(e) for e in noises)
        budget = params.delta // 2
        assert max_abs_noise < budget, (
            f"noise {max_abs_noise} exceeded budget {budget}"
        )


# --------------------------------------------------------------------------
# 6. Determinism (required for Stage 15 fixtures)
# --------------------------------------------------------------------------


class TestDeterminism:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_same_seed_same_ciphertext(self, params) -> None:
        """Two encryptions with the same seed produce byte-identical ciphertexts."""
        seed = 0xDEADBEEF
        s = keygen(params, random.Random(seed))

        rng1 = random.Random(seed + 1)
        rng2 = random.Random(seed + 1)
        ct1 = encrypt(s, 1, params, rng1)
        ct2 = encrypt(s, 1, params, rng2)
        assert ct1.a == ct2.a
        assert ct1.b == ct2.b


# --------------------------------------------------------------------------
# 7. Input validation
# --------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_negative_message_rejected(self, params, rng) -> None:
        s = keygen(params, rng)
        with pytest.raises(ValueError, match="m = -1 not in"):
            encrypt(s, -1, params, rng)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_message_equal_to_p_rejected(self, params, rng) -> None:
        s = keygen(params, rng)
        with pytest.raises(ValueError, match=r"not in plaintext range"):
            encrypt(s, params.p, params, rng)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_message_above_p_rejected(self, params, rng) -> None:
        s = keygen(params, rng)
        with pytest.raises(ValueError):
            encrypt(s, params.p + 100, params, rng)


# --------------------------------------------------------------------------
# 8. Ciphertext structural invariants
# --------------------------------------------------------------------------


class TestCiphertextStructure:
    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_a_coefficients_in_z_q(self, params, rng) -> None:
        s = keygen(params, rng)
        for _ in range(100):
            m = rng.randrange(params.p)
            ct = encrypt(s, m, params, rng)
            assert len(ct.a) == params.d
            assert all(0 <= ai < params.q for ai in ct.a)

    @pytest.mark.parametrize("params", [ORACLE_TINY, ORACLE_SMALL])
    def test_b_in_z_q(self, params, rng) -> None:
        s = keygen(params, rng)
        for _ in range(100):
            m = rng.randrange(params.p)
            ct = encrypt(s, m, params, rng)
            assert 0 <= ct.b < params.q
