"""Empirical validation of Theorem 2 -- end-to-end noise growth in ``pack``.

SPEC.md section 7 (paper Theorem 2):

    sigma_pack^2 <= ell * d^2 * z^2 * sigma_chi^2 / 4

Stages 1 and 2 of Algorithm 1 are noise-free; all noise enters in Stage
3's ``d - 1`` ``KS.Switch`` calls. Per-call noise has variance bounded
by ``ell * d * z^2 * sigma^2 / 4``, and under the independence heuristic
variances add over ``d - 1 <= d`` calls -- giving the bound above.

This file validates the bound **empirically**: run many ``pack``
operations on random inputs, extract the per-coefficient noise of each
output, and check that the sample variance lies inside the theoretical
ceiling.

The intent here is NOT to verify Theorem 2 from scratch (that's a paper
proof, not an oracle's job) -- it's to detect implementation drift:
* If we accidentally introduced extra noise (e.g. by reusing matrices,
  miscomputing automorphisms, or losing the d^{-1} cancellation), the
  empirical sigma would exceed the bound.
* If we accidentally LOST noise (e.g. by zeroing out a step), the
  empirical sigma would drop far below the bound and a
  ``test_noise_above_lower_floor`` check catches it (the noise should
  not be **vanishing**, since at least one ``KS.Switch`` runs).

Test groups
-----------

1. **TestEmpiricalSigmaBelowTheorem2** -- the headline check. Sample
   variance of pack noise stays under the theoretical ``sigma_pack``
   bound (with a small finite-sample slack).

2. **TestNoiseIsNotVanishing** -- regression guard against accidentally
   making pack noise-free (which would silently break security).

3. **TestNoiseIsRoughlyZeroMean** -- catches gross sign / centering
   bugs that would shift the noise distribution.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from inspiring_oracle import key_switching, lwe, rlwe
from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.pack import pack
from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY, RlweParams

PARAMS = [ORACLE_TINY, ORACLE_SMALL]

# Number of ``pack`` operations per parameter set. Each pack contributes
# ``d`` noise samples. With ``N_PACKS`` of order 50, we get hundreds of
# samples per param set -- enough for a stable sample sigma to ~5%.
N_PACKS = 60


def _theorem2_sigma_pack(params: RlweParams) -> float:
    """sigma_pack <= sqrt(ell * d^2 / 4) * z * sigma_chi  (SPEC.md section 7)."""
    return (
        math.sqrt(params.ell * params.d * params.d / 4.0)
        * params.z
        * params.sigma
    )


def _per_step_sigma_one_ks(params: RlweParams) -> float:
    """Single-step bound -- a useful "lower floor" reference."""
    return math.sqrt(params.ell * params.d / 4.0) * params.z * params.sigma


def _collect_pack_noise_samples(
    params: RlweParams, n_packs: int, base_seed: int
) -> list[int]:
    """Run ``n_packs`` independent packs and return all per-coefficient noises.

    Each pack uses fresh keys and fresh CRS material -- treating each
    coefficient sample as an independent draw from the noise distribution.
    """
    rng = random.Random(base_seed)
    samples: list[int] = []
    for _ in range(n_packs):
        s = lwe.keygen(params, rng)
        s_tilde = rlwe.s_tilde_from_s(s, params)
        K_g = key_switching.setup(
            tau(s_tilde, G, params.q), s_tilde, params, rng
        )
        K_h = key_switching.setup(
            tau(s_tilde, h(params.d), params.q), s_tilde, params, rng
        )
        messages = [rng.randrange(params.p) for _ in range(params.d)]
        cts = [lwe.encrypt(s, m, params, rng) for m in messages]
        packed = pack(cts, K_g, K_h, params)
        e = rlwe.extract_noise(s_tilde, packed, messages, params)
        samples.extend(e)
    return samples


# ---------------------------------------------------------------------------
# 1. Sample sigma below Theorem 2 ceiling
# ---------------------------------------------------------------------------


class TestEmpiricalSigmaBelowTheorem2:
    """Sample variance of pack noise must lie under the theoretical bound.

    Theorem 2 is an upper bound; the empirical sigma is typically a
    constant factor *below* it (the proof's bounds are loose). We allow
    a 20% slack on top of the theoretical ceiling to absorb finite-sample
    fluctuation -- a regression that introduced even one extra KS step
    would blow this slack out.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_sample_sigma_under_bound(self, params: RlweParams) -> None:
        seed = 0xCAFE_C0DE
        samples = _collect_pack_noise_samples(params, N_PACKS, seed)
        # Centered (signed) integers; mean should be ~0 (verified separately).
        # Variance of zero-mean iid is sum(x^2) / N.
        n = len(samples)
        empirical_var = sum(c * c for c in samples) / n
        empirical_sigma = math.sqrt(empirical_var)
        bound = _theorem2_sigma_pack(params)
        slack = 1.20  # 20% finite-sample slack
        assert empirical_sigma < slack * bound, (
            f"empirical sigma {empirical_sigma:.1f} exceeds Theorem 2 bound "
            f"{bound:.1f} (slack {slack}); n = {n} samples, params = {params}"
        )


# ---------------------------------------------------------------------------
# 2. Noise must not be (accidentally) zero
# ---------------------------------------------------------------------------


class TestNoiseIsNotVanishing:
    """Pack noise must be substantially non-zero.

    Lower floor: at least ``sigma_one_ks`` (one KS.Switch worth). If we
    accidentally lost the noise (e.g. by dropping the `e[i]` term inside
    ``KS.Setup``), this check fires immediately.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_sample_sigma_above_lower_floor(self, params: RlweParams) -> None:
        seed = 0xCAFE_FACE
        samples = _collect_pack_noise_samples(params, N_PACKS, seed)
        n = len(samples)
        empirical_var = sum(c * c for c in samples) / n
        empirical_sigma = math.sqrt(empirical_var)
        floor = _per_step_sigma_one_ks(params)
        # Pack noise should at least be 1x sigma_one_ks (one step's worth
        # of noise). In practice it's closer to sqrt(d-1) * sigma_one_ks
        # because variances add over d-1 independent steps.
        assert empirical_sigma > floor, (
            f"empirical sigma {empirical_sigma:.1f} is below the lower "
            f"floor {floor:.1f} (expected at least one KS step worth of "
            f"noise); n = {n} samples, params = {params}"
        )


# ---------------------------------------------------------------------------
# 3. Sample mean is approximately zero
# ---------------------------------------------------------------------------


class TestNoiseIsRoughlyZeroMean:
    """Mean of pack noise samples should be close to 0.

    For zero-mean noise with ``sigma`` variance and ``n`` independent
    samples, the sample mean has standard error ``sigma / sqrt(n)``. We
    allow a generous ``5 * sigma / sqrt(n)`` slack -- a true zero-mean
    distribution stays inside this with overwhelming probability, while
    a sign or centering bug typically shifts the mean by an entire
    ``sigma`` worth or more.
    """

    @pytest.mark.parametrize("params", PARAMS)
    def test_sample_mean_close_to_zero(self, params: RlweParams) -> None:
        seed = 0xCAFE_BABE
        samples = _collect_pack_noise_samples(params, N_PACKS, seed)
        n = len(samples)
        sample_mean = statistics.mean(samples)
        sigma = _theorem2_sigma_pack(params)
        slack = 5.0 * sigma / math.sqrt(n)
        assert abs(sample_mean) < slack, (
            f"sample mean {sample_mean:.2f} exceeds slack {slack:.2f} "
            f"(sigma_pack = {sigma:.1f}, n = {n}, params = {params})"
        )
