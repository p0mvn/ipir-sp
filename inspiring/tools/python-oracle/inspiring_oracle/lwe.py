"""Symmetric-key LWE encryption (SPEC.md section 1).

This module implements the **standard LWE** primitive that InspiRING.Pack
takes as input. It is intentionally minimal: the oracle never *sends* LWE
ciphertexts over a wire and never cares about ciphertext serialization or
public-key encryption -- the only operations needed are:

* Generate a secret key ``s`` (Stage 4 here).
* Encrypt a plaintext ``m`` to ``(a, b)`` (Stage 4 here, used as input to
  Stage 8's ``TRANSFORM`` later).
* Decrypt ``(a, b)`` back to ``m`` (Stage 4 here, used as the correctness
  oracle for every downstream stage).
* Extract the residual noise from ``(a, b, m, s)`` (used by Stages 7, 12,
  and 14 to instrument noise growth empirically).

Sign convention -- exactly per SPEC.md section 1 / paper Eq. (1):

    b = -<a, s> + e + Delta * m  (mod q),    Delta = floor(q / p).

The **negative** inner product is the paper's choice; ``decrypt`` therefore
adds the inner product back (positive sign) to recover ``Delta * m + e``.
This sign convention is the #2 from-scratch bug after the negacyclic-ring
reduction in Stage 1; ``test_lwe.py``'s manual KAT is the firewall for it.

Secret distribution -- ternary uniform on ``{-1, 0, 1}^d``:

    Production InspiRING uses a discrete-Gaussian secret matching the
    noise distribution; we use ternary instead because:
      (a) The Theorem 2 noise bound does **not** depend on the secret
          distribution -- only on chi (the noise distribution).
      (b) Ternary keeps ``s_tilde = sum s[i] * X^i`` (Stage 5) with small
          integer coefficients, which makes intermediate values inspectable
          by eye in test failures.
      (c) Security is irrelevant at d in {8, 16}: both parameter sets are
          completely broken regardless of secret distribution. The oracle's
          job is correctness, not security.

Noise distribution -- rounded continuous Gaussian:

    e = round(N(0, sigma)). For sigma ~ 3.2 (typical LWE setting), this
    is statistically indistinguishable from a true discrete Gaussian to
    several decimal places, with no relevance for our small parameter
    sets. spiral-rs uses the same approximation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from inspiring_oracle.params import RlweParams


@dataclass(frozen=True)
class LweCiphertext:
    """A symmetric-key LWE ciphertext ``(a, b) in Z_q^d x Z_q``.

    ``a`` is a length-``d`` list of integers in ``[0, q)``; ``b`` is a
    single integer in ``[0, q)``. The relation ``b = -<a, s> + e + Delta*m``
    holds modulo ``q``.
    """

    a: list[int]
    b: int


# --------------------------------------------------------------------------
# Sampling primitives
# --------------------------------------------------------------------------


def keygen(params: RlweParams, rng: random.Random) -> list[int]:
    """Sample an LWE secret key ``s`` uniformly from ``{-1, 0, 1}^d``.

    Returned values are in ``{-1, 0, 1}`` (NOT mod ``q``); inner-product
    callers handle modular reduction. This is what makes Stage 5's
    ``s_tilde = sum s[i] * X^i`` have small inspectable coefficients.
    """
    return [rng.choice((-1, 0, 1)) for _ in range(params.d)]


def sample_noise(params: RlweParams, rng: random.Random) -> int:
    """Discrete-Gaussian noise sample: ``round(N(0, sigma))``.

    Returned as a signed int (centered at 0). The caller is responsible for
    reducing modulo ``q`` at the appropriate point (typically inside the
    ``b`` computation in ``encrypt``).
    """
    return round(rng.gauss(0.0, params.sigma))


# --------------------------------------------------------------------------
# Encrypt / Decrypt
# --------------------------------------------------------------------------


def _inner_product(a: list[int], s: list[int]) -> int:
    """Plain integer inner product (no mod reduction).

    Pulled out as a helper so ``encrypt``, ``decrypt``, and
    ``extract_noise`` all use the exact same arithmetic.
    """
    return sum(ai * si for ai, si in zip(a, s, strict=True))


def encrypt(
    s: list[int], m: int, params: RlweParams, rng: random.Random
) -> LweCiphertext:
    """Symmetric-key LWE encryption of ``m in [0, p)``.

    Per SPEC.md section 1::

        a   <- uniform Z_q^d
        e   <- chi  (sample_noise)
        b   := -<a, s> + e + Delta * m   (mod q)
        ct  := (a, b)
    """
    if not (0 <= m < params.p):
        raise ValueError(
            f"message m = {m} not in plaintext range [0, p={params.p})"
        )
    d, q = params.d, params.q
    a = [rng.randrange(q) for _ in range(d)]
    e = sample_noise(params, rng)
    b = (-_inner_product(a, s) + e + params.delta * m) % q
    return LweCiphertext(a=a, b=b)


def decrypt(s: list[int], ct: LweCiphertext, params: RlweParams) -> int:
    """Symmetric-key LWE decryption.

    Computes ``raw = (b + <a, s>) mod q``, which equals ``Delta * m + e``
    modulo ``q`` (where ``e`` is the encryption noise). Rounds to the
    nearest multiple of ``Delta`` and returns the result mod ``p``.

    Correctness condition (per SPEC.md section 7): ``|e| < Delta / 2``.
    With ``sigma ~ 3.2``, ``Delta ~ q/p`` typically gives ``Delta / 2``
    several orders of magnitude larger than ``6 * sigma``, so a single
    LWE encryption is comfortably decryptable. Subsequent stages add
    more noise and the final pack must still satisfy the bound -- that
    is what ``RlweParams.correctness_ok`` checks.
    """
    q, p, delta = params.q, params.p, params.delta
    raw = (ct.b + _inner_product(ct.a, s)) % q
    return ((raw + delta // 2) // delta) % p


def extract_noise(
    s: list[int], ct: LweCiphertext, m: int, params: RlweParams
) -> int:
    """Recover the residual noise ``e`` from a known-plaintext ciphertext.

    Computes ``(b + <a, s> - Delta * m) mod q`` then maps to the centered
    representative in ``(-q/2, q/2]``. By construction this equals the
    ``e`` that was sampled inside ``encrypt`` (assuming ``|e| < q/2``,
    which holds with overwhelming probability for valid ``sigma``).

    Used by:
      * ``test_lwe.py`` -- empirical noise statistics (mean ~ 0, std ~ sigma).
      * Stage 7's ``test_key_switching.py`` -- per-step noise budget.
      * Stage 14's ``test_noise.py`` -- end-to-end noise vs Theorem 2.
    """
    q, delta = params.q, params.delta
    raw = (ct.b + _inner_product(ct.a, s) - delta * m) % q
    return raw if raw <= q // 2 else raw - q
