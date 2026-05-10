"""Test-only helpers for "wider" RLWE-style ciphertexts.

A **wider ciphertext** ``(a, b) in R_q^k x R_q`` generalises a normal
two-element RLWE ciphertext (``k = 1``) to any positive ``k``. It is
encrypted under a wider secret ``s_wide in R_q^k`` and decrypts via::

    b + sum_{i=0}^{k-1} a[i] * s_wide[i] = m_bar + e   (mod q)

i.e. the analogue of the two-element relation ``b + a * s = m + e``,
just with a length-``k`` inner product instead of a single scalar
product. With ``k = 1`` and ``s_wide = [s_tilde]`` this reduces exactly
to standard RLWE.

This abstraction is not a "production" data type: the InspiRING crate
never exposes it. It exists only to make ``Collapse`` (Stages 10-12)
testable. A wider ciphertext is the natural intermediate representation
*during* ``Collapse``, and the tests need to:

* Construct a wider ciphertext under a chosen wider secret with a chosen
  message and known noise (``build_wide_ciphertext``).
* Decrypt one under a wider secret to recover its plaintext slots
  (``decrypt_wide``).
* Extract the residual noise polynomial after partial collapse to
  validate the per-step noise-growth claim (``extract_wide_noise``).

These are deliberately the wider-secret analogues of
``decrypt_under_s_hat.py`` -- same file role, different secret structure.
"""

from __future__ import annotations

import random

from inspiring_oracle.params import RlweParams
from inspiring_oracle.ring import add, mul, sub
from inspiring_oracle.rlwe import sample_noise_poly


def build_wide_ciphertext(
    s_wide: list[list[int]],
    m_bar: list[int],
    params: RlweParams,
    rng: random.Random,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Construct a wider ciphertext encrypting ``m_bar`` under ``s_wide``.

    Builds ``a in R_q^k`` uniformly at random and computes::

        b = -sum_{i=0}^{k-1} a[i] * s_wide[i] + e + Delta * m_bar  (mod q)

    where ``e`` is a fresh discrete-Gaussian noise polynomial (with
    coefficients in **signed** centered form, returned for noise-tracking
    tests).

    Returns ``(a, b, e_centered)`` where ``e_centered`` is the centered
    noise polynomial so callers can verify e_after_collapse against
    e_before_collapse.
    """
    k = len(s_wide)
    if k < 1:
        raise ValueError(f"s_wide must be non-empty, got k = {k}")
    d, q, delta = params.d, params.q, params.delta
    if len(m_bar) != d:
        raise ValueError(f"len(m_bar) = {len(m_bar)} != d = {d}")

    a = [[rng.randrange(q) for _ in range(d)] for _ in range(k)]
    e_centered = sample_noise_poly(params, rng)
    e_modq = [ei % q for ei in e_centered]
    delta_m = [(delta * mi) % q for mi in m_bar]

    b = add(e_modq, delta_m, q)
    for i in range(k):
        b = sub(b, mul(a[i], s_wide[i], q), q)

    return a, b, e_centered


def decrypt_polynomial_wide(
    a: list[list[int]],
    b: list[int],
    s_wide: list[list[int]],
    params: RlweParams,
) -> list[int]:
    """Compute ``b + <a, s_wide>`` mod q -- the recovered ``m_bar + e`` polynomial."""
    k = len(a)
    if len(s_wide) != k:
        raise ValueError(f"len(a) = {k} != len(s_wide) = {len(s_wide)}")
    q = params.q
    raw = b[:]
    for i in range(k):
        raw = add(raw, mul(a[i], s_wide[i], q), q)
    return raw


def decrypt_wide(
    a: list[list[int]],
    b: list[int],
    s_wide: list[list[int]],
    params: RlweParams,
) -> list[int]:
    """Decrypt a wider ciphertext and return per-slot plaintexts in Z_p."""
    delta, p = params.delta, params.p
    raw = decrypt_polynomial_wide(a, b, s_wide, params)
    return [((c + delta // 2) // delta) % p for c in raw]


def extract_wide_noise(
    a: list[list[int]],
    b: list[int],
    s_wide: list[list[int]],
    m_bar: list[int],
    params: RlweParams,
) -> list[int]:
    """Recover the residual noise polynomial from a known-plaintext wider ciphertext.

    Returns the noise in **centered** representation ``[-q/2, q/2]``,
    suitable for noise-budget tracking across stages.
    """
    q, delta = params.q, params.delta
    raw = decrypt_polynomial_wide(a, b, s_wide, params)
    delta_m = [(delta * mi) % q for mi in m_bar]
    e_modq = [(r - dm) % q for r, dm in zip(raw, delta_m, strict=True)]
    return [c if c <= q // 2 else c - q for c in e_modq]
