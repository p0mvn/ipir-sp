"""Algorithm 1 stages 1 and 2: ``TRANSFORM`` and ``aggregate``.

This is the first module that implements **InspiRING's Algorithm 1 itself**
(SPEC.md sections 4 and 5). The substrate stages 1-7 give us the
mathematical and cryptographic primitives; this module begins assembling
them into the actual packing algorithm.

Stages 1 and 2 of Algorithm 1 are both **noise-free algebraic
rearrangements** -- no key switching, no fresh randomness, no new noise.
All algorithm noise enters at Stage 3 (``collapse``, Stages 10-12 of the
oracle plan), via the ``d - 1`` ``KS.Switch`` calls.

Algorithm 1 stage 1 -- ``transform`` (this stage):
    Convert a single LWE ciphertext ``(a, b) in Z_q^d x Z_q`` into an
    intermediate ciphertext ``IRCtx(m_hat) = (a_hat, b_tilde) in R_q^d x R_q``.
    The conversion uses the trace operator (Lemma 1) to put the LWE
    message ``m'`` into the constant slot of a polynomial ``m_hat``, with
    every other slot exactly zero.

Algorithm 1 stage 2 -- ``aggregate`` (added in oracle Stage 9):
    Combine ``d`` IRCtxs into one whose ``m_hat`` polynomial encodes one
    plaintext per coefficient slot.

The IRCtx data type:

    IRCtx(m_hat) = (a_hat, b_tilde) in R_q^d x R_q

    is encrypted under the wider secret ``s_hat in R_q^d`` (built from
    the base RLWE secret ``s_tilde`` via Galois automorphisms; see
    ``decrypt_under_s_hat.py``). The relation::

        b_tilde = -<a_hat, s_hat> + m_hat   (mod q)

    For a fresh ``transform`` output, ``m_hat = (e + Delta * m) * X^0`` --
    a constant polynomial. Aggregation combines multiple constant ``m_hat``
    polynomials into a single non-constant one.

The ``transform`` formula (SPEC.md section 4 / paper Algorithm 1)::

    a_tilde[0]   := a[0]                             # negative-exponent
    a_tilde[d-i] := -a[i]    for i in [1, d)         # embedding of a

    b_tilde      := [b, 0, 0, ..., 0]                # constant polynomial

    For j in [0, d/2):
        a_hat[j]         := d^{-1} * tau_g^j(a_tilde)
        a_hat[j + d/2]   := d^{-1} * tau_h(tau_g^j(a_tilde))

    Return IRCtx(a_hat, b_tilde)

Why the negative-exponent embedding ``a_tilde[d-i] = -a[i]``? Because
``(a_tilde * s_tilde)[0] == <a, s>`` in ``R_q``. The ring product's
**constant coefficient** equals the LWE inner product. This is what makes
the rest of the derivation collapse to a clean "message in slot 0 only"
form when we apply Lemma 1's trace.

The d^{-1} scaling absorbs the trace's ``d * (...)`` factor (Lemma 1
gives ``Tr(p) = d * p[0]``). Since ``q`` is odd, ``d^{-1} mod q`` exists --
this is the reason ``RlweParams`` requires odd ``q``.

The d^{-1} factor lives entirely in ``a_hat`` (a uniformly random
component); the message and noise survive at their original scale, so
Stage 1 is **truly noise-free** even though ``d^{-1} mod q`` is a large
modular integer.
"""

from __future__ import annotations

from dataclasses import dataclass

from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.lwe import LweCiphertext
from inspiring_oracle.params import RlweParams
from inspiring_oracle.ring import scalar_mul


@dataclass(frozen=True)
class IRCtx:
    """Intermediate ciphertext ``(a_hat, b_tilde) in R_q^d x R_q``.

    Encrypted under the wider secret ``s_hat in R_q^d`` derived from
    ``s_tilde`` via Galois automorphisms. See
    ``inspiring_oracle.decrypt_under_s_hat`` for the secret-derivation and
    decryption helpers (which exist only for tests / fixture validation;
    in production the IRCtx is consumed by ``Collapse`` instead).

    Attributes:
      a_hat: Length-``d`` list of length-``d`` polynomials, each in ``[0, q)``.
        Depends only on the input ``a``, not on ``b`` -- see
        ``test_transform.py::TestAHatIndependentOfB`` (basis of the
        offline/online split in SPEC.md section 8).
      b_tilde: Length-``d`` polynomial in ``[0, q)``. For a fresh
        ``transform`` output, position 0 holds ``b mod q`` and every other
        position is exactly ``0``.
    """

    a_hat: list[list[int]]
    b_tilde: list[int]


def transform(lwe: LweCiphertext, params: RlweParams) -> IRCtx:
    """Stage 1 of Algorithm 1 (SPEC.md section 4): convert LWE -> IRCtx.

    Input:  ``(a, b) in Z_q^d x Z_q`` with ``b = -<a, s> + e + Delta * m``.

    Output: ``IRCtx(a_hat, b_tilde)`` such that decrypting under
    ``s_hat = s_hat_from_s_tilde(s_tilde)`` recovers the constant
    polynomial ``[Delta * m + e, 0, 0, ..., 0]`` -- the LWE message and
    noise sitting in slot 0 of an otherwise-zero plaintext polynomial.

    Adds **no** new noise: the original LWE noise ``e`` survives unchanged
    in slot 0 of the recovered ``m_hat``; all higher slots of the
    decrypted polynomial are exactly ``0`` (in canonical ``[0, q)`` form),
    not "approximately zero" -- the trace operator is a Z-linear
    rearrangement, not a noisy operation.
    """
    d, q = params.d, params.q
    a, b = lwe.a, lwe.b

    # Negative-exponent embedding of a.
    # Since X^{-i} = -X^{d-i} in R_q (because X^d = -1), we have
    # a_tilde = sum a[i] * X^{-i} = a[0] + sum_{i>0} (-a[i]) * X^{d-i}.
    a_tilde = [0] * d
    a_tilde[0] = a[0] % q
    for i in range(1, d):
        a_tilde[d - i] = (-a[i]) % q

    # b_tilde = b * X^0 (constant polynomial).
    b_tilde = [b % q] + [0] * (d - 1)

    # a_hat[j]       = d^{-1} * tau_g^j(a_tilde)            for j in [0, d/2)
    # a_hat[j + d/2] = d^{-1} * tau_h(tau_g^j(a_tilde))     for j in [0, d/2)
    a_hat: list[list[int]] = [[] for _ in range(d)]
    two_d = 2 * d
    h_d = h(d)
    for j in range(d // 2):
        gj = pow(G, j, two_d)
        a_hat[j] = scalar_mul(tau(a_tilde, gj, q), params.d_inv, q)
        a_hat[j + d // 2] = scalar_mul(
            tau(a_tilde, (gj * h_d) % two_d, q), params.d_inv, q
        )

    return IRCtx(a_hat=a_hat, b_tilde=b_tilde)
