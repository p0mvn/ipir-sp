"""Test-only helpers: decrypt an IRCtx under the wider secret ``s_hat``.

These helpers exist to verify the **internal** correctness of Algorithm 1's
stages 1 and 2. In production, an IRCtx is never decrypted directly --
it is consumed by Stage 3 (``Collapse``) which produces an honest
two-element RLWE ciphertext that gets decrypted under ``s_tilde`` via
``rlwe.decrypt``.

The wider secret ``s_hat in R_q^d`` is built from the base RLWE secret
``s_tilde`` by applying the same Galois automorphism pattern that
``transform`` uses for ``a_hat`` (SPEC.md section 4)::

    s_hat[j]         = tau_g^j(s_tilde)              for j in [0, d/2)
    s_hat[j + d/2]   = tau_h(tau_g^j(s_tilde))       for j in [0, d/2)

Note that ``s_hat[0] = tau_g^0(s_tilde) = s_tilde``: the wider secret's
first slot is the base secret unchanged.

The decryption relation for an IRCtx ``(a_hat, b_tilde)``::

    b_tilde + <a_hat, s_hat> = m_hat   (mod q)

For a fresh ``transform(lwe)`` output, ``m_hat`` is the constant
polynomial ``(Delta * m + e) * X^0`` -- the LWE message scaled and
noised, in slot 0; every other slot exactly zero.

For an aggregated IRCtx (Stage 9), ``m_hat`` is non-constant: each slot
``k`` holds ``Delta * m_k + e_k`` for the ``k``-th LWE input.
"""

from __future__ import annotations

from inspiring_oracle.automorph import G, h, tau
from inspiring_oracle.intermediate import IRCtx
from inspiring_oracle.params import RlweParams
from inspiring_oracle.ring import add, mul


def s_hat_from_s_tilde(s_tilde: list[int], params: RlweParams) -> list[list[int]]:
    """Build the wider secret ``s_hat in R_q^d`` from ``s_tilde``.

    Per SPEC.md section 4::

        s_hat[j]         = tau_g^j(s_tilde)              for j in [0, d/2)
        s_hat[j + d/2]   = tau_h(tau_g^j(s_tilde))       for j in [0, d/2)

    ``s_hat[0] = s_tilde`` (since ``tau_g^0`` is the identity).
    """
    d, q = params.d, params.q
    s_hat: list[list[int]] = [[] for _ in range(d)]
    two_d = 2 * d
    h_d = h(d)
    for j in range(d // 2):
        gj = pow(G, j, two_d)
        s_hat[j] = tau(s_tilde, gj, q)
        s_hat[j + d // 2] = tau(s_tilde, (gj * h_d) % two_d, q)
    return s_hat


def decrypt_polynomial_under_s_hat(
    ictx: IRCtx,
    s_hat: list[list[int]],
    params: RlweParams,
) -> list[int]:
    """Compute ``b_tilde + <a_hat, s_hat>`` mod q -- the recovered ``m_hat``.

    Returns the raw plaintext polynomial in canonical ``[0, q)`` form,
    **without** any rounding to the message space ``Z_p``.

    For a fresh ``transform`` output this should be exactly
    ``[Delta * m + e, 0, 0, ..., 0]`` (slot 0 noisy, all other slots zero
    in canonical form). Use ``decrypt_under_s_hat`` to round to ``Z_p``.
    """
    d, q = params.d, params.q
    raw = ictx.b_tilde[:]
    for j in range(d):
        raw = add(raw, mul(ictx.a_hat[j], s_hat[j], q), q)
    return raw


def decrypt_under_s_hat(
    ictx: IRCtx,
    s_hat: list[list[int]],
    params: RlweParams,
) -> list[int]:
    """Decrypt an IRCtx and return per-slot plaintexts in ``Z_p``.

    Each output slot is rounded to the nearest multiple of ``Delta`` and
    reduced mod ``p``. For a fresh ``transform`` output, slot 0 returns
    the original LWE message ``m`` and every other slot returns ``0``.
    """
    delta, p = params.delta, params.p
    raw = decrypt_polynomial_under_s_hat(ictx, s_hat, params)
    return [((c + delta // 2) // delta) % p for c in raw]


def extract_noise_under_s_hat(
    ictx: IRCtx,
    s_hat: list[list[int]],
    m_bar: list[int],
    params: RlweParams,
) -> list[int]:
    """Recover the per-slot residual noise from a known-plaintext IRCtx.

    Each output coefficient is in **centered** representation
    ``[-q/2, q/2]``, suitable for noise-statistics tests (Stage 14) and
    for inspecting noise growth across pipeline stages.
    """
    q, delta = params.q, params.delta
    raw = decrypt_polynomial_under_s_hat(ictx, s_hat, params)
    delta_m = [(delta * mi) % q for mi in m_bar]
    e_modq = [(r - dm) % q for r, dm in zip(raw, delta_m, strict=True)]
    return [c if c <= q // 2 else c - q for c in e_modq]
