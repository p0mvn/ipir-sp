"""Key switching: change which secret a ciphertext is "under" (SPEC.md sections 6, 7).

Given an RLWE ciphertext ``(a, b)`` under secret ``s_in`` -- meaning
``b = -a * s_in + m + e_old (mod q)`` decrypts back to ``m`` only with
``s_in`` -- ``KS.Switch`` produces a new ciphertext ``(a', b')`` under a
different secret ``s_out`` encrypting **the same** ``m``, plus an
additive noise ``e_ks``. The transformation requires a precomputed
"key-switching matrix" ``K = KS.Setup(s_in, s_out)``.

API surface:

* ``KeySwitchingMatrix`` -- ``(w, y, noise)`` triple, length ``ell``.
* ``setup(s_in, s_out, params, rng) -> KeySwitchingMatrix``.
* ``switch(a, b, K, params) -> (a', b')``.
* ``apply_automorph(K, g, params) -> KeySwitchingMatrix`` -- the InspiRING-specific
  trick that produces ``rho(K) = K_for(rho(s_in) -> rho(s_out))`` from a
  base ``K`` by applying ``tau_g`` entry-wise. **The reason InspiRING needs
  only 2 KS matrices instead of CDKS's log d.**
* ``reset_switch_counter`` / ``switch_call_count`` -- a module-level call
  counter for the ``d - 1`` invariant check in Stage 12.

KS.Setup formula -- per SPEC.md section 6 / paper §2::

    For i = 0, ..., ell - 1:
      w[i] <- uniform R_q
      e[i] <- chi^d
      y[i] := -s_out * w[i] + s_in * z^i + e[i]   (mod q)

    K := [w, y]  (each is a length-ell list of length-d polynomials)

KS.Switch formula::

    digits = g_z^{-1}(a)              # ell signed-digit polynomials
    a'     = sum_i (digits[i] * w[i]) (mod q)
    b'     = b + sum_i (digits[i] * y[i]) (mod q)

Why this works (one-line derivation, SPEC.md section 7)::

    b' = b + sum_i digits[i] * y[i]
       = b + sum_i digits[i] * (-s_out * w[i] + s_in * z^i + e[i])
       = b - s_out * a' + s_in * (sum_i z^i * digits[i]) + e_ks
       = b - s_out * a' + s_in * a + e_ks                 [gadget identity]
       = (-a * s_in + m + e_old) - s_out * a' + s_in * a + e_ks
       = -a' * s_out + m + (e_old + e_ks)                 done.

So the message ``m`` is preserved; the secret changes from ``s_in`` to
``s_out``; the noise grows by ``e_ks = sum_i digits[i] * e[i]``, bounded
in variance by ``ell * d * z^2 * sigma^2 / 4`` per coefficient (Theorem 2,
single-step).

The automorphism trick (SPEC.md section 6, "Critical observation"):
Applying ``tau_g`` entry-wise to ``K = [w, y]`` is itself a KS matrix.
Concretely, ``tau_g`` is a ring homomorphism, so::

    tau_g(y[i]) = -tau_g(s_out) * tau_g(w[i]) + tau_g(s_in) * tau_g(z^i) + tau_g(e[i])
                = -tau_g(s_out) * w'[i] + tau_g(s_in) * z^i + tau_g(e[i])

(The scalar ``z^i`` is fixed by every automorphism since it's a constant
polynomial.) So ``tau_g(K)`` is a valid KS matrix from ``tau_g(s_in)``
to ``tau_g(s_out)``, with noise ``tau_g(e[i])`` -- coefficient-permuted
with sign flips, **same subgaussian parameter** as ``e[i]``.

This is what lets ``CollapseHalf`` (Stage 11) iterate ``d/2 - 1`` times
using only the **single** base matrix ``K_g``, by applying
``tau_g^{k-1}`` to it at each step. CDKS, by contrast, needs ``log d``
distinct base matrices (one per binary-tree level).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from inspiring_oracle.automorph import tau
from inspiring_oracle.gadget import gz_inv_poly
from inspiring_oracle.params import RlweParams
from inspiring_oracle.ring import add, mul, neg, scalar_mul
from inspiring_oracle.rlwe import sample_noise_poly


@dataclass(frozen=True)
class KeySwitchingMatrix:
    """A key-switching matrix ``K = [w, y]`` with bookkeeping noise.

    All three lists have length ``ell``, with each entry a length-``d``
    polynomial:

    * ``w[i]`` -- uniform random in ``R_q``, kept in canonical ``[0, q)`` form.
    * ``y[i]`` -- the formula ``-s_out * w[i] + s_in * z^i + e[i] mod q``.
    * ``noise[i]`` -- the **signed centered** noise polynomial sampled
      inside ``y[i]``, exposed for noise-statistics tests and Stage 15
      fixtures. Coefficients have ``|c| << q/2`` typically.
    """

    w: list[list[int]]
    y: list[list[int]]
    noise: list[list[int]]


# --------------------------------------------------------------------------
# Module-level call counter (SPEC.md section 6: "exactly d-1 calls per pack")
# --------------------------------------------------------------------------

_switch_call_count: int = 0


def reset_switch_counter() -> None:
    """Reset the global ``KS.Switch`` call counter to 0.

    Tests that need to count ``KS.Switch`` invocations (Stages 7, 11, 12)
    call this at the start. The counter is module-global; not thread-safe,
    but pytest doesn't parallelize tests by default.
    """
    global _switch_call_count
    _switch_call_count = 0


def switch_call_count() -> int:
    """Return the global ``KS.Switch`` call count since the last reset."""
    return _switch_call_count


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def setup(
    s_in: list[int],
    s_out: list[int],
    params: RlweParams,
    rng: random.Random,
) -> KeySwitchingMatrix:
    """Generate a KS matrix that switches ciphertexts from ``s_in`` to ``s_out``.

    Each of the ``ell`` rows is an independent fresh RLWE encryption of
    ``s_in * z^i`` under ``s_out`` (with the same negative-product sign
    convention as ``rlwe.encrypt``).
    """
    if len(s_in) != params.d or len(s_out) != params.d:
        raise ValueError(
            f"s_in (len {len(s_in)}) and s_out (len {len(s_out)}) "
            f"must both have length d = {params.d}"
        )
    d, q, z, ell = params.d, params.q, params.z, params.ell

    w_rows: list[list[int]] = []
    y_rows: list[list[int]] = []
    noise_rows: list[list[int]] = []
    for i in range(ell):
        w_i = [rng.randrange(q) for _ in range(d)]
        e_i = sample_noise_poly(params, rng)
        e_i_modq = [v % q for v in e_i]
        z_pow_i = pow(z, i, q)
        s_in_zi = scalar_mul(s_in, z_pow_i, q)
        # y_i = -s_out * w_i + s_in * z^i + e_i  (all mod q)
        y_i = add(
            add(neg(mul(s_out, w_i, q), q), s_in_zi, q),
            e_i_modq,
            q,
        )
        w_rows.append(w_i)
        y_rows.append(y_i)
        noise_rows.append(e_i)

    return KeySwitchingMatrix(w=w_rows, y=y_rows, noise=noise_rows)


# --------------------------------------------------------------------------
# Switch
# --------------------------------------------------------------------------


def switch(
    a_poly: list[int],
    b_poly: list[int],
    K: KeySwitchingMatrix,
    params: RlweParams,
) -> tuple[list[int], list[int]]:
    """Apply ``KS.Switch`` to ``(a_poly, b_poly)``: return ``(a', b')`` under
    the new secret implied by ``K``.

    Increments the global ``KS.Switch`` call counter; reset between tests
    via ``reset_switch_counter()``.
    """
    global _switch_call_count
    _switch_call_count += 1

    d, q, ell = params.d, params.q, params.ell
    digits = gz_inv_poly(a_poly, params)
    a_new = [0] * d
    b_new = b_poly[:]
    for i in range(ell):
        a_new = add(a_new, mul(digits[i], K.w[i], q), q)
        b_new = add(b_new, mul(digits[i], K.y[i], q), q)
    return a_new, b_new


# --------------------------------------------------------------------------
# Automorphism trick (SPEC.md section 6, "Critical observation")
# --------------------------------------------------------------------------


def _tau_signed(p_signed: list[int], g: int, d: int) -> list[int]:
    """Apply ``tau_g`` to a polynomial whose coefficients are signed ints.

    Operates entirely in ``Z`` (no mod-``q`` reduction), so small noise
    coefficients stay small. ``tau`` is a coefficient permutation with
    sign flips, so per-coefficient magnitudes are exactly preserved.
    """
    out = [0] * d
    two_d = 2 * d
    for i, c in enumerate(p_signed):
        if c == 0:
            continue
        e = (i * g) % two_d
        if e < d:
            out[e] += c
        else:
            out[e - d] -= c
    return out


def apply_automorph(
    K: KeySwitchingMatrix, g: int, params: RlweParams
) -> KeySwitchingMatrix:
    """Return ``rho(K)`` where ``rho = tau_g``.

    If ``K = setup(s_in, s_out, ...)`` then ``apply_automorph(K, g, ...)``
    is a valid KS matrix from ``tau_g(s_in)`` to ``tau_g(s_out)`` with
    noise ``tau_g(e[i])``. **No fresh randomness, no fresh noise** --
    just an entry-wise application of ``tau_g``.

    This is the cornerstone of Stage 11 (``CollapseHalf``): every iteration
    of the cascade uses ``apply_automorph(K_g, G^{k-1}, ...)`` to get a
    KS matrix that switches ``tau_g^k(s_tilde) -> tau_g^{k-1}(s_tilde)``,
    without ever needing more than the single base matrix ``K_g``.
    """
    q, d = params.q, params.d
    return KeySwitchingMatrix(
        w=[tau(w_i, g, q) for w_i in K.w],
        y=[tau(y_i, g, q) for y_i in K.y],
        noise=[_tau_signed(e_i, g, d) for e_i in K.noise],
    )
