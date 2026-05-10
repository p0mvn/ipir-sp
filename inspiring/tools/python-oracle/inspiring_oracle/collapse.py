"""Algorithm 1 stage 3: ``Collapse`` -- IRCtx -> two-element RLWE.

This module implements the third and final stage of InspiRING.Pack
(SPEC.md section 6). Unlike Stages 1 and 2 (algebraic rearrangements,
noise-free), Stage 3 is **where every byte of algorithm noise enters**.
It runs exactly ``d - 1`` ``KS.Switch`` calls, each contributing one
unit of key-switching noise.

The stage is built up in three pieces, added across oracle Stages 10-12:

* ``collapse_one`` (this stage) -- the atomic step. Reduces a wider
  ciphertext ``(a, b) in R_q^k x R_q`` to ``R_q^{k-1} x R_q`` via one
  ``KS.Switch``. The newly-shortened ciphertext encrypts the same
  message under a wider secret with one fewer component.
* ``collapse_half`` (oracle Stage 11) -- iteratively applies
  ``collapse_one`` ``d/2 - 1`` times to fold an entire half-vector down
  to a single component.
* ``collapse`` (oracle Stage 12) -- runs ``collapse_half`` on each half
  of an aggregated IRCtx, then a final ``collapse_one`` with ``K_h`` to
  fuse the two halves and produce a single RLWE ciphertext under the
  base secret ``s_tilde``.

The InspiRING innovation -- needing only **two** base key-switching
matrices ``K_g`` and ``K_h`` instead of CDKS's ``log d`` -- is realised
in Stage 11 via ``key_switching.apply_automorph`` (Stage 7). Stage 10
itself is parametric in the supplied ``K``: it never inspects how the
matrix was derived.

``CollapseOne`` formula (SPEC.md section 6 / paper Algorithm 1)::

    COLLAPSEONE((a, b) in R_q^k x R_q,
                K = [w, y]) -> (a', b') in R_q^{k-1} x R_q:
        # K switches secret share s'_{k-1} into share s'_{k-2}.
        (Da, Db) := KS.Switch((a[k-1], b), K)
        a' := (a[0], a[1], ..., a[k-3], a[k-2] + Da)
        b' := Db                                 # already includes b
        return (a', b')

Why this works (one-line derivation, expanding KS.Switch's invariant)::

    Db = b + sum_i digits[i] * y[i]
       = b - s'_{k-2} * Da + s'_{k-1} * a[k-1] + e_ks
       = -<a, s'> + m + e_old + e_ks - s'_{k-2} * Da + s'_{k-1} * a[k-1]
       = -<a[:k-1], s'[:k-1]> - (a[k-2] + Da) * s'_{k-2} + m + e_old + e_ks
       = -<a', s''> + m + e_old + e_ks         (where s'' := s'[:k-1])

So the wider secret loses its last component ``s'_{k-1}``; the message
``m`` and the noise contribution ``e_old`` from before are preserved;
new noise ``e_ks`` is added (the ``KS.Switch`` noise from Stage 7).
"""

from __future__ import annotations

from typing import Literal

from inspiring_oracle import key_switching
from inspiring_oracle.automorph import G, h
from inspiring_oracle.intermediate import IRCtx
from inspiring_oracle.key_switching import KeySwitchingMatrix
from inspiring_oracle.params import RlweParams
from inspiring_oracle.ring import add
from inspiring_oracle.rlwe import RlweCiphertext

# Selector for the "rho" automorphism in CollapseHalf (SPEC.md section 6).
# "identity" -> rho is the identity (used for the LEFT half of an aggregated
#               IRCtx: secret pattern s_hat[j] = tau_g^j(s_tilde)).
# "tau_h"    -> rho = tau_h (used for the RIGHT half: secret pattern
#               s_hat[j + d/2] = tau_h(tau_g^j(s_tilde))).
RhoChoice = Literal["identity", "tau_h"]


def collapse_one(
    a: list[list[int]],
    b: list[int],
    K: KeySwitchingMatrix,
    params: RlweParams,
) -> tuple[list[list[int]], list[int]]:
    """Reduce a wider ciphertext by one component via one ``KS.Switch``.

    Args:
      a: Length-``k`` list of length-``d`` polynomials (``k >= 2``).
        The "wider random component" of the input ciphertext.
      b: Length-``d`` polynomial -- the running ``b`` value carrying the
        message and accumulated noise.
      K: A key-switching matrix that switches the wider secret's last
        share ``s'_{k-1}`` into its second-to-last share ``s'_{k-2}``.
        I.e. ``K = KS.Setup(s'_{k-1}, s'_{k-2}, params, rng)`` (or any
        automorphic image thereof, per the Stage 11 trick).
      params: Ring parameters.

    Returns:
      ``(a', b')`` where ``a'`` is length-``(k - 1)`` and ``b'`` is the
      new running ``b``. Encrypted under the shortened wider secret
      ``s'' = (s'_0, ..., s'_{k-2})``, with one fresh ``KS.Switch``'s
      worth of additional noise (analyzed in SPEC.md section 7).

    Raises:
      ValueError: if ``k < 2`` (cannot collapse a single-secret
      ciphertext; that's already the goal state).

    Side effect: increments ``key_switching.switch_call_count()`` by
    exactly 1 (a single ``KS.Switch`` invocation).
    """
    k = len(a)
    if k < 2:
        raise ValueError(
            f"collapse_one requires k >= 2 (got k = {k}); "
            "the ciphertext is already as collapsed as possible"
        )

    delta_a, delta_b = key_switching.switch(a[k - 1], b, K, params)
    # New a' has k - 1 components: indices 0 .. k-3 unchanged, index k-2
    # absorbs delta_a (which carries the contribution of the dropped
    # share s'_{k-1} re-expressed under s'_{k-2}).
    new_a = list(a[: k - 2]) + [add(a[k - 2], delta_a, params.q)]
    return new_a, delta_b


def collapse_half(
    a: list[list[int]],
    b: list[int],
    K_g: KeySwitchingMatrix,
    rho: RhoChoice,
    params: RlweParams,
) -> tuple[list[int], list[int]]:
    """Iteratively ``collapse_one`` an entire half down to a single secret.

    The InspiRING crown jewel: every per-step key-switching matrix is
    derived from the **single** base matrix ``K_g`` by ``apply_automorph``
    (Stage 7). No additional ``KS.Setup`` calls; no ``log d`` distinct
    base matrices like CDKS needs. This is the entire reason InspiRING's
    KS-matrix count drops from CDKS's ``log d`` to **2** (``K_g`` plus
    one final ``K_h`` used by ``collapse``, oracle Stage 12).

    Per SPEC.md section 6::

        COLLAPSEHALF((a, b) in R_q^{d/2} x R_q,
                     K_g = [w_g, y_g] in R_q^{ell x 2},
                     rho in {identity, tau_h}) -> (a, b) in R_q x R_q:
            for k = d/2 - 1, d/2 - 2, ..., 1:
                K_step := rho(tau_g^{k-1}(K_g))
                (a, b) := COLLAPSEONE((a, b), K_step)
            return (a, b)

    Args:
      a: Length-``d/2`` list of length-``d`` polynomials. The "wider
        random component" of the input ciphertext.
      b: Length-``d`` polynomial -- the running ``b`` value.
      K_g: The base key-switching matrix for ``tau_g(s_tilde) -> s_tilde``.
        Generated once by ``KS.Setup(tau_g(s_tilde), s_tilde, params)``
        and reused unchanged across both halves.
      rho:
        * ``"identity"`` -- input is encrypted under ``s_hat_left``,
          where ``s_hat_left[j] = tau_g^j(s_tilde)``. Output is under
          ``s_tilde`` (= ``tau_g^0(s_tilde)``).
        * ``"tau_h"`` -- input is encrypted under ``s_hat_right``,
          where ``s_hat_right[j] = tau_h(tau_g^j(s_tilde))``. Output is
          under ``tau_h(s_tilde)``.
      params: Ring parameters.

    Returns:
      ``(a_out, b_out)`` -- both length-``d`` polynomials, encrypted
      under either ``s_tilde`` (rho = identity) or ``tau_h(s_tilde)``
      (rho = tau_h). Decryption recovers the same plaintext that the
      input wider ciphertext encoded.

    Raises:
      ValueError: if ``len(a) != d/2`` or ``rho`` is not one of the
      two allowed values.

    Side effect: increments ``key_switching.switch_call_count()`` by
    exactly ``d/2 - 1`` (one ``KS.Switch`` per ``collapse_one`` step).
    """
    d = params.d
    half = d // 2
    if len(a) != half:
        raise ValueError(
            f"collapse_half expects len(a) = d/2 = {half}, got {len(a)}"
        )
    if rho not in ("identity", "tau_h"):
        raise ValueError(
            f"rho must be 'identity' or 'tau_h', got {rho!r}"
        )

    two_d = 2 * d
    h_d = h(d)
    cur_a: list[list[int]] = list(a)
    cur_b: list[int] = list(b)

    # k iterates from d/2 - 1 down to 1.
    for k in range(half - 1, 0, -1):
        # Compose the automorphism: tau_g^{k-1} first, then optionally tau_h.
        # Composition rule: tau_alpha . tau_beta = tau_{(alpha * beta) mod 2d}.
        g_exp = pow(G, k - 1, two_d)
        if rho == "tau_h":
            g_exp = (g_exp * h_d) % two_d
        K_step = key_switching.apply_automorph(K_g, g_exp, params)
        cur_a, cur_b = collapse_one(cur_a, cur_b, K_step, params)

    # After the loop cur_a has length 1 (or equals input a if half == 1,
    # which is impossible for any valid params since d >= 4 -> half >= 2).
    # Unwrap to expose the single remaining secret share.
    assert len(cur_a) == 1, f"expected length-1 result, got len {len(cur_a)}"
    return cur_a[0], cur_b


def collapse(
    ictx_agg: IRCtx,
    K_g: KeySwitchingMatrix,
    K_h: KeySwitchingMatrix,
    params: RlweParams,
) -> RlweCiphertext:
    """Stage 3 of Algorithm 1 (SPEC.md section 6) -- the top-level collapse.

    Combines the two halves of an aggregated IRCtx and folds them into a
    single honest two-element RLWE ciphertext under the base secret
    ``s_tilde``.

    Per SPEC.md section 6::

        COLLAPSE((a_hat_agg, b_tilde_agg), K_g, K_h) -> (c_1, c_2):
            a_left  := a_hat_agg[0 : d/2]
            a_right := a_hat_agg[d/2 : d]
            (a_1, b_1) := COLLAPSEHALF((a_left,  b_tilde_agg), K_g, identity)
            (a_2, b_2) := COLLAPSEHALF((a_right, b_1        ), K_g, tau_h)
            (c_1, c_2) := COLLAPSEONE(([a_1, a_2], b_2), K_h)
            return RlweCiphertext(c_1, c_2)

    Total ``KS.Switch`` invocations: ``(d/2 - 1) + (d/2 - 1) + 1 = d - 1``.
    **This count is the central design distinction of InspiRING vs CDKS**
    (CDKS would use ``(d - 1) * log d`` for the same packing) and is
    pinned down at runtime by ``test_collapse.py::TestSwitchCallCount``.

    Args:
      ictx_agg: Output of ``intermediate.aggregate`` -- a single IRCtx
        whose ``m_hat_agg`` polynomial encodes ``d`` plaintexts (one per
        coefficient slot), encrypted under the wider secret ``s_hat``.
      K_g: ``KS.Setup(tau_g(s_tilde), s_tilde, params, rng)``. Used by
        both halves' ``collapse_half`` calls (offline-generated once).
      K_h: ``KS.Setup(tau_h(s_tilde), s_tilde, params, rng)``. Used only
        by the final fold step.
      params: Ring parameters.

    Returns:
      ``RlweCiphertext(c_1, c_2)`` encrypting ``m_hat_agg + e_total``
      under ``s_tilde`` -- ready for ``rlwe.decrypt``.

    The "subtle but important" trick (SPEC.md section 6): the second
    ``collapse_half`` is invoked with ``b_1`` (the running ``b`` from the
    first half) as its starting ``b``, NOT with ``b_tilde_agg`` again.
    This is correct because the message and accumulated noise are all
    carried in the running ``b``; the right half only needs to undo its
    own ``-<a_right, s_hat_right>`` masking.
    """
    d = params.d
    half = d // 2

    a_left = list(ictx_agg.a_hat[:half])
    a_right = list(ictx_agg.a_hat[half:])

    # Left half: folds tau_g^j(s_tilde) shares down to s_tilde.
    a_1, b_1 = collapse_half(a_left, ictx_agg.b_tilde, K_g, "identity", params)

    # Right half: folds tau_h(tau_g^j(s_tilde)) down to tau_h(s_tilde).
    # NOTE the b_1 (not b_tilde_agg!) -- the running b carries everything.
    a_2, b_2 = collapse_half(a_right, b_1, K_g, "tau_h", params)

    # Final fold: collapse the (s_tilde, tau_h(s_tilde)) wider ciphertext
    # via K_h. After this single collapse_one, the result is under s_tilde
    # alone.
    a_pair = [a_1, a_2]
    a_fin_list, b_fin = collapse_one(a_pair, b_2, K_h, params)
    return RlweCiphertext(c1=a_fin_list[0], c2=b_fin)
