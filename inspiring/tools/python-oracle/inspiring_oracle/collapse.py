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

from inspiring_oracle import key_switching
from inspiring_oracle.key_switching import KeySwitchingMatrix
from inspiring_oracle.params import RlweParams
from inspiring_oracle.ring import add


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
