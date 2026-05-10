"""Algorithm 1: ``InspiRING.Pack`` -- the top-level entry point.

This module is the **headline deliverable** of the oracle: a single
function ``pack(lwes, K_g, K_h, params) -> RlweCiphertext`` that
implements the full Algorithm 1 of the InsPIRe paper (eprint 2025/1352).

The implementation is just three lines plus boilerplate -- almost all of
the work lives in the lower stages:

1. ``transform(lwe)`` -- Stage 1 (``intermediate.py``, oracle Stage 8).
   Convert each of the ``d`` input LWE ciphertexts into an IRCtx via
   the trace operator. Noise-free.
2. ``aggregate(irctxs)`` -- Stage 2 (``intermediate.py``, oracle Stage 9).
   Combine the ``d`` IRCtxs into one whose ``m_hat`` polynomial encodes
   one plaintext per coefficient. Noise-free.
3. ``collapse(ictx_agg, K_g, K_h)`` -- Stage 3 (``collapse.py``, oracle
   Stage 12). Run ``d - 1`` ``KS.Switch`` calls to fold the wider secret
   ``s_hat`` down to the base secret ``s_tilde``. **All algorithm noise
   enters here.**

Per SPEC.md sections 4-6:

    PACK(lwe_0, ..., lwe_{d-1}, K_g, K_h) -> RlweCiphertext:
        for k in [0, d):
            ictx_k := TRANSFORM(lwe_k)
        ictx_agg := AGGREGATE(ictx_0, ..., ictx_{d-1})
        return COLLAPSE(ictx_agg, K_g, K_h)

Inputs / outputs:

* Inputs: ``d`` LWE ciphertexts ``(a_k, b_k) in Z_q^d x Z_q``
  encrypting messages ``m_k in Z_p`` under a shared LWE secret ``s``,
  plus the two CRS-fixed key-switching matrices ``K_g`` and ``K_h``
  derived from ``s_tilde = s_tilde_from_s(s, params)``.
* Output: a single RLWE ciphertext ``(c_1, c_2)`` under ``s_tilde``
  encrypting the message polynomial ``m_packed = sum_k m_k * X^k``.

The ``KS.Switch`` cost is exactly ``d - 1`` (vs CDKS's ``(d - 1) * log d``
for the equivalent operation). This count is asserted by
``test_pack_roundtrip.py::TestSwitchCallCount``.
"""

from __future__ import annotations

from inspiring_oracle.collapse import collapse
from inspiring_oracle.intermediate import aggregate, transform
from inspiring_oracle.key_switching import KeySwitchingMatrix
from inspiring_oracle.lwe import LweCiphertext
from inspiring_oracle.params import RlweParams
from inspiring_oracle.rlwe import RlweCiphertext


def pack(
    lwes: list[LweCiphertext],
    K_g: KeySwitchingMatrix,
    K_h: KeySwitchingMatrix,
    params: RlweParams,
) -> RlweCiphertext:
    """Algorithm 1 of the InsPIRe paper -- pack ``d`` LWEs into one RLWE.

    Args:
      lwes: Exactly ``d`` LWE ciphertexts encrypting messages
        ``m_0, ..., m_{d-1}``, all under the same LWE secret ``s``.
      K_g: ``KS.Setup(tau_g(s_tilde), s_tilde)`` -- one of the two
        CRS-fixed base key-switching matrices.
      K_h: ``KS.Setup(tau_h(s_tilde), s_tilde)`` -- the other.
      params: Ring parameters.

    Returns:
      ``RlweCiphertext(c1, c2)`` under ``s_tilde``, encrypting
      ``m_packed = sum_{k=0}^{d-1} m_k * X^k``. Decrypt with
      ``rlwe.decrypt(s_tilde, ct, params)``.

    Raises:
      ValueError: if ``len(lwes) != params.d``.

    Cost:
      * ``d`` independent ``transform`` calls (no noise, embarrassingly
        parallel).
      * 1 ``aggregate`` call (no noise, ``O(d^3)`` ring ops).
      * 1 ``collapse`` call -- exactly ``d - 1`` ``KS.Switch``
        invocations, each adding ``O(d * z * sigma)`` noise.

      Total noise budget: ``sigma_pack^2 <= ell * d^2 * z^2 * sigma^2 / 4``
      (Theorem 2). Validated empirically by Stage 14's
      ``test_noise.py``.
    """
    if len(lwes) != params.d:
        raise ValueError(
            f"pack expects exactly d={params.d} LWE ciphertexts, "
            f"got {len(lwes)}"
        )

    irctxs = [transform(lwe, params) for lwe in lwes]
    ictx_agg = aggregate(irctxs, params)
    return collapse(ictx_agg, K_g, K_h, params)
