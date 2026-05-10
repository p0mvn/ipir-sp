//! Top-level `pack` entry point — Algorithm 1 of [eprint 2025/1352].
//!
//! [eprint 2025/1352]: https://eprint.iacr.org/2025/1352
//!
//! See SPEC.md §8 for the offline/online split, §10 for the symbol table,
//! and §9 for the structural comparison with CDKS.
//!
//! Phase 8 status: online entry point is implemented.

use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};

use crate::collapse::collapse;
use crate::error::InspiringError;
use crate::intermediate::IRCtx;
use crate::lwe::LweBatch;
use crate::preprocess::PackPreprocessed;

/// An RLWE ciphertext under the base secret `s̃`.
///
/// Internally a `[2, 1]` `PolyMatrixNTT` (the spiral-rs convention) wrapped
/// in a newtype so callers can't accidentally mix it up with intermediate
/// pseudo-ciphertexts.
pub struct RlweCiphertext<'a> {
    /// `(c1, c2)` stacked vertically. `inner.rows == 2`, `inner.cols == 1`.
    pub inner: PolyMatrixNTT<'a>,
}

/// `InspiRING.Pack(b, pre) -> RlweCiphertext` — Algorithm 1.
///
/// **Online** entry point. Takes the `d` `b_k` scalars (via [`LweBatch`])
/// and a [`PackPreprocessed`] cache; returns a single RLWE ciphertext under
/// `s̃` that decrypts to `Σ_{k=0}^{d-1} m_k · X^k`.
///
/// API invariants (SPEC.md §10):
///
/// 1. Deterministic: no fresh randomness is sampled in this function.
/// 2. Calls `KS.Switch` exactly `d − 1` times. Asserted by
///    `tests/inspiring_vs_cdks_recursion.rs`.
/// 3. Touches `pre.kg` and `pre.kh` only via their precomputed
///    automorphic images cached in `pre.kg_images_left` and
///    `pre.kg_images_right`.
///
pub fn pack<'a>(
    b: &LweBatch,
    pre: &'a PackPreprocessed<'a>,
) -> Result<RlweCiphertext<'a>, InspiringError> {
    b.validate(pre.params)?;

    let mut b_tilde = PolyMatrixRaw::zero(&pre.params.spiral, 1, 1);
    for (idx, ct) in b.inner.iter().enumerate() {
        b_tilde.get_poly_mut(0, 0)[idx] = ct.b % pre.params.q;
    }

    Ok(collapse(
        pre.params,
        IRCtx {
            a_hat: pre.a_agg.clone(),
            b_tilde,
        },
        &pre.kg_images_left,
        &pre.kg_images_right,
        &pre.kh,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_switching::{ks_call_count, KeySwitchingMatrix};
    use crate::params::GadgetParams;
    use crate::preprocess::PackPreprocessed;
    use spiral_rs::poly::{from_ntt_alloc, to_ntt_alloc};

    fn params() -> crate::RlweParams {
        crate::RlweParams::new(
            8,
            12289,
            4,
            3.2,
            GadgetParams {
                bits_per: 3,
                ell: 5,
            },
        )
        .expect("valid params")
    }

    fn zero_ks<'a>(params: &'a crate::RlweParams) -> KeySwitchingMatrix<'a> {
        KeySwitchingMatrix {
            mat: PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell),
            params,
        }
    }

    fn crs<'a>(params: &'a crate::RlweParams) -> PolyMatrixNTT<'a> {
        let mut raw = PolyMatrixRaw::zero(&params.spiral, params.d, 1);
        for row in 0..params.d {
            for col in 0..params.d {
                raw.get_poly_mut(row, 0)[col] = (row * 3 + col + 1) as u64;
            }
        }
        to_ntt_alloc(&raw)
    }

    fn batch(params: &crate::RlweParams, b_values: &[u64], a_seed: u64) -> LweBatch {
        LweBatch {
            inner: b_values
                .iter()
                .enumerate()
                .map(|(idx, b)| crate::LweCiphertext {
                    a: vec![a_seed + idx as u64; params.d],
                    b: *b,
                })
                .collect(),
        }
    }

    #[test]
    fn pack_uses_preprocessed_a_and_online_b_values() {
        let params = params();
        let crs = crs(&params);
        let pre = PackPreprocessed::build(&params, &crs, zero_ks(&params), zero_ks(&params))
            .expect("valid preprocessing");
        let b_values: Vec<_> = (0..params.d).map(|idx| idx as u64 + 10).collect();
        let left = batch(&params, &b_values, 1);
        let right = batch(&params, &b_values, 999);

        let ct_left = pack(&left, &pre).expect("pack succeeds");
        let ct_right = pack(&right, &pre).expect("pack succeeds");

        assert_eq!(ct_left.inner.as_slice(), ct_right.inner.as_slice());
        let raw = from_ntt_alloc(&ct_left.inner);
        assert_eq!(
            raw.get_poly(1, 0),
            b_values.iter().map(|v| v % params.q).collect::<Vec<_>>()
        );
    }

    #[test]
    fn pack_runs_linear_cascade_switch_count() {
        let params = params();
        let crs = crs(&params);
        let pre = PackPreprocessed::build(&params, &crs, zero_ks(&params), zero_ks(&params))
            .expect("valid preprocessing");
        let b_values: Vec<_> = (0..params.d).map(|idx| idx as u64).collect();
        let batch = batch(&params, &b_values, 0);

        ks_call_count::reset();
        let _ = pack(&batch, &pre).expect("pack succeeds");

        assert_eq!(ks_call_count::get(), (params.d - 1) as u64);
    }
}
