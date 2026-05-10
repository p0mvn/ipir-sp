//! `PackPreprocessed`: the CRS-model offline cache.
//!
//! See SPEC.md §8 (offline / online split). Every quantity in Algorithm 1
//! that depends only on `(A, K_g, K_h)` (and not on the LWE `b` scalars)
//! is materialised here, in NTT form, so the online [`crate::pack::pack`]
//! call is a pure function of `(b_0, …, b_{d-1}, &PackPreprocessed)`.
//!
//! Phase 8 status: offline cache construction is implemented.

use spiral_rs::poly::{from_ntt_alloc, PolyMatrix, PolyMatrixNTT};

use crate::automorph::{h, tau_g_pow};
use crate::error::InspiringError;
use crate::intermediate::{aggregate, transform};
use crate::key_switching::{automorphic_image, KeySwitchingMatrix};
use crate::lwe::LweCiphertext;
use crate::params::RlweParams;

/// All preprocessable data for a single CRS `A` and a single pair of
/// key-switching matrices `(K_g, K_h)`.
///
/// **API invariant (SPEC.md §10)**: this struct holds **exactly two**
/// key-switching matrices. Any reviewer asked to add a third should
/// instead read SPEC.md §9.h and the test `tests/inspiring_vs_cdks_recursion.rs`.
///
pub struct PackPreprocessed<'a> {
    /// Underlying parameter set.
    pub params: &'a RlweParams,

    /// Per-LWE-slot Stage-1 result: `a_hat[k][j]` is the `j`-th
    /// component of `IRCtx`'s `â` for input slot `k`. `a_hat.len() == d`,
    /// `a_hat[k].len() == d`. SPEC.md §4.
    ///
    /// All NTT-form. CRS-side, fully preprocessable.
    pub a_hat: Vec<Vec<PolyMatrixNTT<'a>>>,

    /// Stage-2 aggregated `â_agg = Σ_k a_hat[k] · X^k`. SPEC.md §5.
    pub a_agg: Vec<PolyMatrixNTT<'a>>,

    /// `K_g`: the base key-switching matrix for the `τ_g`-cycle.
    pub kg: KeySwitchingMatrix<'a>,

    /// `K_h`: the final-step key-switching matrix that folds the
    /// `τ_h(s̃)` share into `s̃`.
    pub kh: KeySwitchingMatrix<'a>,

    /// Cache of `τ_g^i(K_g)` for `i ∈ [0, d/2 - 1)`, plus
    /// `τ_h(τ_g^i(K_g))` for the second half. Computed once per
    /// CRS so the online path never invokes an automorphism on `K_g`.
    /// SPEC.md §6.
    pub kg_images_left: Vec<KeySwitchingMatrix<'a>>,
    /// Same as `kg_images_left` but pre-composed with `τ_h` for the
    /// right-half collapse.
    pub kg_images_right: Vec<KeySwitchingMatrix<'a>>,
}

impl<'a> PackPreprocessed<'a> {
    /// Build all CRS-side data from `(A, K_g, K_h)`. Online callers then
    /// call [`crate::pack::pack`] with just the `b_k` scalars.
    ///
    /// API invariant: this signature accepts exactly two key-switching
    /// matrices. Adding a third is a breaking change and a CDKS-drift
    /// red flag (SPEC.md §9.h).
    ///
    pub fn build(
        params: &'a RlweParams,
        crs: &PolyMatrixNTT<'a>,
        kg: KeySwitchingMatrix<'a>,
        kh: KeySwitchingMatrix<'a>,
    ) -> Result<Self, InspiringError> {
        if crs.rows != params.d || crs.cols != 1 {
            return Err(InspiringError::PreprocessMismatch(format!(
                "expected CRS shape {}x1, got {}x{}",
                params.d, crs.rows, crs.cols
            )));
        }
        if kg.mat.rows != 2 || kg.mat.cols != params.gadget.ell {
            return Err(InspiringError::PreprocessMismatch(format!(
                "K_g must have shape 2x{}, got {}x{}",
                params.gadget.ell, kg.mat.rows, kg.mat.cols
            )));
        }
        if kh.mat.rows != 2 || kh.mat.cols != params.gadget.ell {
            return Err(InspiringError::PreprocessMismatch(format!(
                "K_h must have shape 2x{}, got {}x{}",
                params.gadget.ell, kh.mat.rows, kh.mat.cols
            )));
        }

        let crs_raw = from_ntt_alloc(crs);
        let irctxs: Vec<_> = (0..params.d)
            .map(|k| {
                let a = crs_raw.get_poly(k, 0).to_vec();
                transform(params, &LweCiphertext { a, b: 0 })
            })
            .collect();
        let agg = aggregate(params, &irctxs);
        let a_hat = irctxs.into_iter().map(|ictx| ictx.a_hat).collect();

        let two_d = 2 * params.d as u64;
        let h_d = h(params.d);
        let kg_images_left = (0..(params.d / 2 - 1))
            .map(|i| automorphic_image(&kg, tau_g_pow(i, params.d)))
            .collect();
        let kg_images_right = (0..(params.d / 2 - 1))
            .map(|i| automorphic_image(&kg, (tau_g_pow(i, params.d) * h_d) % two_d))
            .collect();

        Ok(Self {
            params,
            a_hat,
            a_agg: agg.a_hat,
            kg,
            kh,
            kg_images_left,
            kg_images_right,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::GadgetParams;
    use spiral_rs::poly::{to_ntt_alloc, PolyMatrixRaw};

    fn params() -> RlweParams {
        RlweParams::new(
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

    fn zero_ks<'a>(params: &'a RlweParams) -> KeySwitchingMatrix<'a> {
        KeySwitchingMatrix {
            mat: PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell),
            params,
        }
    }

    fn crs<'a>(params: &'a RlweParams) -> PolyMatrixNTT<'a> {
        let mut raw = PolyMatrixRaw::zero(&params.spiral, params.d, 1);
        for row in 0..params.d {
            for col in 0..params.d {
                raw.get_poly_mut(row, 0)[col] = (row * params.d + col + 1) as u64;
            }
        }
        to_ntt_alloc(&raw)
    }

    #[test]
    fn build_precomputes_transform_aggregate_and_key_images() {
        let params = params();
        let crs = crs(&params);

        let pre = PackPreprocessed::build(&params, &crs, zero_ks(&params), zero_ks(&params))
            .expect("valid preprocessing");

        assert_eq!(pre.a_hat.len(), params.d);
        assert_eq!(pre.a_agg.len(), params.d);
        assert_eq!(pre.kg_images_left.len(), params.d / 2 - 1);
        assert_eq!(pre.kg_images_right.len(), params.d / 2 - 1);
    }

    #[test]
    fn build_rejects_wrong_crs_shape() {
        let params = params();
        let wrong = PolyMatrixNTT::zero(&params.spiral, 1, 1);

        assert!(matches!(
            PackPreprocessed::build(&params, &wrong, zero_ks(&params), zero_ks(&params)),
            Err(InspiringError::PreprocessMismatch(_))
        ));
    }
}
