//! Stage 3 of `InspiRING.Pack`: the collapse from `IRCtx` (a wide
//! `(d+1)`-element ciphertext) down to a 2-element RLWE ciphertext under
//! the base secret `s̃`.
//!
//! Three layered subroutines, exactly as in Algorithm 1 / Appendix C
//! (see SPEC.md §6):
//!
//! - [`collapse_one`] — one key-switch step.
//! - [`collapse_half`] — `d/2 - 1` `collapse_one` calls applied to one
//!   half of `a_agg` using automorphic images of `K_g`, optionally
//!   pre-composed with `τ_h`.
//! - [`collapse`] — runs `collapse_half` twice, then a final `KS.Switch`
//!   with `K_h` to fold the `τ_h(s̃)` share into `s̃`.
//!
//! **Linear-cascade invariant** (SPEC.md §6 + §9):
//!
//! `# KS.Switch calls per pack = (d/2 - 1) + (d/2 - 1) + 1 = d - 1`.
//!
//! A CDKS-style implementation would have `(d - 1) · log₂ d` calls.
//! `tests/inspiring_vs_cdks_recursion.rs` asserts this empirically by
//! reading `key_switching::ks_call_count::get()` after each pack.
//!
//! Stage 3 is implemented.

use spiral_rs::poly::{add_into, stack_ntt, to_ntt_alloc, PolyMatrixNTT};

use crate::intermediate::IRCtx;
use crate::key_switching::{ks_switch, KeySwitchingMatrix};
use crate::pack::RlweCiphertext;
use crate::params::RlweParams;

/// `CollapseOne` — one cascade step. Takes the running collapse state
/// (a `(2 + remaining)`-element pseudo-ciphertext) plus the appropriate
/// automorphic image of the base KS matrix, and produces a state that
/// has one fewer element. SPEC.md §6 / paper Appendix C.
///
/// The `RlweParams` for the underlying `ks_switch` call is read from
/// `k_image.params` — see [`KeySwitchingMatrix`] for why parameters are
/// bundled onto the matrix rather than threaded as a separate argument.
///
pub fn collapse_one<'a>(state: &mut CollapseState<'a>, k_image: &KeySwitchingMatrix<'a>) {
    let k = state.a.len();
    assert!(
        k >= 2,
        "collapse::collapse_one requires at least two a components"
    );

    let (delta_a, delta_b) = ks_switch(k_image, &state.a[k - 1], &state.b);
    add_into(&mut state.a[k - 2], &delta_a);
    state.a.pop();
    state.b = delta_b;
}

/// `CollapseHalf` — runs `d/2 - 1` `collapse_one` calls over one half
/// (either the `τ_g^j` half or the `τ_h ∘ τ_g^j` half) of `a_agg`.
///
/// SPEC.md §6.
///
pub fn collapse_half<'a>(state: &mut CollapseState<'a>, kg_images: &[KeySwitchingMatrix<'a>]) {
    assert_eq!(
        kg_images.len(),
        state.a.len().saturating_sub(1),
        "collapse::collapse_half expects one K_g image per collapse step"
    );

    while state.a.len() > 1 {
        let image_idx = state.a.len() - 2;
        collapse_one(state, &kg_images[image_idx]);
    }
}

/// `Collapse` — full Stage 3. Runs `collapse_half` twice, then a final
/// `KS.Switch` with `K_h`. Output is an RLWE ciphertext under `s̃`.
///
/// SPEC.md §6.
///
pub fn collapse<'a>(
    params: &'a RlweParams,
    agg: IRCtx<'a>,
    kg_images_left: &[KeySwitchingMatrix<'a>],
    kg_images_right: &[KeySwitchingMatrix<'a>],
    kh: &KeySwitchingMatrix<'a>,
) -> RlweCiphertext<'a> {
    assert_eq!(
        agg.a_hat.len(),
        params.d,
        "collapse::collapse expects d a_hat slots"
    );
    assert_eq!(
        kg_images_left.len(),
        params.d / 2 - 1,
        "collapse::collapse expects d/2 - 1 left K_g images"
    );
    assert_eq!(
        kg_images_right.len(),
        params.d / 2 - 1,
        "collapse::collapse expects d/2 - 1 right K_g images"
    );

    let mut slots = agg.a_hat;
    let right = slots.split_off(params.d / 2);
    let left = slots;
    let b = to_ntt_alloc(&agg.b_tilde);

    let mut left_state = CollapseState { a: left, b };
    collapse_half(&mut left_state, kg_images_left);
    let left_a = left_state
        .a
        .pop()
        .expect("collapse_half leaves one left component");

    let mut right_state = CollapseState {
        a: right,
        b: left_state.b,
    };
    collapse_half(&mut right_state, kg_images_right);
    let right_a = right_state
        .a
        .pop()
        .expect("collapse_half leaves one right component");

    let mut final_state = CollapseState {
        a: vec![left_a, right_a],
        b: right_state.b,
    };
    collapse_one(&mut final_state, kh);

    RlweCiphertext {
        inner: stack_ntt(&final_state.a[0], &final_state.b),
    }
}

/// Running state of the collapse cascade. At each step it carries
/// `(c1, c2, …)` where the head two slots are the proto-RLWE pair
/// being assembled and the tail slots are the as-yet-untouched part
/// of `a_agg`. SPEC.md §6 / Appendix C.
///
pub struct CollapseState<'a> {
    /// Wider random components still present in the running state.
    pub a: Vec<PolyMatrixNTT<'a>>,
    /// Running `b` polynomial in NTT form.
    pub b: PolyMatrixNTT<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_switching::ks_call_count;
    use crate::params::GadgetParams;
    use spiral_rs::poly::{from_ntt_alloc, PolyMatrix, PolyMatrixRaw};

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

    fn raw_from_coeffs<'a>(params: &'a RlweParams, coeffs: &[u64]) -> PolyMatrixRaw<'a> {
        let mut raw = PolyMatrixRaw::zero(&params.spiral, 1, 1);
        raw.get_poly_mut(0, 0).copy_from_slice(coeffs);
        raw
    }

    fn coeffs(poly: &PolyMatrixNTT<'_>) -> Vec<u64> {
        from_ntt_alloc(poly).get_poly(0, 0).to_vec()
    }

    #[test]
    fn collapse_one_drops_one_component_and_increments_switch_count() {
        let params = params();
        let k = zero_ks(&params);
        let mut state = CollapseState {
            a: vec![
                to_ntt_alloc(&raw_from_coeffs(&params, &[1, 2, 3, 4, 5, 6, 7, 8])),
                to_ntt_alloc(&raw_from_coeffs(&params, &[8, 7, 6, 5, 4, 3, 2, 1])),
            ],
            b: to_ntt_alloc(&raw_from_coeffs(&params, &[9, 0, 0, 0, 0, 0, 0, 0])),
        };

        ks_call_count::reset();
        collapse_one(&mut state, &k);

        assert_eq!(state.a.len(), 1);
        assert_eq!(ks_call_count::get(), 1);
        assert_eq!(coeffs(&state.b), vec![9, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn collapse_half_runs_one_switch_per_supplied_image() {
        let params = params();
        let images: Vec<_> = (0..params.d / 2 - 1).map(|_| zero_ks(&params)).collect();
        let mut state = CollapseState {
            a: (0..params.d / 2)
                .map(|_| PolyMatrixNTT::zero(&params.spiral, 1, 1))
                .collect(),
            b: PolyMatrixNTT::zero(&params.spiral, 1, 1),
        };

        ks_call_count::reset();
        collapse_half(&mut state, &images);

        assert_eq!(state.a.len(), 1);
        assert_eq!(ks_call_count::get(), (params.d / 2 - 1) as u64);
    }

    #[test]
    fn collapse_runs_exactly_d_minus_one_switches() {
        let params = params();
        let left_images: Vec<_> = (0..params.d / 2 - 1).map(|_| zero_ks(&params)).collect();
        let right_images: Vec<_> = (0..params.d / 2 - 1).map(|_| zero_ks(&params)).collect();
        let kh = zero_ks(&params);
        let agg = IRCtx {
            a_hat: (0..params.d)
                .map(|_| PolyMatrixNTT::zero(&params.spiral, 1, 1))
                .collect(),
            b_tilde: raw_from_coeffs(&params, &[1, 2, 3, 4, 5, 6, 7, 8]),
        };

        ks_call_count::reset();
        let ct = collapse(&params, agg, &left_images, &right_images, &kh);

        assert_eq!(ct.inner.rows, 2);
        assert_eq!(ct.inner.cols, 1);
        assert_eq!(ks_call_count::get(), (params.d - 1) as u64);
    }
}
