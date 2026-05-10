//! Key-switching primitives `KS.Setup` and `KS.Switch`, plus helpers to
//! compute automorphic images `τ_g^{k-1}(K_g)` of a base matrix locally
//! (without extra key material). See SPEC.md §6 (Stage 3) and §9.b
//! (the structural reason InspiRING needs only two base KS matrices vs.
//! CDKS's `lg d`).
//!
//! The implementation patterns `KS.Switch` on the inline KS body of
//! `spiral_rs::server::coefficient_expansion` (lines 80–103 of
//! `spiral-rs/src/server.rs` at the pinned revision); we cannot call
//! `coefficient_expansion` directly because it is fused with
//! Spiral-PIR's expansion loop. See `docs/spiral-rs-mapping.md` §3.

use rand_chacha::ChaCha20Rng;
use spiral_rs::discrete_gaussian::DiscreteGaussian;
use spiral_rs::gadget::{build_gadget, gadget_invert_alloc};
use spiral_rs::poly::{
    add_into, from_ntt_alloc, multiply, scalar_multiply_alloc, stack_ntt, to_ntt_alloc, PolyMatrix,
    PolyMatrixNTT, PolyMatrixRaw,
};

use crate::automorph::tau_ntt;
use crate::params::RlweParams;

/// A single key-switching matrix `K`. Internally a `[2, ℓ]` `PolyMatrixNTT`
/// (the row-2-by-cols-ℓ shape used by spiral-rs's gadget machinery).
///
/// `K = KS.Setup(s', s)` lets one transform a ciphertext under `s'`
/// (one of `τ_g(s̃)`, `τ_h(s̃)`, …) into one under `s = s̃`. SPEC.md §6.
///
/// `params` is bundled directly so [`ks_switch`] (and the cascade in
/// [`crate::collapse`]) does not need to be threaded a second `&RlweParams`
/// reference at every call site, *and* so a key matrix can never be paired
/// with mismatched gadget settings: callers literally cannot construct a
/// well-typed `(K, params)` pair where `K` was built under a different
/// gadget. Both `params.spiral` (the inner allocator the matrix borrows
/// from) and `params.gadget.ell` (the gadget width the matrix was built
/// for) come from the same `params` reference here, by construction.
///
/// Note: `Debug` / `Clone` are not derived because [`PolyMatrixNTT`] does
/// not implement them upstream; Phase 7 adds hand-written impls if needed.
pub struct KeySwitchingMatrix<'a> {
    /// The encrypted gadget-scaled secret. Shape `[2, ℓ]`.
    pub mat: PolyMatrixNTT<'a>,
    /// The RLWE parameter set this matrix was built against. Tied to the
    /// same `'a` lifetime as the inner spiral-rs allocator referenced by
    /// `mat`, so the borrow checker enforces consistency for free.
    pub params: &'a RlweParams,
}

/// `KS.Setup(s_from → s_to)` — encrypt the gadget-scaled `s_from` under
/// `s_to`, producing a key-switching matrix, per SPEC.md §6 / paper §2.
///
/// Concretely the returned matrix is a stacked `[2 × ℓ]` `PolyMatrixNTT`
///
/// ```text
/// K = [   −a    ]
///     [ s · a + e + s_from · g_z ]
/// ```
///
/// where `a ← R_q^ℓ` is uniformly random, `e ← χ^ℓ` is a discrete-Gaussian
/// noise vector of width `σ_χ · √(2π)` (the paper's centred convention), and
/// `g_z = [1, z, z^2, …, z^{ℓ-1}]` is the spiral-rs gadget vector. With this
/// matrix, [`ks_switch`] takes a ciphertext under `s_from` to one under
/// `s_to` (= `s` here) at a noise budget controlled by `σ_χ` and `ℓ`
/// (Theorem 2 in the paper, SPEC.md §7).
///
/// The function is **offline-only**: it samples fresh randomness from `rng`
/// and is never called on the online `pack` path. Callers should derive
/// `s_from_ntt` and `s_to_ntt` from the same `params.spiral` allocator that
/// is used everywhere else, so memory layouts match.
///
pub fn ks_setup<'a>(
    params: &'a RlweParams,
    s_from_ntt: &PolyMatrixNTT<'a>,
    s_to_ntt: &PolyMatrixNTT<'a>,
    rng: &mut ChaCha20Rng,
) -> KeySwitchingMatrix<'a> {
    assert_eq!(s_from_ntt.rows, 1);
    assert_eq!(s_from_ntt.cols, 1);
    assert_eq!(s_to_ntt.rows, 1);
    assert_eq!(s_to_ntt.cols, 1);

    let spiral = &params.spiral;
    let ell = params.gadget.ell;

    let gadget = build_gadget(spiral, 1, ell);
    let scaled = scalar_multiply_alloc(s_from_ntt, &to_ntt_alloc(&gadget));

    let dg = DiscreteGaussian::init(params.sigma_chi * std::f64::consts::TAU.sqrt());
    let a = PolyMatrixRaw::random_rng(spiral, 1, ell, rng);
    let e = PolyMatrixRaw::noise(spiral, 1, ell, &dg, rng);
    let a_ntt = to_ntt_alloc(&a);
    let w = (-&a).ntt();
    let mut y = PolyMatrixNTT::zero(spiral, 1, ell);
    multiply(&mut y, s_to_ntt, &a_ntt);
    add_into(&mut y, &to_ntt_alloc(&e));
    add_into(&mut y, &scaled);

    KeySwitchingMatrix {
        mat: stack_ntt(&w, &y),
        params,
    }
}

/// `KS.Switch(K, (c1, c2)) → (c1', c2')` — apply a key-switching matrix
/// to an RLWE pair, returning a new pair under `s_to`. SPEC.md §6.
///
/// The gadget shape comes from `k.params` (the `RlweParams` that `K` was
/// built against — see [`KeySwitchingMatrix`] — which makes it impossible
/// to call this with a `K` and an unrelated `params`). The function asserts
/// `K.mat` has the matching `[2 × ℓ]` layout. The body mirrors the inline
/// KS sequence in `spiral-rs/src/server.rs` lines 80–103 (which is fused
/// into Spiral-PIR's coefficient-expansion loop and therefore not reusable
/// directly):
///
/// 1. Round-trip `c1` to coefficient form and gadget-decompose it into `ℓ`
///    base-`z` digit polynomials. The choice of width here MUST match
///    `RlweParams::gadget.ell` so the digit decomposition is the inverse of
///    the `g_z` factor encoded into `K.mat` by [`ks_setup`]. We pass `ℓ`
///    explicitly via `k.params.gadget.ell` instead of reading
///    `K.mat.cols` so a malformed key matrix is caught by the assertion
///    below rather than silently miscomputing.
/// 2. NTT-forward the digits and multiply by `K.mat`. The result is a
///    `[2 × 1]` `PolyMatrixNTT` whose top half is the new `c1'` and whose
///    bottom half is `K.bottom · digits = s_to · c1' + e + s_from · c1`,
///    i.e. `c1' = -K.top · digits` and `c2' (before adding original c2) =
///    s_from · c1 + (small noise)`.
/// 3. Add the original `c2` into the bottom half. The output decrypts under
///    `s_to` to the same plaintext as the input did under `s_from`.
///
/// **Test-only instrumentation**: in `cfg(test)` builds a thread-local
/// counter is incremented on every call. `tests/inspiring_vs_cdks_recursion.rs`
/// asserts the counter equals exactly `d − 1` per call to
/// [`crate::pack::pack`]. Tampering with this is a production-blocker.
///
pub fn ks_switch<'a>(
    k: &KeySwitchingMatrix<'a>,
    c1: &PolyMatrixNTT<'a>,
    c2: &PolyMatrixNTT<'a>,
) -> (PolyMatrixNTT<'a>, PolyMatrixNTT<'a>) {
    ks_call_count::inc();

    let params = k.params;
    assert_eq!(k.mat.rows, 2, "KS matrix must have 2 rows ([w; y])");
    assert_eq!(
        k.mat.cols, params.gadget.ell,
        "KS matrix width must match the gadget length ℓ",
    );
    assert_eq!(c1.rows, 1);
    assert_eq!(c1.cols, 1);
    assert_eq!(c2.rows, 1);
    assert_eq!(c2.cols, 1);

    // The gadget width passed here MUST match the `build_gadget(_, 1, ℓ)`
    // call in `ks_setup`. Anything else makes the digit decomposition
    // non-inverse to the `g_z` factor encoded in `K.mat`, breaking the KS
    // identity. `params.gadget.ell` is the validated source-of-truth (see
    // `RlweParams::new`), which is why `KeySwitchingMatrix` carries its own
    // `params` reference rather than letting us infer the width from
    // `K.mat.cols`.
    let digits_raw = gadget_invert_alloc(params.gadget.ell, &from_ntt_alloc(c1));
    let digits_ntt = to_ntt_alloc(&digits_raw);
    let mut switched = PolyMatrixNTT::zero(c1.params, 2, 1);
    multiply(&mut switched, &k.mat, &digits_ntt);

    let delta_a = switched.submatrix(0, 0, 1, 1);
    let mut delta_b = switched.submatrix(1, 0, 1, 1);
    add_into(&mut delta_b, c2);
    (delta_a, delta_b)
}

/// Compute `τ_g^{k-1}(K_g)` from `K_g` without any extra key material.
/// The image is just `K_g` with `τ_g^{k-1}` applied component-wise to
/// each polynomial of the matrix. SPEC.md §6 / Appendix C.
///
/// The `params` reference is forwarded from the input — local images of a
/// KS matrix share its parameter set by definition.
#[must_use]
pub fn automorphic_image<'a>(k: &KeySwitchingMatrix<'a>, t: u64) -> KeySwitchingMatrix<'a> {
    KeySwitchingMatrix {
        mat: tau_ntt(&k.mat, t),
        params: k.params,
    }
}

/// Test/diagnostic thread-local counter for `KS.Switch` calls. Used by
/// `tests/inspiring_vs_cdks_recursion.rs` to assert the linear-cascade
/// `KS.Switch` count of exactly `d − 1` per pack — the runtime structural
/// guard against accidental CDKS-style implementation drift (SPEC.md §9.h).
#[doc(hidden)]
pub mod ks_call_count {
    use std::cell::Cell;

    thread_local! {
        static COUNTER: Cell<u64> = const { Cell::new(0) };
    }

    /// Reset to 0. Call before a measured `pack`.
    pub fn reset() {
        COUNTER.with(|c| c.set(0));
    }

    /// Increment by one. Called from inside `ks_switch`.
    pub fn inc() {
        COUNTER.with(|c| c.set(c.get() + 1));
    }

    /// Read the current count.
    #[must_use]
    pub fn get() -> u64 {
        COUNTER.with(Cell::get)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automorph::tau_g_pow;
    use crate::params::GadgetParams;
    use rand::SeedableRng;
    use spiral_rs::poly::PolyMatrix;

    // ---- helpers --------------------------------------------------------

    fn params() -> RlweParams {
        // Small parameters: d=8, q=12289 (NTT-friendly 14-bit prime), p=4,
        // gadget z=8, ℓ=5 so z^ℓ = 32768 ≥ q. Noise width is intentionally
        // tiny (σ=0.1) so that round-trip tests below decrypt exactly even
        // without rounding-margin reasoning.
        RlweParams::new(
            8,
            12289,
            4,
            0.1,
            GadgetParams {
                bits_per: 3,
                ell: 5,
            },
        )
        .expect("valid params")
    }

    fn raw_from_coeffs<'a>(params: &'a RlweParams, coeffs: &[u64]) -> PolyMatrixRaw<'a> {
        let mut raw = PolyMatrixRaw::zero(&params.spiral, 1, 1);
        raw.get_poly_mut(0, 0).copy_from_slice(coeffs);
        raw
    }

    fn ntt_from_coeffs<'a>(params: &'a RlweParams, coeffs: &[u64]) -> PolyMatrixNTT<'a> {
        to_ntt_alloc(&raw_from_coeffs(params, coeffs))
    }

    /// Apply `τ_t : p(X) ↦ p(X^t)` to a length-`d` coefficient vector at the
    /// quotient ring `R_q = Z_q[X]/(X^d + 1)`.
    fn tau_coeffs(poly: &[u64], exponent: u64, q: u64) -> Vec<u64> {
        let d = poly.len();
        let mut out = vec![0; d];
        for (i, coeff) in poly.iter().enumerate() {
            let exp = (i as u64 * exponent) % (2 * d as u64);
            let reduced = coeff % q;
            let (idx, value) = if exp < d as u64 {
                (exp as usize, reduced)
            } else {
                (
                    (exp - d as u64) as usize,
                    if reduced == 0 { 0 } else { q - reduced },
                )
            };
            out[idx] = (out[idx] + value) % q;
        }
        out
    }

    /// Negacyclic polynomial multiplication in `R_q`, computed in `u128` so
    /// the test oracle is independent of spiral-rs's NTT path.
    fn negacyclic_mul(lhs: &[u64], rhs: &[u64], q: u64) -> Vec<u64> {
        let d = lhs.len();
        let mut out = vec![0; d];
        for (i, l) in lhs.iter().enumerate() {
            for (j, r) in rhs.iter().enumerate() {
                let product = (u128::from(*l) * u128::from(*r) % u128::from(q)) as u64;
                let degree = i + j;
                if degree < d {
                    out[degree] = (out[degree] + product) % q;
                } else if product != 0 {
                    out[degree - d] = (out[degree - d] + q - product) % q;
                }
            }
        }
        out
    }

    fn add_poly(lhs: &[u64], rhs: &[u64], q: u64) -> Vec<u64> {
        lhs.iter().zip(rhs).map(|(x, y)| (x + y) % q).collect()
    }

    fn sub_poly(lhs: &[u64], rhs: &[u64], q: u64) -> Vec<u64> {
        lhs.iter().zip(rhs).map(|(x, y)| (q + x - y) % q).collect()
    }

    /// Decrypt `(c1, c2)` under `s` and round to the plaintext slot of `Δ·m`.
    fn decrypt(params: &RlweParams, c1: &[u64], c2: &[u64], s: &[u64]) -> Vec<u64> {
        let inner = add_poly(c2, &negacyclic_mul(c1, s, params.q), params.q);
        inner
            .iter()
            .map(|coeff| ((coeff + params.delta / 2) / params.delta) % params.p)
            .collect()
    }

    // ---- regression test for the upstream `spiral-rs` scalar bug --------

    /// **Regression guard for the spiral-rs `multiply_add_modular` bug**.
    ///
    /// The crate's AVX-512 gate (see `src/lib.rs`) exists *because* the
    /// scalar `multiply` fallback in our pinned `spiral-rs` revision drops
    /// the accumulator on `crt_count == 1`, collapsing `[1 × ℓ] · [ℓ × 1]`
    /// products to "last term only". This test multiplies a `[1 × 3]` by a
    /// `[3 × 1]` matrix where every inner term is non-zero, with a known
    /// reference computed in `u128` outside spiral-rs. If anyone ever
    /// removes the AVX-512 gate or drops back to the scalar fallback path,
    /// this test fails with an obvious "expected SUM, got LAST" mismatch.
    #[test]
    fn spiral_matrix_multiply_accumulates_along_inner_dim() {
        let params = params();
        let mut a = PolyMatrixNTT::zero(&params.spiral, 1, 3);
        let mut b = PolyMatrixNTT::zero(&params.spiral, 3, 1);
        let inputs: [[u64; 8]; 3] = [
            [3, 1, 4, 1, 5, 9, 2, 6],
            [2, 7, 1, 8, 2, 8, 1, 8],
            [1, 6, 1, 8, 0, 3, 3, 9],
        ];
        let factors: [[u64; 8]; 3] = [
            [11, 13, 17, 19, 23, 29, 31, 37],
            [41, 43, 47, 53, 59, 61, 67, 71],
            [73, 79, 83, 89, 97, 101, 103, 107],
        ];

        for k in 0..3 {
            let a_ntt = to_ntt_alloc(&raw_from_coeffs(&params, &inputs[k]));
            a.get_poly_mut(0, k).copy_from_slice(a_ntt.get_poly(0, 0));
            let b_ntt = to_ntt_alloc(&raw_from_coeffs(&params, &factors[k]));
            b.get_poly_mut(k, 0).copy_from_slice(b_ntt.get_poly(0, 0));
        }

        let mut prod = PolyMatrixNTT::zero(&params.spiral, 1, 1);
        multiply(&mut prod, &a, &b);
        let prod_raw = from_ntt_alloc(&prod);

        let mut expected = vec![0_u64; params.d];
        for k in 0..3 {
            expected = add_poly(
                &expected,
                &negacyclic_mul(&inputs[k], &factors[k], params.q),
                params.q,
            );
        }

        assert_eq!(
            prod_raw.get_poly(0, 0).to_vec(),
            expected,
            "spiral-rs multiply lost the accumulator — AVX-512 gate has been bypassed; \
             see src/lib.rs and docs/spiral-rs-mapping.md §1"
        );
    }

    // ---- gadget sanity --------------------------------------------------

    #[test]
    fn spiral_gadget_invert_reconstructs_each_coefficient_mod_q() {
        let params = params();
        let mut input = PolyMatrixRaw::zero(&params.spiral, 1, 1);
        input
            .get_poly_mut(0, 0)
            .copy_from_slice(&[0, 1, 7, 8, 63, 64, params.q - 1, params.q - 4]);

        let digits = gadget_invert_alloc(params.gadget.ell, &input);
        let z = u128::from(params.gadget.z());
        let q = u128::from(params.q);
        for coeff_idx in 0..params.d {
            let mut acc = 0u128;
            for digit_row in 0..params.gadget.ell {
                acc +=
                    u128::from(digits.get_poly(digit_row, 0)[coeff_idx]) * z.pow(digit_row as u32);
            }
            assert_eq!(
                input.get_poly(0, 0)[coeff_idx],
                (acc % q) as u64,
                "coefficient index {coeff_idx}"
            );
        }
    }

    // ---- KS round-trip --------------------------------------------------

    /// Encrypt a known plaintext under `s_from`, apply `ks_switch` with a
    /// real `ks_setup` matrix (real RNG, real noise), decrypt under
    /// `s_to`, expect the plaintext back. Exercises the production code
    /// path end-to-end at small parameters where the noise budget is
    /// comfortably below `Δ/2`.
    #[test]
    fn ks_setup_then_ks_switch_recovers_plaintext_under_target_secret() {
        let params = params();
        let s_from = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let s_to = vec![1, 0, params.q - 1, 1, 0, 1, params.q - 1, 0];
        let messages = vec![0_u64, 1, 2, 3, 3, 2, 1, 0];
        let c1 = vec![5_u64, 7, 11, 13, 17, 19, 23, 29];

        // c2 = Δ·m − c1·s_from (mod q), so (c1, c2) decrypts to `messages`
        // under s_from with zero noise.
        let encoded: Vec<_> = messages.iter().map(|m| (params.delta * m) % params.q).collect();
        let c2 = sub_poly(&encoded, &negacyclic_mul(&c1, &s_from, params.q), params.q);

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xC0DE_C0DE);
        let k = ks_setup(
            &params,
            &ntt_from_coeffs(&params, &s_from),
            &ntt_from_coeffs(&params, &s_to),
            &mut rng,
        );

        let (c1_new, c2_new) = ks_switch(
            &k,
            &ntt_from_coeffs(&params, &c1),
            &ntt_from_coeffs(&params, &c2),
        );

        let c1_new_raw = from_ntt_alloc(&c1_new);
        let c2_new_raw = from_ntt_alloc(&c2_new);
        assert_eq!(
            decrypt(&params, c1_new_raw.get_poly(0, 0), c2_new_raw.get_poly(0, 0), &s_to),
            messages,
        );
    }

    // ---- automorphic image ----------------------------------------------

    /// Local automorphic images of `K = KS.Setup(s_from → s_to)` are
    /// themselves valid KS matrices — for `K' = automorphic_image(K, t)`,
    /// switching a ciphertext under `τ_t(s_from)` through `K'` produces a
    /// ciphertext that decrypts under `τ_t(s_to)`. SPEC.md §6 / paper
    /// Appendix C is the formal statement; this test pins it down at small
    /// parameters with a non-identity rotation.
    #[test]
    fn automorphic_image_yields_ks_for_rotated_secret_pair() {
        let params = params();
        let s_from = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let s_to = vec![1, 0, params.q - 1, 1, 0, 1, params.q - 1, 0];
        let messages = vec![0_u64, 1, 2, 3, 3, 2, 1, 0];
        let c1 = vec![5_u64, 7, 11, 13, 17, 19, 23, 29];

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xA70_A0E);
        let k = ks_setup(
            &params,
            &ntt_from_coeffs(&params, &s_from),
            &ntt_from_coeffs(&params, &s_to),
            &mut rng,
        );

        // Pick a non-trivial rotation `t = τ_g^2`.
        let t = tau_g_pow(2, params.d);
        let s_from_rot = tau_coeffs(&s_from, t, params.q);
        let s_to_rot = tau_coeffs(&s_to, t, params.q);
        let k_image = automorphic_image(&k, t);

        // Encrypt `messages` under the rotated source secret with c1.
        let encoded: Vec<_> = messages.iter().map(|m| (params.delta * m) % params.q).collect();
        let c2 = sub_poly(&encoded, &negacyclic_mul(&c1, &s_from_rot, params.q), params.q);

        let (c1_new, c2_new) = ks_switch(
            &k_image,
            &ntt_from_coeffs(&params, &c1),
            &ntt_from_coeffs(&params, &c2),
        );
        let c1_new_raw = from_ntt_alloc(&c1_new);
        let c2_new_raw = from_ntt_alloc(&c2_new);
        assert_eq!(
            decrypt(&params, c1_new_raw.get_poly(0, 0), c2_new_raw.get_poly(0, 0), &s_to_rot),
            messages,
            "ks_switch through automorphic_image(K, t) must decrypt under τ_t(s_to)",
        );
    }

    // ---- ks_call_count instrumentation ----------------------------------

    /// Sanity check on the test-only call counter that
    /// `tests/inspiring_vs_cdks_recursion.rs` relies on. If this counter
    /// stops working, the linear-cascade structural invariant
    /// (`#KS.Switch == d − 1` per pack) becomes unobservable.
    #[test]
    fn ks_call_count_increments_once_per_ks_switch() {
        let params = params();
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0xCA11);
        let k = ks_setup(
            &params,
            &ntt_from_coeffs(&params, &[1, 2, 3, 4, 5, 6, 7, 8]),
            &ntt_from_coeffs(&params, &[8, 7, 6, 5, 4, 3, 2, 1]),
            &mut rng,
        );
        let c1 = ntt_from_coeffs(&params, &[1, 0, 0, 0, 0, 0, 0, 0]);
        let c2 = PolyMatrixNTT::zero(&params.spiral, 1, 1);

        ks_call_count::reset();
        let _ = ks_switch(&k, &c1, &c2);
        let _ = ks_switch(&k, &c1, &c2);
        let _ = ks_switch(&k, &c1, &c2);
        assert_eq!(ks_call_count::get(), 3);
    }
}
