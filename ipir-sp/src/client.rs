//! Client-side key material for the IPIR-SP packing layer.
//!
//! YPIR's CDKS path uploads `log d` expansion matrices. The InspiRING path
//! instead uploads two key-switching matrices per preprocessing block:
//! `K_g = KS.Setup(τ_g(s) -> s)` and `K_h = KS.Setup(τ_h(s) -> s)`.

use inspiring::automorph::{h, tau_g_pow, tau_ntt};
use inspiring::key_switching::{ks_setup, KeySwitchingMatrix};
use inspiring::RlweParams;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use spiral_rs::poly::{to_ntt_alloc, PolyMatrix, PolyMatrixRaw};

/// A client secret in coefficient form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientSecret {
    /// Secret coefficients modulo `q`.
    pub coeffs: Vec<u64>,
}

impl ClientSecret {
    /// Build a secret from coefficients, reducing each coefficient modulo `q`.
    #[must_use]
    pub fn from_coeffs(params: &RlweParams, coeffs: impl Into<Vec<u64>>) -> Self {
        let coeffs = coeffs.into();
        assert_eq!(
            coeffs.len(),
            params.d,
            "client secret must have d coefficients"
        );

        Self {
            coeffs: coeffs.into_iter().map(|coeff| coeff % params.q).collect(),
        }
    }

    /// Sample a ternary secret with coefficients in `{0, 1, -1 mod q}`.
    pub fn sample_ternary(params: &RlweParams, rng: &mut ChaCha20Rng) -> Self {
        let coeffs = (0..params.d)
            .map(|_| match rng.gen_range(0..3) {
                0 => 0,
                1 => 1,
                _ => params.q - 1,
            })
            .collect();

        Self { coeffs }
    }

    /// Convert the secret to a `[1, 1]` NTT polynomial matrix.
    #[must_use]
    pub fn to_ntt<'a>(&self, params: &'a RlweParams) -> spiral_rs::poly::PolyMatrixNTT<'a> {
        assert_eq!(
            self.coeffs.len(),
            params.d,
            "client secret must have d coefficients"
        );

        let mut raw = PolyMatrixRaw::zero(&params.spiral, 1, 1);
        raw.get_poly_mut(0, 0).copy_from_slice(&self.coeffs);
        to_ntt_alloc(&raw)
    }
}

/// Generate one `(K_g, K_h)` pair from a base secret.
pub fn generate_ks_pair<'a>(
    params: &'a RlweParams,
    secret: &ClientSecret,
    rng: &mut ChaCha20Rng,
) -> (KeySwitchingMatrix<'a>, KeySwitchingMatrix<'a>) {
    let s = secret.to_ntt(params);
    let tau_g_s = tau_ntt(&s, tau_g_pow(1, params.d));
    let tau_h_s = tau_ntt(&s, h(params.d));

    let kg = ks_setup(params, &tau_g_s, &s, rng);
    let kh = ks_setup(params, &tau_h_s, &s, rng);

    (kg, kh)
}

/// Generate `count` owned `(K_g, K_h)` pairs for preprocessing blocks.
///
/// `PackPreprocessed` owns its keys, so callers that build many blocks need
/// many owned pairs. They all encode the same automorphic source/target secret
/// relation, but use fresh setup randomness from `rng`.
pub fn generate_ks_pairs<'a>(
    params: &'a RlweParams,
    secret: &ClientSecret,
    count: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<(KeySwitchingMatrix<'a>, KeySwitchingMatrix<'a>)> {
    (0..count)
        .map(|_| generate_ks_pair(params, secret, rng))
        .collect()
}

#[cfg(test)]
mod tests {
    use inspiring::{GadgetParams, RlweParams};
    use rand_chacha::rand_core::SeedableRng;

    use super::*;

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

    #[test]
    fn client_secret_reduces_coefficients_mod_q() {
        let params = params();
        let secret =
            ClientSecret::from_coeffs(&params, vec![0, 1, params.q, params.q + 2, 5, 6, 7, 8]);

        assert_eq!(secret.coeffs, vec![0, 1, 0, 2, 5, 6, 7, 8]);
    }

    #[test]
    fn sampled_ternary_secret_uses_mod_q_minus_one_for_negative_one() {
        let params = params();
        let mut rng = ChaCha20Rng::seed_from_u64(0x5350);

        let secret = ClientSecret::sample_ternary(&params, &mut rng);

        assert_eq!(secret.coeffs.len(), params.d);
        assert!(secret
            .coeffs
            .iter()
            .all(|coeff| matches!(*coeff, 0 | 1) || *coeff == params.q - 1));
    }

    #[test]
    fn generate_ks_pair_returns_two_expected_matrix_shapes() {
        let params = params();
        let secret = ClientSecret::from_coeffs(&params, vec![1, 0, params.q - 1, 1, 0, 1, 0, 0]);
        let mut rng = ChaCha20Rng::seed_from_u64(0xBEEF);

        let (kg, kh) = generate_ks_pair(&params, &secret, &mut rng);

        assert_eq!(kg.mat.rows, 2);
        assert_eq!(kg.mat.cols, params.gadget.ell);
        assert_eq!(kh.mat.rows, 2);
        assert_eq!(kh.mat.cols, params.gadget.ell);
        assert_eq!(kg.params.q, params.q);
        assert_eq!(kh.params.q, params.q);
    }

    #[test]
    fn generate_ks_pair_is_deterministic_under_fixed_seed() {
        let params = params();
        let secret = ClientSecret::from_coeffs(&params, vec![1, 0, params.q - 1, 1, 0, 1, 0, 0]);
        let mut left_rng = ChaCha20Rng::seed_from_u64(0xC0DE);
        let mut right_rng = ChaCha20Rng::seed_from_u64(0xC0DE);

        let left = generate_ks_pair(&params, &secret, &mut left_rng);
        let right = generate_ks_pair(&params, &secret, &mut right_rng);

        assert_eq!(left.0.mat.as_slice(), right.0.mat.as_slice());
        assert_eq!(left.1.mat.as_slice(), right.1.mat.as_slice());
    }

    #[test]
    fn generate_ks_pairs_returns_owned_pair_per_block() {
        let params = params();
        let secret = ClientSecret::from_coeffs(&params, vec![1, 0, params.q - 1, 1, 0, 1, 0, 0]);
        let mut rng = ChaCha20Rng::seed_from_u64(0xFACE);

        let pairs = generate_ks_pairs(&params, &secret, 3, &mut rng);

        assert_eq!(pairs.len(), 3);
        assert_eq!(pairs[2].0.mat.rows, 2);
        assert_eq!(pairs[2].1.mat.cols, params.gadget.ell);
    }
}
