//! Wire serialization helpers for IPIR-SP key material.
//!
//! YPIR's CDKS upload serializes `log d` expansion matrices after condensing
//! two CRT limbs into one `u64`. IPIR-SP uses single-CRT InspiRING matrices,
//! so the stable wire format is simply the little-endian `u64` coefficient
//! stream for `K_g` followed by `K_h`.

use inspiring::key_switching::KeySwitchingMatrix;
use inspiring::{InspiringError, RlweParams};
use spiral_rs::poly::PolyMatrix;

/// Number of bytes used by one serialized `(K_g, K_h)` pair.
#[must_use]
pub fn serialized_ks_pair_len(params: &RlweParams) -> usize {
    2 * key_matrix_u64_len(params) * std::mem::size_of::<u64>()
}

/// Serialize the two InspiRING key-switching matrices uploaded by the client.
pub fn serialize_ks_pair(
    params: &RlweParams,
    kg: &KeySwitchingMatrix<'_>,
    kh: &KeySwitchingMatrix<'_>,
) -> Result<Vec<u8>, InspiringError> {
    validate_ks_matrix(params, kg, "K_g")?;
    validate_ks_matrix(params, kh, "K_h")?;

    let mut out = Vec::with_capacity(serialized_ks_pair_len(params));
    write_u64s_le(&mut out, kg.mat.as_slice());
    write_u64s_le(&mut out, kh.mat.as_slice());
    Ok(out)
}

fn key_matrix_u64_len(params: &RlweParams) -> usize {
    2 * params.gadget.ell * params.d
}

fn validate_ks_matrix(
    params: &RlweParams,
    key: &KeySwitchingMatrix<'_>,
    label: &'static str,
) -> Result<(), InspiringError> {
    if key.params.d != params.d
        || key.params.q != params.q
        || key.params.gadget.bits_per != params.gadget.bits_per
        || key.params.gadget.ell != params.gadget.ell
    {
        return Err(InspiringError::PreprocessMismatch(format!(
            "{label} parameters do not match serialization params"
        )));
    }

    if key.mat.rows != 2 || key.mat.cols != params.gadget.ell {
        return Err(InspiringError::PreprocessMismatch(format!(
            "{label} must have shape 2x{}, got {}x{}",
            params.gadget.ell, key.mat.rows, key.mat.cols
        )));
    }

    let expected_len = key_matrix_u64_len(params);
    if key.mat.as_slice().len() != expected_len {
        return Err(InspiringError::PreprocessMismatch(format!(
            "{label} coefficient length must be {expected_len}, got {}",
            key.mat.as_slice().len()
        )));
    }

    Ok(())
}

fn write_u64s_le(out: &mut Vec<u8>, data: &[u64]) {
    out.reserve(data.len() * std::mem::size_of::<u64>());
    for coeff in data {
        out.extend_from_slice(&coeff.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use inspiring::key_switching::KeySwitchingMatrix;
    use inspiring::{GadgetParams, RlweParams};
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use spiral_rs::poly::PolyMatrixNTT;

    use crate::client::{generate_ks_pair, ClientSecret};

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

    fn secret(params: &RlweParams) -> ClientSecret {
        ClientSecret::from_coeffs(params, vec![1, 0, params.q - 1, 1, 0, 1, 0, 0])
    }

    #[test]
    fn serialized_ks_pair_len_matches_two_single_crt_matrices() {
        let params = params();

        assert_eq!(
            serialized_ks_pair_len(&params),
            2 * 2 * params.gadget.ell * params.d * 8
        );
    }

    #[test]
    fn serialize_ks_pair_is_stable_under_fixed_seed() {
        let params = params();
        let secret = secret(&params);
        let mut left_rng = ChaCha20Rng::seed_from_u64(0x5150);
        let mut right_rng = ChaCha20Rng::seed_from_u64(0x5150);
        let left = generate_ks_pair(&params, &secret, &mut left_rng);
        let right = generate_ks_pair(&params, &secret, &mut right_rng);

        let left_bytes = serialize_ks_pair(&params, &left.0, &left.1).expect("serialize");
        let right_bytes = serialize_ks_pair(&params, &right.0, &right.1).expect("serialize");

        assert_eq!(left_bytes, right_bytes);
        assert_eq!(left_bytes.len(), serialized_ks_pair_len(&params));
    }

    #[test]
    fn serialize_ks_pair_writes_kg_then_kh_little_endian() {
        let params = params();
        let kg = KeySwitchingMatrix {
            mat: PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell),
            params: &params,
        };
        let mut kh_mat = PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell);
        kh_mat.as_mut_slice()[0] = 42;
        let kh = KeySwitchingMatrix {
            mat: kh_mat,
            params: &params,
        };

        let bytes = serialize_ks_pair(&params, &kg, &kh).expect("serialize");
        let kg_len = key_matrix_u64_len(&params) * 8;

        assert!(bytes[..kg_len].iter().all(|byte| *byte == 0));
        assert_eq!(&bytes[kg_len..kg_len + 8], &42u64.to_le_bytes());
    }

    #[test]
    fn serialize_ks_pair_rejects_wrong_shape() {
        let params = params();
        let good = KeySwitchingMatrix {
            mat: PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell),
            params: &params,
        };
        let bad = KeySwitchingMatrix {
            mat: PolyMatrixNTT::zero(&params.spiral, 1, params.gadget.ell),
            params: &params,
        };

        let err = serialize_ks_pair(&params, &bad, &good).expect_err("wrong shape must fail");

        assert!(matches!(err, InspiringError::PreprocessMismatch(_)));
    }
}
