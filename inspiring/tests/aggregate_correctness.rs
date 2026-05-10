use inspiring::automorph::{h, tau_g_pow};
use inspiring::intermediate::{aggregate, transform};
use inspiring::{GadgetParams, LweCiphertext, RlweParams};
use spiral_rs::poly::{from_ntt_alloc, PolyMatrix};

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
    .expect("valid tiny test parameters")
}

fn raw_coeffs(poly: &spiral_rs::poly::PolyMatrixRaw<'_>) -> Vec<u64> {
    poly.get_poly(0, 0).to_vec()
}

fn ntt_coeffs(poly: &spiral_rs::poly::PolyMatrixNTT<'_>) -> Vec<u64> {
    raw_coeffs(&from_ntt_alloc(poly))
}

fn add_assign(acc: &mut [u64], rhs: &[u64], q: u64) {
    for (out, value) in acc.iter_mut().zip(rhs) {
        *out = (*out + value) % q;
    }
}

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

fn negacyclic_mul(lhs: &[u64], rhs: &[u64], q: u64) -> Vec<u64> {
    let d = lhs.len();
    let mut out = vec![0; d];

    for (i, lhs_coeff) in lhs.iter().enumerate() {
        for (j, rhs_coeff) in rhs.iter().enumerate() {
            let product = (u128::from(*lhs_coeff) * u128::from(*rhs_coeff) % u128::from(q)) as u64;
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

fn s_hat_from_s_tilde(s_tilde: &[u64], q: u64) -> Vec<Vec<u64>> {
    let d = s_tilde.len();
    let two_d = 2 * d as u64;
    let h_d = h(d);
    let mut out = Vec::with_capacity(d);

    for j in 0..(d / 2) {
        out.push(tau_coeffs(s_tilde, tau_g_pow(j, d), q));
    }
    for j in 0..(d / 2) {
        out.push(tau_coeffs(s_tilde, (tau_g_pow(j, d) * h_d) % two_d, q));
    }

    out
}

fn lwe_for_message(params: &RlweParams, a: Vec<u64>, s: &[u64], message: u64) -> LweCiphertext {
    let inner_product = a.iter().zip(s).fold(0_u64, |acc, (ai, si)| {
        (acc + (u128::from(*ai) * u128::from(*si) % u128::from(params.q)) as u64) % params.q
    });
    let encoded = (params.delta * message) % params.q;
    let b = (params.q + encoded - inner_product) % params.q;

    LweCiphertext { a, b }
}

fn decrypt_under_s_hat(
    ictx: &inspiring::intermediate::IRCtx<'_>,
    s_hat: &[Vec<u64>],
    q: u64,
) -> Vec<u64> {
    let mut decrypted = raw_coeffs(&ictx.b_tilde);
    for (a_hat_slot, s_hat_slot) in ictx.a_hat.iter().zip(s_hat) {
        add_assign(
            &mut decrypted,
            &negacyclic_mul(&ntt_coeffs(a_hat_slot), s_hat_slot, q),
            q,
        );
    }
    decrypted
}

#[test]
fn aggregate_decrypts_to_one_encoded_message_per_slot() {
    let params = params();
    let s_tilde = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let messages = vec![0, 1, 2, 3, 3, 2, 1, 0];
    let irctxs: Vec<_> = messages
        .iter()
        .enumerate()
        .map(|(row, message)| {
            let a = (0..params.d)
                .map(|col| (row * params.d + col + 7) as u64)
                .collect();
            transform(&params, &lwe_for_message(&params, a, &s_tilde, *message))
        })
        .collect();

    let agg = aggregate(&params, &irctxs);
    let s_hat = s_hat_from_s_tilde(&s_tilde, params.q);
    let decrypted = decrypt_under_s_hat(&agg, &s_hat, params.q);
    let expected: Vec<_> = messages
        .iter()
        .map(|message| (params.delta * message) % params.q)
        .collect();

    assert_eq!(decrypted, expected);
}

#[test]
fn aggregate_routes_b_values_to_matching_coefficients() {
    let params = params();
    let irctxs: Vec<_> = (0..params.d)
        .map(|idx| {
            transform(
                &params,
                &LweCiphertext {
                    a: vec![idx as u64; params.d],
                    b: params.q + idx as u64 * 13,
                },
            )
        })
        .collect();

    let agg = aggregate(&params, &irctxs);
    let expected: Vec<_> = (0..params.d).map(|idx| idx as u64 * 13).collect();

    assert_eq!(raw_coeffs(&agg.b_tilde), expected);
}
