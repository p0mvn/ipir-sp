use inspiring::intermediate::{aggregate, transform};
use inspiring::key_switching::KeySwitchingMatrix;
use inspiring::{pack, GadgetParams, LweBatch, LweCiphertext, PackPreprocessed, RlweParams};
use serde_json::Value;
use spiral_rs::poly::{from_ntt_alloc, stack_ntt, PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};

const FIXTURE_JSON: &str = include_str!("../fixtures/tiny_random_seed_42.json");

fn array_u64(value: &Value) -> Vec<u64> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|v| v.as_u64().expect("expected u64"))
        .collect()
}

fn array_i64_mod_q(value: &Value, q: u64) -> Vec<u64> {
    value
        .as_array()
        .expect("expected array")
        .iter()
        .map(|v| v.as_i64().expect("expected i64").rem_euclid(q as i64) as u64)
        .collect()
}

fn matrix(value: &Value) -> Vec<Vec<u64>> {
    value
        .as_array()
        .expect("expected matrix")
        .iter()
        .map(array_u64)
        .collect()
}

fn coeffs_ntt(poly: &PolyMatrixNTT<'_>) -> Vec<u64> {
    from_ntt_alloc(poly).get_poly(0, 0).to_vec()
}

fn coeffs_stacked(poly: &PolyMatrixNTT<'_>) -> (Vec<u64>, Vec<u64>) {
    let raw = from_ntt_alloc(poly);
    (raw.get_poly(0, 0).to_vec(), raw.get_poly(1, 0).to_vec())
}

fn ks_from_fixture<'a>(params: &'a RlweParams, value: &Value) -> KeySwitchingMatrix<'a> {
    let w_rows = matrix(&value["w"]);
    let y_rows = matrix(&value["y"]);
    assert_eq!(w_rows.len(), params.gadget.ell);
    assert_eq!(y_rows.len(), params.gadget.ell);

    let mut w = PolyMatrixRaw::zero(&params.spiral, 1, params.gadget.ell);
    let mut y = PolyMatrixRaw::zero(&params.spiral, 1, params.gadget.ell);
    for col in 0..params.gadget.ell {
        w.get_poly_mut(0, col).copy_from_slice(&w_rows[col]);
        y.get_poly_mut(0, col).copy_from_slice(&y_rows[col]);
    }

    KeySwitchingMatrix {
        mat: stack_ntt(&w.ntt(), &y.ntt()),
        params,
    }
}

fn crs_from_lwes<'a>(params: &'a RlweParams, lwes: &[LweCiphertext]) -> PolyMatrixNTT<'a> {
    let mut crs = PolyMatrixRaw::zero(&params.spiral, params.d, 1);
    for (row, ct) in lwes.iter().enumerate() {
        crs.get_poly_mut(row, 0).copy_from_slice(&ct.a);
    }
    crs.ntt()
}

#[test]
fn rust_stages_match_python_oracle_fixture_for_d8_seed_42() {
    let fixture: Value = serde_json::from_str(FIXTURE_JSON).expect("valid fixture JSON");
    assert_eq!(fixture["schema_version"].as_u64(), Some(1));

    let p = &fixture["params"];
    let params = RlweParams::new(
        p["d"].as_u64().expect("d") as usize,
        p["q"].as_u64().expect("q"),
        p["p"].as_u64().expect("p"),
        p["sigma"].as_f64().expect("sigma"),
        GadgetParams {
            bits_per: p["z"].as_u64().expect("z").ilog2(),
            ell: p["ell"].as_u64().expect("ell") as usize,
        },
    )
    .expect("fixture params are valid");

    let lwes: Vec<_> = fixture["lwes"]
        .as_array()
        .expect("lwes")
        .iter()
        .map(|lwe| LweCiphertext {
            a: array_u64(&lwe["a"]),
            b: lwe["b"].as_u64().expect("b"),
        })
        .collect();
    let irctxs: Vec<_> = lwes.iter().map(|ct| transform(&params, ct)).collect();

    for (idx, (actual, expected)) in irctxs
        .iter()
        .zip(
            fixture["transform_outputs"]
                .as_array()
                .expect("transform outputs"),
        )
        .enumerate()
    {
        assert_eq!(
            actual.a_hat.iter().map(coeffs_ntt).collect::<Vec<_>>(),
            matrix(&expected["a_hat"]),
            "transform[{idx}].a_hat"
        );
        assert_eq!(
            actual.b_tilde.get_poly(0, 0),
            array_u64(&expected["b_tilde"]),
            "transform[{idx}].b_tilde"
        );
    }

    let agg = aggregate(&params, &irctxs);
    assert_eq!(
        agg.a_hat.iter().map(coeffs_ntt).collect::<Vec<_>>(),
        matrix(&fixture["aggregate_output"]["a_hat"]),
        "aggregate.a_hat"
    );
    assert_eq!(
        agg.b_tilde.get_poly(0, 0),
        array_u64(&fixture["aggregate_output"]["b_tilde"]),
        "aggregate.b_tilde"
    );

    let batch = LweBatch { inner: lwes };
    let crs = crs_from_lwes(&params, &batch.inner);
    let kg = ks_from_fixture(&params, &fixture["K_g"]);
    let kh = ks_from_fixture(&params, &fixture["K_h"]);
    let pre = PackPreprocessed::build(&params, &crs, &kg, &kh).expect("valid preprocessing");
    let packed = pack(&batch, &pre).expect("pack succeeds");
    let (actual_c1, actual_c2) = coeffs_stacked(&packed.inner);

    assert_eq!(actual_c1, array_u64(&fixture["packed"]["c1"]), "packed.c1");
    assert_eq!(actual_c2, array_u64(&fixture["packed"]["c2"]), "packed.c2");
    assert_eq!(
        array_i64_mod_q(&fixture["s"], params.q),
        array_u64(&fixture["s_tilde"]),
        "fixture secret conversion sanity"
    );
}
