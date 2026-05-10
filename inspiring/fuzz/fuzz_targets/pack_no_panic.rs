#![no_main]

use inspiring::key_switching::KeySwitchingMatrix;
use inspiring::{pack, GadgetParams, LweBatch, LweCiphertext, PackPreprocessed, RlweParams};
use libfuzzer_sys::fuzz_target;
use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};

fn zero_ks<'a>(params: &'a RlweParams) -> KeySwitchingMatrix<'a> {
    KeySwitchingMatrix {
        mat: PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell),
        params,
    }
}

fn crs<'a>(params: &'a RlweParams) -> PolyMatrixNTT<'a> {
    let raw = PolyMatrixRaw::zero(&params.spiral, params.d, 1);
    raw.ntt()
}

fn b_at(data: &[u8], idx: usize, q: u64) -> u64 {
    let mut bytes = [0_u8; 8];
    let start = idx * 8;
    for (dst, src) in bytes.iter_mut().zip(data.get(start..).unwrap_or_default()) {
        *dst = *src;
    }
    u64::from_le_bytes(bytes) % q
}

fuzz_target!(|data: &[u8]| {
    let params = RlweParams::new(
        8,
        12289,
        4,
        0.1,
        GadgetParams {
            bits_per: 3,
            ell: 5,
        },
    )
    .expect("valid fuzz params");
    let crs = crs(&params);
    let kg = zero_ks(&params);
    let kh = zero_ks(&params);
    let pre = PackPreprocessed::build(&params, &crs, &kg, &kh)
        .expect("valid fuzz preprocessing");
    let batch = LweBatch {
        inner: (0..params.d)
            .map(|idx| LweCiphertext {
                a: vec![0; params.d],
                b: b_at(data, idx, params.q),
            })
            .collect(),
    };

    let _ = pack(&batch, &pre);
});
