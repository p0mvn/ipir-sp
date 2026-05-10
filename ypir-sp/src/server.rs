//! Server-side SimplePIR database layout and offline block extraction.
//!
//! This module ports the parts of `/root/ypir/src/server.rs` that are not tied
//! to YPIR's old two-CRT CDKS packing path: transposed DB storage, plaintext
//! query/database multiplication, and the `hint_0` block layout consumed by
//! InspiRING preprocessing.

use inspiring::key_switching::KeySwitchingMatrix;
use inspiring::{
    pack, InspiringError, LweBatch, LweCiphertext, PackPreprocessed, RlweCiphertext, RlweParams,
};
use spiral_rs::poly::{to_ntt_alloc, PolyMatrix, PolyMatrixNTT, PolyMatrixRaw};

use crate::params::YpirSchemeParams;

/// Scalar types accepted by the SimplePIR database.
pub trait ToU64 {
    /// Convert a plaintext database element to a `u64`.
    fn to_u64(self) -> u64;
}

impl ToU64 for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}

/// A YPIR-formatted server database.
///
/// Internally the DB is stored column-major (`col * padded_rows + row`), which
/// matches the layout YPIR's fast dot-product kernels consume. The arithmetic
/// here is intentionally scalar and portable; optimized kernels can be added
/// later without changing the surrounding InspiRING boundary.
#[derive(Debug, Clone)]
pub struct YServer<T> {
    params: YpirSchemeParams,
    db: Vec<T>,
    pad_rows: bool,
}

impl<T> YServer<T>
where
    T: Copy + Default + ToU64,
{
    /// Build a server from a database iterator.
    ///
    /// If `input_is_transposed` is false, the iterator is logical row-major
    /// (`row, col`). If true, it is already in server column-major order. In
    /// both cases the internal storage is column-major.
    pub fn new<I>(
        params: YpirSchemeParams,
        mut db: I,
        input_is_transposed: bool,
        pad_rows: bool,
    ) -> Self
    where
        I: Iterator<Item = T>,
    {
        let rows = params.db_rows;
        let padded_rows = if pad_rows {
            params.db_rows_padded_simplepir()
        } else {
            rows
        };
        let cols = params.db_cols;
        let mut stored = vec![T::default(); padded_rows * cols];

        if input_is_transposed {
            for col in 0..cols {
                for row in 0..rows {
                    stored[col * padded_rows + row] = db.next().expect("database is too short");
                }
            }
        } else {
            for row in 0..rows {
                for col in 0..cols {
                    stored[col * padded_rows + row] = db.next().expect("database is too short");
                }
            }
        }

        Self {
            params,
            db: stored,
            pad_rows,
        }
    }

    /// YPIR scheme parameters.
    #[must_use]
    pub fn params(&self) -> &YpirSchemeParams {
        &self.params
    }

    /// Logical database rows.
    #[must_use]
    pub fn db_rows(&self) -> usize {
        self.params.db_rows
    }

    /// Stored database rows, including optional padding.
    #[must_use]
    pub fn db_rows_padded(&self) -> usize {
        if self.pad_rows {
            self.params.db_rows_padded_simplepir()
        } else {
            self.params.db_rows
        }
    }

    /// SimplePIR database columns.
    #[must_use]
    pub fn db_cols(&self) -> usize {
        self.params.db_cols
    }

    /// Internal column-major DB storage.
    #[must_use]
    pub fn db(&self) -> &[T] {
        &self.db
    }

    /// Return the element at logical `(row, col)`.
    #[must_use]
    pub fn get_elem_row_col(&self, row: usize, col: usize) -> T {
        assert!(row < self.db_rows(), "row out of bounds");
        assert!(col < self.db_cols(), "column out of bounds");
        self.db[col * self.db_rows_padded() + row]
    }

    /// Return a logical database row.
    #[must_use]
    pub fn get_row(&self, row: usize) -> Vec<T> {
        (0..self.db_cols())
            .map(|col| self.get_elem_row_col(row, col))
            .collect()
    }

    /// Multiply one packed first-dimension query by the stored database.
    ///
    /// This is the scalar equivalent of YPIR's `fast_batched_dot_product::<1,
    /// T>` call in `perform_online_computation_simplepir`, with reduction into
    /// InspiRING's single CRT modulus.
    #[must_use]
    pub fn multiply_query(&self, rlwe: &RlweParams, query: &[u64]) -> Vec<u64> {
        let rows = self.db_rows_padded();
        let cols = self.db_cols();
        assert_eq!(query.len(), rows, "query length must match padded rows");

        let mut out = vec![0u64; cols];
        for col in 0..cols {
            let mut acc = 0u128;
            for (row, query_val) in query.iter().enumerate() {
                let db_val = self.db[col * rows + row].to_u64();
                acc += (*query_val as u128) * (db_val as u128);
                acc %= rlwe.q as u128;
            }
            out[col] = acc as u64;
        }
        out
    }
}

/// Offline values that are independent of the user's online query.
#[derive(Debug, Clone)]
pub struct OfflinePrecomputedValues {
    /// YPIR's `hint_0`, laid out as `poly_len x db_cols` in row-major order.
    pub hint_0: Vec<u64>,
    /// CRS blocks extracted from `hint_0`; one block per RLWE output.
    pub crs_blocks: Vec<CrsBlock>,
}

/// One InspiRING CRS block, represented before conversion to `PolyMatrixNTT`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrsBlock {
    /// `d` LWE `a` rows, each with `d` coefficients.
    pub rows: Vec<Vec<u64>>,
}

impl CrsBlock {
    /// Convert this block into the `[d, 1]` NTT CRS shape expected by
    /// [`PackPreprocessed::build`].
    pub fn to_ntt<'a>(&self, params: &'a RlweParams) -> PolyMatrixNTT<'a> {
        assert_eq!(self.rows.len(), params.d, "CRS block must have d rows");

        let mut raw = PolyMatrixRaw::zero(&params.spiral, params.d, 1);
        for (row_idx, row) in self.rows.iter().enumerate() {
            assert_eq!(row.len(), params.d, "CRS row must have d coefficients");
            let poly = raw.get_poly_mut(row_idx, 0);
            for (coeff_idx, coeff) in row.iter().enumerate() {
                poly[coeff_idx] = coeff % params.q;
            }
        }

        to_ntt_alloc(&raw)
    }
}

/// Produce offline values from a supplied `hint_0`.
///
/// The old YPIR implementation continues from this point into CDKS
/// `prep_pack_many_lwes` and `precompute_pack`; `ypir-sp` stops at CRS block
/// extraction so the next layer can call `inspiring::PackPreprocessed::build`.
#[must_use]
pub fn offline_precompute_from_hint(
    rlwe: &RlweParams,
    ypir: &YpirSchemeParams,
    hint_0: Vec<u64>,
) -> OfflinePrecomputedValues {
    assert_eq!(
        hint_0.len(),
        rlwe.d * ypir.db_cols,
        "hint_0 must be poly_len x db_cols"
    );
    assert_eq!(
        ypir.db_cols % rlwe.d,
        0,
        "db_cols must split into RLWE blocks"
    );

    let num_rlwe_outputs = ypir.db_cols / rlwe.d;
    let crs_blocks = (0..num_rlwe_outputs)
        .map(|block| extract_crs_block(rlwe, ypir, &hint_0, block))
        .collect();

    OfflinePrecomputedValues { hint_0, crs_blocks }
}

/// Build InspiRING preprocessing values for already extracted CRS blocks.
///
/// `PackPreprocessed` owns its two key-switching matrices, so the caller
/// supplies one owned `(K_g, K_h)` pair per block. This makes the ownership
/// boundary explicit and avoids adding clone semantics to `inspiring`'s
/// key-switching matrices.
pub fn build_pack_preprocessed_blocks<'a, I>(
    params: &'a RlweParams,
    crs_blocks: &[CrsBlock],
    key_pairs: I,
) -> Result<Vec<PackPreprocessed<'a>>, InspiringError>
where
    I: IntoIterator<Item = (KeySwitchingMatrix<'a>, KeySwitchingMatrix<'a>)>,
{
    let mut out = Vec::with_capacity(crs_blocks.len());
    let mut key_pairs = key_pairs.into_iter();

    for block in crs_blocks {
        let (kg, kh) = key_pairs.next().ok_or_else(|| {
            InspiringError::PreprocessMismatch(format!(
                "expected {} key-switching pairs, got fewer",
                crs_blocks.len()
            ))
        })?;
        let crs = block.to_ntt(params);
        out.push(PackPreprocessed::build(params, &crs, kg, kh)?);
    }

    if key_pairs.next().is_some() {
        return Err(InspiringError::PreprocessMismatch(format!(
            "expected {} key-switching pairs, got more",
            crs_blocks.len()
        )));
    }

    Ok(out)
}

/// Convenience wrapper that extracts CRS blocks from `hint_0` and immediately
/// builds the corresponding InspiRING offline caches.
pub fn build_pack_preprocessed_from_hint<'a, I>(
    params: &'a RlweParams,
    ypir: &YpirSchemeParams,
    hint_0: Vec<u64>,
    key_pairs: I,
) -> Result<(OfflinePrecomputedValues, Vec<PackPreprocessed<'a>>), InspiringError>
where
    I: IntoIterator<Item = (KeySwitchingMatrix<'a>, KeySwitchingMatrix<'a>)>,
{
    let offline = offline_precompute_from_hint(params, ypir, hint_0);
    let pre = build_pack_preprocessed_blocks(params, &offline.crs_blocks, key_pairs)?;
    Ok((offline, pre))
}

/// Pack online SimplePIR intermediate values into RLWE ciphertexts.
///
/// The CRS-side `a` vectors were already absorbed into [`PackPreprocessed`].
/// `inspiring::pack` only reads online `b` scalars, so each constructed
/// [`LweCiphertext`] carries a zero dummy `a` vector of the right shape.
pub fn pack_intermediate_blocks<'a>(
    intermediate: &[u64],
    preprocessed: &'a [PackPreprocessed<'a>],
) -> Result<Vec<RlweCiphertext<'a>>, InspiringError> {
    let Some(first) = preprocessed.first() else {
        return if intermediate.is_empty() {
            Ok(Vec::new())
        } else {
            Err(InspiringError::PreprocessMismatch(
                "non-empty intermediate with no preprocessing blocks".to_string(),
            ))
        };
    };

    let params = first.params;
    if intermediate.len() != preprocessed.len() * params.d {
        return Err(InspiringError::LweShape(format!(
            "expected {} intermediate values for {} blocks of d={}, got {}",
            preprocessed.len() * params.d,
            preprocessed.len(),
            params.d,
            intermediate.len()
        )));
    }

    let mut out = Vec::with_capacity(preprocessed.len());
    for (block_idx, (b_block, pre)) in intermediate
        .chunks_exact(params.d)
        .zip(preprocessed.iter())
        .enumerate()
    {
        if pre.params.d != params.d || pre.params.q != params.q {
            return Err(InspiringError::PreprocessMismatch(format!(
                "preprocessing block {block_idx} uses mismatched RLWE parameters"
            )));
        }

        let batch = LweBatch {
            inner: b_block
                .iter()
                .map(|b| LweCiphertext {
                    a: vec![0; params.d],
                    b: *b,
                })
                .collect(),
        };
        out.push(pack(&batch, pre)?);
    }

    Ok(out)
}

/// Extract one `d x d` InspiRING CRS block from `hint_0`.
///
/// `hint_0` is row-major as `hint_0[row * db_cols + col]`. For RLWE output
/// block `i`, column range `[i*d, (i+1)*d)` becomes the `d` CRS rows. This
/// keeps the single-CRT InspiRING modulus boundary explicit by reducing every
/// coefficient modulo `rlwe.q`.
#[must_use]
pub fn extract_crs_block(
    rlwe: &RlweParams,
    ypir: &YpirSchemeParams,
    hint_0: &[u64],
    block: usize,
) -> CrsBlock {
    assert_eq!(
        hint_0.len(),
        rlwe.d * ypir.db_cols,
        "hint_0 must be poly_len x db_cols"
    );
    assert!(block < ypir.db_cols / rlwe.d, "CRS block out of bounds");

    let col_start = block * rlwe.d;
    let mut rows = vec![vec![0u64; rlwe.d]; rlwe.d];
    for crs_row in 0..rlwe.d {
        let hint_col = col_start + crs_row;
        for coeff in 0..rlwe.d {
            rows[crs_row][coeff] = hint_0[coeff * ypir.db_cols + hint_col] % rlwe.q;
        }
    }

    CrsBlock { rows }
}

impl YpirSchemeParams {
    fn db_rows_padded_simplepir(&self) -> usize {
        self.db_rows
    }
}

#[cfg(test)]
mod tests {
    use inspiring::key_switching::KeySwitchingMatrix;
    use inspiring::{GadgetParams, RlweParams};
    use spiral_rs::poly::{from_ntt_alloc, PolyMatrix, PolyMatrixNTT};

    use super::*;

    fn tiny_rlwe() -> RlweParams {
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

    fn tiny_ypir(db_rows: usize, db_cols: usize) -> YpirSchemeParams {
        YpirSchemeParams {
            num_items: db_rows as u64,
            item_size_bits: (db_cols * 14) as u64,
            poly_len: 8,
            db_dim_1: 0,
            db_dim_2: 1,
            instances: db_cols / 8,
            db_rows,
            db_cols,
            p: 4,
            q_prime_1: 16,
            q_prime_2: 257,
            q2_bits: 8,
            t_exp_left: 3,
            t_exp_right: 2,
        }
    }

    fn zero_ks<'a>(params: &'a RlweParams) -> KeySwitchingMatrix<'a> {
        KeySwitchingMatrix {
            mat: PolyMatrixNTT::zero(&params.spiral, 2, params.gadget.ell),
            params,
        }
    }

    #[test]
    fn server_stores_row_major_input_as_column_major() {
        let ypir = tiny_ypir(4, 3);
        let input = 0u16..12;
        let server = YServer::new(ypir, input, false, true);

        assert_eq!(server.db(), &[0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]);
        assert_eq!(server.get_row(2), vec![6, 7, 8]);
    }

    #[test]
    fn multiply_query_matches_plain_matrix_vector_product_mod_q() {
        let rlwe = tiny_rlwe();
        let ypir = tiny_ypir(4, 3);
        let server = YServer::new(ypir, 0u16..12, false, true);
        let query = [2, 3, 5, 7];

        let result = server.multiply_query(&rlwe, &query);

        assert_eq!(
            result,
            vec![
                2 * 0 + 3 * 3 + 5 * 6 + 7 * 9,
                2 + 3 * 4 + 5 * 7 + 7 * 10,
                2 * 2 + 3 * 5 + 5 * 8 + 7 * 11
            ]
        );
    }

    #[test]
    fn extract_crs_block_maps_hint_columns_to_crs_rows() {
        let rlwe = tiny_rlwe();
        let ypir = tiny_ypir(4, 16);
        let hint_0: Vec<_> = (0..rlwe.d)
            .flat_map(|row| (0..ypir.db_cols).map(move |col| (row * 100 + col) as u64))
            .collect();

        let block = extract_crs_block(&rlwe, &ypir, &hint_0, 1);

        assert_eq!(block.rows[0], vec![8, 108, 208, 308, 408, 508, 608, 708]);
        assert_eq!(block.rows[7], vec![15, 115, 215, 315, 415, 515, 615, 715]);
    }

    #[test]
    fn offline_precompute_splits_one_block_per_rlwe_output() {
        let rlwe = tiny_rlwe();
        let ypir = tiny_ypir(4, 16);
        let hint_0 = vec![1u64; rlwe.d * ypir.db_cols];

        let offline = offline_precompute_from_hint(&rlwe, &ypir, hint_0.clone());

        assert_eq!(offline.hint_0, hint_0);
        assert_eq!(offline.crs_blocks.len(), 2);
        assert_eq!(offline.crs_blocks[0].rows.len(), rlwe.d);
        assert_eq!(offline.crs_blocks[0].rows[0].len(), rlwe.d);
    }

    #[test]
    fn crs_block_converts_to_inspiring_ntt_shape() {
        let rlwe = tiny_rlwe();
        let block = CrsBlock {
            rows: (0..rlwe.d)
                .map(|row| {
                    (0..rlwe.d)
                        .map(|coeff| (row * 100 + coeff) as u64)
                        .collect()
                })
                .collect(),
        };

        let crs = block.to_ntt(&rlwe);
        let raw = from_ntt_alloc(&crs);

        assert_eq!(crs.rows, rlwe.d);
        assert_eq!(crs.cols, 1);
        assert_eq!(
            raw.get_poly(3, 0),
            vec![300, 301, 302, 303, 304, 305, 306, 307]
        );
    }

    #[test]
    fn build_pack_preprocessed_blocks_consumes_one_key_pair_per_block() {
        let rlwe = tiny_rlwe();
        let ypir = tiny_ypir(4, 16);
        let hint_0 = vec![1u64; rlwe.d * ypir.db_cols];
        let offline = offline_precompute_from_hint(&rlwe, &ypir, hint_0);
        let key_pairs = (0..offline.crs_blocks.len()).map(|_| (zero_ks(&rlwe), zero_ks(&rlwe)));

        let pre =
            build_pack_preprocessed_blocks(&rlwe, &offline.crs_blocks, key_pairs).expect("build");

        assert_eq!(pre.len(), 2);
        assert_eq!(pre[0].a_hat.len(), rlwe.d);
        assert_eq!(pre[0].a_agg.len(), rlwe.d);
    }

    #[test]
    fn build_pack_preprocessed_blocks_rejects_wrong_key_pair_count() {
        let rlwe = tiny_rlwe();
        let block = CrsBlock {
            rows: vec![vec![0; rlwe.d]; rlwe.d],
        };

        let err = match build_pack_preprocessed_blocks(&rlwe, &[block], std::iter::empty()) {
            Ok(_) => panic!("missing key pair must fail"),
            Err(err) => err,
        };

        assert!(matches!(err, InspiringError::PreprocessMismatch(_)));
    }

    #[test]
    fn pack_intermediate_blocks_routes_b_values_to_matching_rlwe_outputs() {
        let rlwe = tiny_rlwe();
        let ypir = tiny_ypir(4, 16);
        let hint_0 = vec![0u64; rlwe.d * ypir.db_cols];
        let offline = offline_precompute_from_hint(&rlwe, &ypir, hint_0);
        let key_pairs = (0..offline.crs_blocks.len()).map(|_| (zero_ks(&rlwe), zero_ks(&rlwe)));
        let pre =
            build_pack_preprocessed_blocks(&rlwe, &offline.crs_blocks, key_pairs).expect("build");
        let intermediate: Vec<_> = (0..ypir.db_cols).map(|idx| idx as u64 + 10).collect();

        let packed = pack_intermediate_blocks(&intermediate, &pre).expect("pack");

        assert_eq!(packed.len(), 2);
        for (block_idx, ct) in packed.iter().enumerate() {
            let raw = from_ntt_alloc(&ct.inner);
            let expected = intermediate[block_idx * rlwe.d..(block_idx + 1) * rlwe.d].to_vec();
            assert_eq!(raw.get_poly(1, 0), expected);
        }
    }

    #[test]
    fn pack_intermediate_blocks_rejects_wrong_intermediate_length() {
        let rlwe = tiny_rlwe();
        let block = CrsBlock {
            rows: vec![vec![0; rlwe.d]; rlwe.d],
        };
        let pre =
            build_pack_preprocessed_blocks(&rlwe, &[block], [(zero_ks(&rlwe), zero_ks(&rlwe))])
                .expect("build");

        let err = match pack_intermediate_blocks(&[1, 2, 3], &pre) {
            Ok(_) => panic!("wrong intermediate length must fail"),
            Err(err) => err,
        };

        assert!(matches!(err, InspiringError::LweShape(_)));
    }
}
