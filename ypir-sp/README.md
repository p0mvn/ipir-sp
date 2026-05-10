# ypir-sp

`ypir-sp` is the YPIR-SP integration crate for this workspace. It keeps YPIR's
SimplePIR database/query arithmetic and replaces the old CDKS ring-packing path
with `inspiring::pack`, the Rust implementation of `InspiRING.Pack`.

The implementation targets the YPIR-SP parameter set corresponding to Table 5
row 2 of ePrint 2024/270, using a single CRT modulus on the RLWE side. The
packing primitive and its invariants come from the sibling `inspiring` crate,
which implements Algorithm 1 from ePrint 2025/1352 and documents the math in
`../inspiring/SPEC.md`.

## Workspace Role

This crate is intentionally a glue layer:

- `params` maps YPIR SimplePIR scenario inputs to `inspiring::RlweParams` plus
  YPIR-specific transport and database dimensions.
- `client` generates the two InspiRING key-switching matrices, `K_g` and `K_h`,
  replacing YPIR's `log d` CDKS expansion matrices.
- `server` stores the SimplePIR database, computes YPIR's `hint_0`, extracts CRS
  blocks, builds `PackPreprocessed`, and runs online packing.
- `modulus_switch` serializes single-CRT packed RLWE responses into YPIR-style
  transport bytes.
- `serialize` provides a stable wire helper for uploaded key material.

`spiral-rs` is resolved once at the workspace root through Valar's
`valar-spiral-rs` fork. `ypir-sp` depends on `inspiring` by path and shares that
same resolved backend.

## Basic Flow

```rust
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use ypir_sp::client::{generate_ks_pairs, ClientSecret};
use ypir_sp::server::{build_pack_preprocessed_blocks, YServer};
use ypir_sp::params_for_simplepir;

let (rlwe, ypir) = params_for_simplepir(1 << 14, 16_384 * 8)?;
let db = vec![0u16; ypir.db_rows * ypir.db_cols];
let server = YServer::new(ypir.clone(), db.into_iter(), false, true);

let secret = ClientSecret::sample_ternary(&rlwe, &mut ChaCha20Rng::seed_from_u64(7));
let offline_query = vec![vec![0; rlwe.d]; ypir.db_rows / rlwe.d];
let offline = server.perform_offline_precomputation_simplepir(&rlwe, &offline_query);

let mut rng = ChaCha20Rng::seed_from_u64(8);
let key_pairs = generate_ks_pairs(&rlwe, &secret, offline.crs_blocks.len(), &mut rng);
let preprocessed = build_pack_preprocessed_blocks(&rlwe, &offline.crs_blocks, key_pairs)?;

let first_dim_query = vec![0; ypir.db_rows];
let response = server.perform_online_computation_simplepir(
    &rlwe,
    &first_dim_query,
    &preprocessed,
)?;
# Ok::<(), inspiring::InspiringError>(())
```

## Tests And Benchmarks

Run the crate tests with:

```bash
cargo test -p ypir-sp
```

The integration tests cover the offline/online flow, exact row recovery for
small deterministic fixtures, single-CRT response switching, and the linear
`d - 1` key-switch count per InspiRING pack.

Criterion benchmarks live in `benches/end_to_end.rs`:

```bash
cargo bench -p ypir-sp --bench end_to_end
```

The default benchmark uses a smaller development profile. Set
`YPIR_SP_BENCH_FULL=1` to attempt the full `params_for_simplepir(32768, 131072)`
profile. See `bench/REPORT.md` for the latest local run notes and the paper
comparison targets.

## References

- YPIR-SP: ePrint 2024/270, `https://eprint.iacr.org/2024/270`
- InspiRING / InsPIRe: ePrint 2025/1352, `https://eprint.iacr.org/2025/1352`
- Local InspiRING specification: `../inspiring/SPEC.md`
