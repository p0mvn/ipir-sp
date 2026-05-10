# IPIR+SP

A Rust workspace implementing **IPIR+SP**: a Private Information Retrieval (PIR)
scheme that replaces the CDKS ring-packing path in
[YPIR+SP](https://eprint.iacr.org/2024/270) with a custom implementation of
**InsPIRing.Pack** from [ePrint 2025/1352](https://eprint.iacr.org/2025/1352).

The motivation is communication efficiency: InsPIRing collapses `d` LWE
ciphertexts into one RLWE ciphertext using exactly **two** key-switching
matrices (`K_g`, `K_h`), instead of the `log d` expansion matrices CDKS
uploads. The result is a smaller offline upload and reduced online server time
at comparable parameter sets, in exchange for heavier server-side preprocessing.

See [`roman_notes.md`](roman_notes.md) for an informal walkthrough of the math
(trace tricks, partial trace, key-switch chain) and
[`inspiring/SPEC.md`](inspiring/SPEC.md) for the full paper-to-code contract.

## Workspace Layout

```
.
|-- inspiring/        # Custom InsPIRing.Pack implementation (Algorithm 1)
|-- ipir-sp/          # IPIR+SP integration: YPIR+SP wired to inspiring::pack
|-- plans/            # Implementation plans for both crates
|-- roman_notes.md    # Informal notes on InsPIRing math and design choices
`-- Cargo.toml        # Workspace manifest (resolver = "2")
```

### `inspiring/` — InsPIRing.Pack

Standalone crate exposing a single primitive:

```rust
pub fn pack<'a>(b: &LweBatch, pre: &'a PackPreprocessed<'a>)
    -> Result<RlweCiphertext<'a>, InspiringError>;
```

It compresses `d` LWE ciphertexts (each of LWE dimension `d`) into one degree-`d`
RLWE ciphertext using two key-switching matrices. The implementation tracks
Algorithm 1 of the InsPIRe paper line by line and is cross-checked against a
Python reference oracle and the public Google reference implementation. See
[`inspiring/README.md`](inspiring/README.md).

### `ipir-sp/` — IPIR+SP glue

Integration crate that keeps YPIR's SimplePIR database/query arithmetic and
swaps the CDKS packing boundary for `inspiring::pack`. It targets the IPIR-SP
parameter set from Table 5, row 2 of ePrint 2024/270, using a single CRT modulus
on the RLWE side. See [`ipir-sp/README.md`](ipir-sp/README.md) and
[`ipir-sp/MIGRATION.md`](ipir-sp/MIGRATION.md) for the YPIR-to-IPIR+SP API map.

## Backend

Both crates share a single resolved [`spiral-rs`](https://github.com/valargroup/spiral-rs)
backend, pinned at the workspace root to Valar's fork:

```toml
[workspace.dependencies]
spiral-rs = { package = "valar-spiral-rs", git = "https://github.com/valargroup/spiral-rs.git", rev = "6f5b66c6a5a639827c6486c59d31c7ec2d4399a8" }
```

The fork keeps the scalar single-CRT multiply path correct and provides a
non-AVX-512 NTT, so the workspace builds on stable Rust without any
`target-cpu` override.

## Build, Test, Bench

From the workspace root:

```bash
cargo build --release
cargo test
cargo test -p inspiring
cargo test -p ipir-sp
```

Benchmarks (Criterion) are per-crate:

```bash
cargo bench -p inspiring --bench pack
cargo bench -p ipir-sp --bench end_to_end
```

The default `ipir-sp` benchmark uses a small `d = 64` development profile.
Set `IPIR_SP_BENCH_FULL=1` to attempt the full
`params_for_simplepir(32768, 131072)` profile (`d = 2048`, ~7+ GiB RAM
during preprocessing). Latest local benchmark notes live in
[`ipir-sp/bench/REPORT.md`](ipir-sp/bench/REPORT.md) and
[`inspiring/bench/REPORT.md`](inspiring/bench/REPORT.md).

## High-Level Flow

For a single SimplePIR query:

1. **Params.** `ipir_sp::params_for_simplepir(num_items, item_size_bits)` returns
   an `inspiring::RlweParams` plus YPIR transport and database dimensions.
2. **Client setup.** Sample a ternary RLWE secret and generate one `(K_g, K_h)`
   pair per RLWE output block via `client::generate_ks_pairs`.
3. **Server offline.** `YServer::perform_offline_precomputation_simplepir`
   computes `hint_0`, splits it into CRS blocks, and builds an
   `inspiring::PackPreprocessed` cache for each block.
4. **Server online.** `YServer::perform_online_computation_simplepir` runs the
   SimplePIR matrix product, packs each intermediate `b` block with
   `inspiring::pack`, and serializes the response with single-CRT row-wise
   modulus switching.
5. **Client decode.** Standard RLWE decryption on the recovered rows; do **not**
   apply YPIR's extra `poly_len` multiplier (InsPIRing absorbs the `d^-1`
   scaling internally).

A worked example lives in [`ipir-sp/README.md`](ipir-sp/README.md#basic-flow).

## References

- IPIR+SP / YPIR+SP: ePrint 2024/270 — <https://eprint.iacr.org/2024/270>
- InsPIRe / InsPIRing.Pack: ePrint 2025/1352 — <https://eprint.iacr.org/2025/1352>
- Google reference implementation:
  <https://github.com/google/private-membership/tree/main/research/InsPIRe>
- Local InsPIRing spec: [`inspiring/SPEC.md`](inspiring/SPEC.md)

## License

Dual-licensed under MIT or Apache-2.0, at your option. See
[`inspiring/LICENSE-MIT`](inspiring/LICENSE-MIT) and
[`inspiring/LICENSE-APACHE`](inspiring/LICENSE-APACHE).
