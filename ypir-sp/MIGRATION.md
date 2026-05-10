# Migrating YPIR Packing Code To ypir-sp

This note maps the YPIR CDKS packing surface to the corresponding `ypir-sp`
entry points. The SimplePIR matrix layer remains the same conceptually; only the
LWE-to-RLWE packing boundary changes.

## Main Conceptual Changes

YPIR's original packing path uploads `log d` expansion matrices and performs a
CDKS-style recursive packing. `ypir-sp` uploads exactly two InspiRING
key-switching matrices per preprocessing block, `K_g` and `K_h`, then calls
`inspiring::pack` once per RLWE output block.

The RLWE side is single-CRT throughout. Response transport still uses YPIR-style
reduced moduli, but `modulus_switch` performs row-wise switching from one
InspiRING modulus rather than from two CRT limbs.

## API Mapping

`params_for_scenario_simplepir`

Use `ypir_sp::params_for_simplepir(num_items, item_size_bits)`. It returns both
the `inspiring::RlweParams` used by the packing layer and the
`YpirSchemeParams` values retained for database shape and transport.

`raw_generate_expansion_params`

Use `ypir_sp::client::generate_ks_pair` for one `(K_g, K_h)` pair or
`ypir_sp::client::generate_ks_pairs` when building several preprocessing
blocks. The uploaded key material is serialized with
`ypir_sp::serialize::serialize_ks_pair`.

`pack_pub_params`

There is no separate public-parameter struct in `ypir-sp`. The CRS data is
derived from YPIR's `hint_0` and represented as `ypir_sp::server::CrsBlock`
values.

`generate_fake_pack_pub_params`

Use deterministic `hint_0` fixtures and `ypir_sp::server::offline_precompute_from_hint`
in tests. For a real server flow, use `YServer::perform_offline_precomputation_simplepir`.

`prep_pack_many_lwes`

Use `ypir_sp::server::offline_precompute_from_hint` to split `hint_0` into
CRS blocks, then `ypir_sp::server::build_pack_preprocessed_blocks` to call
`inspiring::PackPreprocessed::build` for each block.

`precompute_pack`

Use `ypir_sp::server::build_pack_preprocessed_blocks` if CRS blocks already
exist, or `ypir_sp::server::build_pack_preprocessed_from_hint` to extract CRS
blocks and build `PackPreprocessed` caches in one step.

`pack_many_lwes`

Use `ypir_sp::server::pack_intermediate_blocks`. It constructs the online
`LweBatch` values and invokes `inspiring::pack` once for each RLWE output block.

`pack_lwes_inner_non_recursive`

There is no `ypir-sp` equivalent. InspiRING's linear cascade is implemented in
the sibling `inspiring` crate and is exercised through `inspiring::pack`.

`perform_offline_precomputation_simplepir`

Use `YServer::perform_offline_precomputation_simplepir` to compute `hint_0` and
extract CRS blocks. Then pass the blocks and generated key pairs to
`build_pack_preprocessed_blocks`.

`perform_online_computation_simplepir`

Use `YServer::perform_online_computation_simplepir`. It runs the SimplePIR
matrix product, packs each intermediate block with InspiRING, and returns
serialized response bytes.

`modulus_switch` helpers for two CRT limbs

Use `ypir_sp::modulus_switch::switch_rlwe_ciphertext` or
`ypir_sp::modulus_switch::serialize_rlwe_response` for server responses. Tests
and local decoders can use `recover_rlwe_rows`.

`bits::{write_bits, read_bits}`

The helpers remain available as `ypir_sp::bits::{write_bits, read_bits}` and
are used by the single-CRT response serializer.

## Decode Note

Do not apply YPIR's extra `poly_len` multiplier to packed `b` values when
decoding `ypir-sp` responses. InspiRING absorbs the relevant `d^-1` scaling
inside its transform, so applying the old multiplier would double-scale the
message.

## Validation Checklist

After moving code to the `ypir-sp` API, run:

```bash
cargo test -p ypir-sp
cargo test -p inspiring
```

For performance checks, run:

```bash
cargo bench -p ypir-sp --bench end_to_end
```

Use `YPIR_SP_BENCH_FULL=1` only on a host with enough memory for the full
`d = 2048` preprocessing fixture.
