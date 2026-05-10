# InspiRING.Pack Benchmark Report

Phase 10 adds Criterion benchmarks in `benches/pack.rs` for the two parameter
sets reported in paper Table 5.

## How To Run

```bash
cd inspiring
cargo bench --bench pack
```

The benchmark processes `4096` input LWE ciphertexts per Criterion iteration:

- Parameter set 1 runs `4` independent `d = 1024` pack chunks.
- Parameter set 2 runs `2` independent `d = 2048` pack chunks.

Each run reports:

- `offline_preprocess/4096`: CRS-side preprocessing for all chunks.
- `online_pack/4096`: online `pack(b, pre)` for all chunks.
- A one-time stderr summary with raw key size, packed ciphertext size, and
  observed `||e_pack||_inf` for the deterministic benchmark fixture.

## Parameter Sets

| Set | `(log d, log q, log p, ell, z)` | Concrete Rust params |
| --- | --- | --- |
| 1 | `(10, 28, 6, 8, 2^4)` | `d=1024`, `q=268369921`, `p=64`, `ell=8`, `bits_per=4` |
| 2 | `(11, 56, 15, 3, 2^19)` | `d=2048`, `q=36028797018972161`, `p=32768`, `ell=3`, `bits_per=19` |

Both concrete moduli are odd NTT-friendly primes with `q - 1` divisible by
`2d`, and satisfy the `spiral-rs` gadget-width rule enforced by
`RlweParams::new`.

## Paper Targets

| Set | Paper comparison | Paper key material | Paper online time |
| --- | --- | ---: | ---: |
| 1 | HintlessPIR vs InspiRING | 60 KiB | ~16 ms |
| 2 | CDKS vs InspiRING | 84 KiB | ~40 ms |

The benchmark's printed `raw_key` is the current in-memory Rust estimate for
two `[2 x ell]` key-switching matrices stored as `u64` coefficient limbs. It is
not compressed to the paper's byte accounting, so compare it as an engineering
memory footprint rather than as the publication's serialized key-material
number.

## Latest Local Results

Not recorded in-tree yet. Run `cargo bench --bench pack` on the target
AVX-512 host and copy Criterion's `offline_preprocess/4096` and
`online_pack/4096` medians here when producing a release artifact.
