# Security Policy

`inspiring` is a research implementation of Algorithm 1 (`InspiRING.Pack`) from
InsPIRe. It is not a complete PIR system and has not received an independent
cryptographic audit.

## Scope

In scope:

- Memory-safety, panic, and denial-of-service bugs in the Rust crate.
- Deviations from Algorithm 1, Theorem 2, or the documented parameter
  constraints in `SPEC.md`.
- Incorrect offline/online separation that would require fresh online
  randomness or extra key material.

Out of scope:

- Implementations of `PartialPack`, PIR query generation, database encoding, or
  response decoding. This crate intentionally does not implement them.
- Security claims for parameters not documented here or in `SPEC.md`.

## Parameter Checklist

The paper reports two 128-bit-security parameter sets for the packing layer:

- `(log d, log q, log p, ell, z) = (10, 28, 6, 8, 2^4)`
- `(log d, log q, log p, ell, z) = (11, 56, 15, 3, 2^19)`

The benchmark harness uses concrete NTT-friendly primes with the same bit-size
and gadget shape:

- `d=1024`, `q=268369921`, `p=64`, `ell=8`, `z=16`
- `d=2048`, `q=36028797018972161`, `p=32768`, `ell=3`, `z=524288`

Before using any other parameters, verify:

- `d` is a power of two.
- `q` is odd, prime or otherwise compatible with the required negacyclic NTT,
  and `q - 1` is divisible by `2d`.
- `p <= q` and the plaintext embedding has enough decryption margin.
- `z = 2^bits_per` and `z^ell >= q`.
- The Theorem 2 bound
  `sigma_pack^2 <= ell * d^2 * z^2 * sigma_chi^2 / 4`
  leaves margin below `Delta / 2`.
- A current lattice-estimator run supports the intended security level.

## Assumptions

- The scheme inherits the circular-security/key-switching assumptions from the
  paper and from the RLWE/LWE setting.
- `KS.Setup` is offline key-generation code and consumes secret-key material.
  The online `pack(b, pre)` path is deterministic and consumes public
  preprocessed key material plus LWE `b` scalars.
- The crate inherits its low-level polynomial arithmetic from the pinned
  Valar `spiral-rs` fork. See `docs/spiral-rs-mapping.md`.

## Reporting

Please report suspected vulnerabilities privately to the repository maintainer.
Do not open public issues for vulnerabilities until a fix or mitigation is
available.
