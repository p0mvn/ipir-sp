"""Negacyclic polynomial ring R_q = Z_q[X] / (X^d + 1).

This is the substrate for everything else in the oracle. Implementation is
deliberately naive: schoolbook O(d^2) multiplication followed by an explicit
"X^d -> -1" wraparound. No NTT, no batching. Performance is irrelevant.

Conventions:

* Ring elements are length-d ``list[int]`` with coefficients in ``[0, q)``.
* Coefficients are stored **lowest-degree-first**: ``a[i]`` is the coefficient
  of ``X^i``. Index ``0`` is the constant term.
* Every operation takes ``q`` as an explicit parameter; there is no global
  state. (Higher-level modules thread ``params.q`` through.)
* Inputs are length-checked but not range-checked; callers should ensure
  ``0 <= a[i] < q``. Outputs are guaranteed in ``[0, q)``.

The "negacyclic" property -- ``X^d = -1`` in R_q -- is the only non-trivial
piece of mathematics here, and it is the single place where every from-scratch
ring implementation gets it wrong first. See ``test_ring.py`` for the
known-answer test that catches this (``X^{d-1} * X = -1``) and the sympy
oracle-of-oracle that cross-checks 100 random products.
"""

from __future__ import annotations


def add(a: list[int], b: list[int], q: int) -> list[int]:
    """Coefficient-wise addition mod q."""
    return [(x + y) % q for x, y in zip(a, b, strict=True)]


def sub(a: list[int], b: list[int], q: int) -> list[int]:
    """Coefficient-wise subtraction mod q."""
    return [(x - y) % q for x, y in zip(a, b, strict=True)]


def neg(a: list[int], q: int) -> list[int]:
    """Additive inverse mod q (negate every coefficient)."""
    return [(-x) % q for x in a]


def scalar_mul(a: list[int], k: int, q: int) -> list[int]:
    """Multiply every coefficient by integer ``k`` mod q."""
    return [(x * k) % q for x in a]


def mul(a: list[int], b: list[int], q: int) -> list[int]:
    """Polynomial multiplication in R_q = Z_q[X] / (X^d + 1).

    Two-step:

    1. **Schoolbook**: build the length-``2d`` raw product, accumulating
       integer coefficients (no mod q yet -- Python ints are unbounded).
    2. **Negacyclic reduction**: a term of degree ``k >= d`` in the raw
       product contributes ``-1 * coeff`` to position ``k - d`` in the
       output. This is the ``X^d = -1`` rule. Equivalently:
       ``out[k] = raw[k] - raw[k + d]`` for ``k in [0, d)``.

    Caller must supply ``len(a) == len(b)``; raises ``ValueError`` otherwise.
    """
    d = len(a)
    if len(b) != d:
        raise ValueError(f"length mismatch: len(a) = {d}, len(b) = {len(b)}")

    raw = [0] * (2 * d)
    for i in range(d):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(d):
            raw[i + j] += ai * b[j]

    # The negacyclic reduction is:
    # raw[k + d] * X^(k+d)
    # = raw[k + d] * X^k * X^d
    # = -raw[k + d] * X^k
    # high-degree coefficient at degree k + d wraps around to degree k,
    # but with a minus sign.
    return [(raw[k] - raw[k + d]) % q for k in range(d)]
