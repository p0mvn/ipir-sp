"""Stage 1: tests for the negacyclic ring R_q = Z_q[X] / (X^d + 1).

Test groups:

1. ``TestAdd`` / ``TestSub`` / ``TestNeg`` / ``TestScalarMul`` -- standard
   abelian group / module axioms for the additive operations.
2. ``TestMul`` -- ring axioms for polynomial multiplication: zero, one,
   commutativity, associativity, distributivity over addition.
3. ``TestNegacyclic`` -- known-answer tests for the SPEC.md section 1
   identity ``X^d = -1``. Includes the canonical KAT
   ``X^{d-1} * X = -1`` plus a hand-computed ``X^4 * X^5`` at d=8.
4. ``TestSympyOracle`` -- independent reference using ``sympy``: 100 random
   products at d=8 and 20 at d=16 must agree with our ``mul``. This is the
   firewall against the negacyclic-vs-cyclic confusion bug.
5. ``TestLengthValidation`` -- length-mismatch rejection.
"""

from __future__ import annotations

import random

import pytest
import sympy as sp

from inspiring_oracle.params import ORACLE_SMALL, ORACLE_TINY
from inspiring_oracle.ring import add, mul, neg, scalar_mul, sub


def rand_poly(rng: random.Random, d: int, q: int) -> list[int]:
    return [rng.randrange(q) for _ in range(d)]


@pytest.fixture
def params():
    return ORACLE_TINY


@pytest.fixture
def rng():
    return random.Random(0xC0FFEE)


# --------------------------------------------------------------------------
# Additive operations
# --------------------------------------------------------------------------


class TestAdd:
    def test_zero_is_identity(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        zero = [0] * params.d
        assert add(a, zero, params.q) == a
        assert add(zero, a, params.q) == a

    def test_commutative(self, params, rng):
        for _ in range(50):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            assert add(a, b, params.q) == add(b, a, params.q)

    def test_associative(self, params, rng):
        for _ in range(50):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            c = rand_poly(rng, params.d, params.q)
            assert add(add(a, b, params.q), c, params.q) == add(
                a, add(b, c, params.q), params.q
            )

    def test_output_in_range(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        b = rand_poly(rng, params.d, params.q)
        result = add(a, b, params.q)
        assert all(0 <= x < params.q for x in result)


class TestNeg:
    def test_neg_is_additive_inverse(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        zero = [0] * params.d
        assert add(a, neg(a, params.q), params.q) == zero

    def test_neg_neg_is_identity(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        assert neg(neg(a, params.q), params.q) == a


class TestSub:
    def test_sub_equals_add_neg(self, params, rng):
        for _ in range(50):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            assert sub(a, b, params.q) == add(a, neg(b, params.q), params.q)

    def test_sub_self_is_zero(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        assert sub(a, a, params.q) == [0] * params.d


class TestScalarMul:
    def test_scale_by_zero_kills(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        assert scalar_mul(a, 0, params.q) == [0] * params.d

    def test_scale_by_one_is_identity(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        assert scalar_mul(a, 1, params.q) == a

    def test_distributes_over_add(self, params, rng):
        for _ in range(20):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            k = rng.randrange(params.q)
            lhs = scalar_mul(add(a, b, params.q), k, params.q)
            rhs = add(
                scalar_mul(a, k, params.q),
                scalar_mul(b, k, params.q),
                params.q,
            )
            assert lhs == rhs


# --------------------------------------------------------------------------
# Ring multiplication
# --------------------------------------------------------------------------


class TestMul:
    def test_zero_kills(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        zero = [0] * params.d
        assert mul(a, zero, params.q) == zero
        assert mul(zero, a, params.q) == zero

    def test_one_is_identity(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        one = [1] + [0] * (params.d - 1)
        assert mul(a, one, params.q) == a
        assert mul(one, a, params.q) == a

    def test_commutative(self, params, rng):
        for _ in range(50):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            assert mul(a, b, params.q) == mul(b, a, params.q)

    def test_associative(self, params, rng):
        for _ in range(50):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            c = rand_poly(rng, params.d, params.q)
            lhs = mul(a, mul(b, c, params.q), params.q)
            rhs = mul(mul(a, b, params.q), c, params.q)
            assert lhs == rhs

    def test_distributes_over_add(self, params, rng):
        for _ in range(50):
            a = rand_poly(rng, params.d, params.q)
            b = rand_poly(rng, params.d, params.q)
            c = rand_poly(rng, params.d, params.q)
            lhs = mul(a, add(b, c, params.q), params.q)
            rhs = add(
                mul(a, b, params.q),
                mul(a, c, params.q),
                params.q,
            )
            assert lhs == rhs

    def test_output_in_range(self, params, rng):
        a = rand_poly(rng, params.d, params.q)
        b = rand_poly(rng, params.d, params.q)
        result = mul(a, b, params.q)
        assert all(0 <= x < params.q for x in result)


# --------------------------------------------------------------------------
# The negacyclic property: X^d = -1 in R_q
# --------------------------------------------------------------------------


def x_pow(k: int, d: int) -> list[int]:
    """The monomial X^k as a length-d coefficient list (assumes 0 <= k < d)."""
    out = [0] * d
    out[k] = 1
    return out


class TestNegacyclic:
    def test_x_pow_d_minus_1_times_x_equals_minus_one(self, params):
        """The canonical KAT: X^{d-1} * X = X^d = -1, which is [q-1, 0, ..., 0]."""
        d, q = params.d, params.q
        result = mul(x_pow(d - 1, d), x_pow(1, d), q)
        expected = [q - 1] + [0] * (d - 1)
        assert result == expected

    def test_x_pow_d_via_x_times_x_pow_d_minus_1(self, params):
        """Same identity in the other order; rules out a one-sided bug in mul."""
        d, q = params.d, params.q
        result = mul(x_pow(1, d), x_pow(d - 1, d), q)
        expected = [q - 1] + [0] * (d - 1)
        assert result == expected

    def test_x_pow_4_times_x_pow_5_at_d8(self):
        """Hand-checked: X^4 * X^5 = X^9 = X * X^8 = X * (-1) = -X.

        At d = 8, q = 12289 this is ``[0, q-1, 0, 0, 0, 0, 0, 0]``.
        """
        d, q = 8, 12289
        result = mul(x_pow(4, d), x_pow(5, d), q)
        expected = [0, q - 1, 0, 0, 0, 0, 0, 0]
        assert result == expected

    def test_all_monomial_pairs_at_d8(self):
        """Exhaustive: for every pair (i, j) in [0, 8) x [0, 8) check X^i * X^j.

        If i + j < d the result is X^{i+j} (positive); else it is
        -X^{i+j-d} (sign flip by the negacyclic rule).
        """
        d, q = 8, 12289
        for i in range(d):
            for j in range(d):
                result = mul(x_pow(i, d), x_pow(j, d), q)
                expected = [0] * d
                if i + j < d:
                    expected[i + j] = 1
                else:
                    expected[i + j - d] = q - 1  # i.e. -1 mod q
                assert result == expected, (
                    f"X^{i} * X^{j}: got {result}, expected {expected}"
                )


# --------------------------------------------------------------------------
# Sympy oracle-of-oracle: 100 random products at d=8, 20 at d=16
# --------------------------------------------------------------------------


def sympy_mul_reference(a: list[int], b: list[int], q: int) -> list[int]:
    """Independent negacyclic multiplication via sympy.

    Constructs the input polynomials as sympy expressions, multiplies them,
    reduces modulo ``X^d + 1`` using sympy's polynomial division, and packs
    the remainder back into our coefficient-list format. Used only by tests
    -- not a dependency of the runtime oracle.
    """
    d = len(a)
    x = sp.symbols("x")
    expr_a = sum((int(c) * x**i for i, c in enumerate(a)), sp.Integer(0))
    expr_b = sum((int(c) * x**i for i, c in enumerate(b)), sp.Integer(0))
    pa = sp.Poly(expr_a, x, domain=sp.ZZ)
    pb = sp.Poly(expr_b, x, domain=sp.ZZ)
    modulus = sp.Poly(x**d + 1, x, domain=sp.ZZ)
    rem = (pa * pb) % modulus
    result = [0] * d
    for monomial, coeff in rem.as_dict().items():
        deg = monomial[0] if monomial else 0
        if 0 <= deg < d:
            result[deg] = int(coeff) % q
    return result


class TestSympyOracle:
    def test_sympy_reference_self_check_at_d8(self):
        """Sanity-check the sympy reference against itself for a known case."""
        d, q = 8, 12289
        # X^7 * X = X^8 = -1
        assert sympy_mul_reference(x_pow(7, d), x_pow(1, d), q) == [q - 1] + [0] * 7

    def test_our_mul_matches_sympy_at_d8(self, rng):
        d, q = ORACLE_TINY.d, ORACLE_TINY.q
        for _ in range(100):
            a = rand_poly(rng, d, q)
            b = rand_poly(rng, d, q)
            assert mul(a, b, q) == sympy_mul_reference(a, b, q)

    def test_our_mul_matches_sympy_at_d16(self, rng):
        d, q = ORACLE_SMALL.d, ORACLE_SMALL.q
        for _ in range(20):
            a = rand_poly(rng, d, q)
            b = rand_poly(rng, d, q)
            assert mul(a, b, q) == sympy_mul_reference(a, b, q)

    def test_zero_input_matches_sympy(self, rng):
        d, q = 8, 12289
        zero = [0] * d
        a = rand_poly(rng, d, q)
        assert mul(a, zero, q) == sympy_mul_reference(a, zero, q)
        assert mul(zero, zero, q) == sympy_mul_reference(zero, zero, q)


# --------------------------------------------------------------------------
# Length validation
# --------------------------------------------------------------------------


class TestLengthValidation:
    def test_add_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            add([1, 2], [1, 2, 3], 17)

    def test_sub_rejects_length_mismatch(self):
        with pytest.raises(ValueError):
            sub([1, 2], [1, 2, 3], 17)

    def test_mul_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            mul([1, 2], [1, 2, 3], 17)
