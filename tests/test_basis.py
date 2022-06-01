"""Tests for the generation of basis functions"""

import pytest
import devito as dv
import sympy as sp

from schism.basic import Basis
from functools import reduce


class TestBasis:
    """Tests for the basis"""

    x = dv.SpaceDimension('x')
    y = dv.SpaceDimension('y')
    z = dv.SpaceDimension('z')

    @pytest.mark.parametrize('dims, order', [((x,), 2),
                                             ((y,), 2),
                                             ((x, y), 2),
                                             ((y, z), 2),
                                             ((x, y, z), 2),
                                             ((x,), 4),
                                             ((x, y), 4),
                                             ((y, z), 4),
                                             ((x, y, z), 4),
                                             ((x,), 6),
                                             ((x, y), 6),
                                             ((y, z), 6),
                                             ((x, y, z), 6),
                                             ((x, y, z), 8)])
    def test_expression_generation(self, dims, order):
        """Check that the symbolic expression is correctly generated"""
        basis = Basis('f', dims, order)
        for term in basis.terms:
            coeff = basis.expr.coeff(basis.d[term])
            test_coeff = reduce(lambda a, b: a*b,
                                [dims[i]**term[i]/sp.factorial(term[i])
                                 for i in range(len(dims))])
            assert coeff == test_coeff

    @pytest.mark.parametrize('dims, order, reduce',
                             [((x,), 8, 2),
                              ((x,), 8, 6),
                              ((x, y), 8, 2),
                              ((x, y), 8, 4),
                              ((x, y), 8, 8)])
    def test_reduce_order(self, dims, order, reduce):
        """Check that reducing the order works as expected"""
        basis = Basis('f', dims, order)
        check = Basis('f', dims, order-reduce)
        reduced = basis.reduce_order(reduce)
        print(reduced.expr - check.expr.subs(check.d, reduced.d))
        assert reduced.expr == check.expr.subs(check.d, reduced.d)
