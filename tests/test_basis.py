"""Tests for the generation of basis functions"""

import pytest
import devito as dv
import sympy as sp
import numpy as np

from schism.basic import Basis, row_from_expr
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


class TestRowFromExpression:
    """Tests for row_from_expr"""

    params = [(1, None, 0, 2), (1, None, 1, 2), (1, None, 0, 4),
              (2, None, 0, 2), (3, None, 0, 2), (2, 0, 0, 2),
              (2, 1, 0, 2), (3, 0, 0, 2)]

    @pytest.mark.parametrize('ndims, basis_dim, func_pos, s_o', params)
    def test_with_basis(self, ndims, basis_dim, func_pos, s_o):
        """
        Check that expressions consisting of a single basis function are
        correctly converted into a lambdified function, returning the correct
        values.
        """

        setup_str = "Ndims={}, Basis dim={}, Function position={}, order={}"
        print(setup_str.format(ndims, basis_dim, func_pos, s_o))

        # Create a grid with some dimensionality
        shape = tuple([11 for dim in range(ndims)])
        extent = tuple([10. for dim in range(ndims)])
        grid = dv.Grid(shape=shape, extent=extent)

        f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
        g = dv.TimeFunction(name='g', grid=grid, space_order=s_o)

        if func_pos == 0:  # Put f first
            funcs = (f, g)
        else:
            funcs = (g, f)

        if basis_dim is None:
            # Make a basis for f
            basisf = Basis('f', grid.dimensions, s_o)
            # Make a basis for g
            basisg = Basis('g', grid.dimensions, s_o)
        else:
            # Make a basis for f
            basisf = Basis('f', (grid.dimensions[basis_dim],), s_o)
            # Make a basis for g
            basisg = Basis('g', (grid.dimensions[basis_dim],), s_o)

        basis_map = {f: basisf, g: basisg}

        rowfunc = row_from_expr(basisf.expr, funcs, basis_map)

        # Need to generate some points to check the function at
        points = [np.arange(-1, 2) for dim in grid.dimensions]

        print(rowfunc(*points))

        assert False
