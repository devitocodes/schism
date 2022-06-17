"""Tests for the generation of basis functions"""

import pytest
import os
import devito as dv
import sympy as sp
import numpy as np

from schism.basic import Basis, row_from_expr
from schism.conditions.boundary_conditions import SingleCondition
from functools import reduce
from itertools import product


def corner_points(ndims, basis_dim, size):
    """
    Returns points at the corners of some line/square/cube of a specified
    size.
    """
    if basis_dim is None:
        combinations = product([-size, size], repeat=ndims)
        points = np.array([c for c in combinations])
        return tuple([points[:, dim] for dim in range(ndims)])
    else:
        return tuple([np.array([-size, size]) if dim == basis_dim
                      else np.zeros(2) for dim in range(ndims)])


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
        substitution = [(check.d[term], reduced.d[term])
                        for term in check.terms]
        assert reduced.expr == check.expr.subs(substitution)


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
        points = corner_points(ndims, basis_dim, 2)

        rows = rowfunc(*points)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/basis_test_results/row_from_expr/' \
            + str(ndims) + str(basis_dim) + str(func_pos) + str(s_o) + '.npy'

        answer = np.load(fname)  # Load the answer

        # Check against answer
        assert np.all(np.isclose(rows, answer))

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    x, y = grid.dimensions
    f = dv.TimeFunction(name='f', grid=grid, space_order=2)
    v = dv.VectorTimeFunction(name='v', grid=grid, space_order=2)

    basisf2D = Basis('f_2d', grid.dimensions, f.space_order)
    basisfx = Basis('f_x', (x,), f.space_order)
    basisfy = Basis('f_y', (y,), f.space_order)
    basisvx = Basis('vx', grid.dimensions, v[0].space_order)
    basisvy = Basis('vy', grid.dimensions, v[1].space_order)

    @pytest.mark.parametrize('bc, basis_map, ans',
                             [(dv.Eq(f, 0), {f: basisf2D},
                               [1., 0.8, 0.32, 0.6, 0.48, 0.18]),
                              (dv.Eq(f.laplace, 0), {f: basisf2D},
                               [0., 0., 1., 0., 0., 1.]),
                              (dv.Eq(dv.div(v), 0),
                              {v[0]: basisvx, v[1]: basisvy},
                               [0., 0., 0., 1., 0.8, 0.6,
                                0., 1., 0.8, 0., 0.6, 0.]),
                              (dv.Eq(f, 0), {f: basisfx}, [1., 0.6, 0.18]),
                              (dv.Eq(f, 0), {f: basisfy}, [1., 0.8, 0.32]),
                              (dv.Eq(f.dx2, 0), {f: basisfx}, [0., 0., 1.]),
                              (dv.Eq(f.dy2, 0), {f: basisfy}, [0., 0., 1.])])
    def test_with_bc_expressions(self, bc, basis_map, ans):
        """
        Check that expressions generated by substituting basis functions into
        boundary conditions are correctly turned into row functions
        """
        condition = SingleCondition(bc)
        expr = condition.sub_basis(basis_map)
        funcs = tuple(basis_map.keys())
        rowfunc = row_from_expr(expr, funcs, basis_map)

        point = (0.6, 0.8)
        assert np.all(np.isclose(rowfunc(*point), ans))
