"""Tests for misc FD tools"""

import pytest
import devito as dv
import numpy as np
import sympy as sp

from schism.finite_differences.tools import get_sten_vector


class TestStencilVector:
    """Tests for get_sten_vector"""
    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    x, y = grid.dimensions
    f2 = dv.TimeFunction(name='f', grid=grid, space_order=2)
    f4 = dv.TimeFunction(name='f', grid=grid, space_order=4)

    @pytest.mark.parametrize('deriv',
                             [f2.dx, f2.dy, f2.dx2, f4.dx,
                              f2.dx(x0=x+x.spacing/2),
                              f2.dy(x0=y-y.spacing/2)])
    def test_sten_vector_1D(self, deriv):
        """
        Check that the vector is correctly returned for derivatives in a
        single dimension
        """
        s_o = deriv.expr.space_order
        dims = deriv.expr.space_dimensions
        if deriv.dims[0] == dims[0]:
            points = (np.arange(-s_o//2, s_o//2+1), np.zeros(1+s_o, dtype=int))
        elif deriv.dims[0] == dims[1]:
            points = (np.zeros(1+s_o, dtype=int), np.arange(-s_o//2, s_o//2+1))
        result = get_sten_vector(deriv, points)
        if tuple(deriv.x0.keys()) == ():
            x0 = 0
        elif tuple(deriv.x0.keys()) == (dims[0],):
            x0 = 0.5  # Note this is because derivatives are known
            # Make sure this is updated if the test is ever modified
        else:
            x0 = -0.5

        # Compare against sympy weights
        deriv_order = deriv.deriv_order
        inds = np.arange(-s_o//2, s_o//2+1)
        check = np.array(sp.finite_diff_weights(deriv_order, inds, x0)[-1][-1],
                         dtype=float)

        assert np.all(np.isclose(result, check))

    @pytest.mark.parametrize('deriv', [f2.dxdy, f4.dxdy, f4.dx2dy2])
    def test_sten_vector_2D(self, deriv):
        """
        Check that the vector is correctly returned for derivatives with
        respect to two dimensions.
        """
        s_o = deriv.expr.space_order
        msh = np.meshgrid(*[np.arange(-s_o//2, s_o//2+1) for i in range(2)])
        points = tuple([m.flatten() for m in msh])

        result = get_sten_vector(deriv, points)

        deriv_order = deriv.deriv_order[0]
        inds = np.arange(-s_o//2, s_o//2+1)
        check1d = np.array(sp.finite_diff_weights(deriv_order,
                                                  inds, 0)[-1][-1],
                           dtype=float)

        check_msh = np.meshgrid(*[check1d for i in range(2)])
        check = check_msh[0]*check_msh[1]

        assert np.all(np.isclose(result, check.flatten()))
