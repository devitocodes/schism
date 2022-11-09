"""
Tests to check that values of variable coefficients are correctly pulled off
the grid.
"""

import os
import devito as dv
import sympy as sp
import numpy as np
import pytest

from schism.finite_differences.tools import extract_values
from schism import BoundaryGeometry, BoundaryConditions, Boundary


def derivative_error_w_normals(res):
    """
    Get the error in estimated derivatives calculated with stencils based off
    of a BC containing the normal.
    """
    x = np.linspace(-5, 5, res)
    y = np.linspace(0, 10, res)

    xmsh, ymsh = np.meshgrid(x, y, indexing='ij')

    grid = dv.Grid(shape=(res, res), extent=(10., 10.))
    sdf = dv.Function(name='sdf', grid=grid, space_order=2)

    # Read in SDF data and flip interior/exterior
    path = os.path.dirname(os.path.abspath(__file__))
    infile = path + '/sdfs/convergence_test_parabola_' + str(res) + '.npy'
    sdf.data[:] = -np.load(infile)

    # Create and fill function data
    f = dv.Function(name='f', grid=grid, space_order=2)
    f.data[:] = -5*ymsh+0.5*xmsh**2+ymsh**2

    # Functions to store derivatives in
    dfdx = dv.Function(name='dfdx', grid=grid, space_order=2)
    dfdy = dv.Function(name='dfdy', grid=grid, space_order=2)

    # Set up the boundary
    bg = BoundaryGeometry(sdf)

    # Boundary condition is zero derivative in the normal direction
    bc_list = [dv.Eq(bg.n.dot(dv.grad(f)), 0)]

    bcs = BoundaryConditions(bc_list, funcs=(f,))

    boundary = Boundary(bcs, bg)

    derivs = (f.dx, f.dy)
    subs = boundary.substitutions(derivs)

    eq_dfdx = dv.Eq(dfdx, subs[f.dx])
    eq_dfdy = dv.Eq(dfdy, subs[f.dy])

    dv.Operator([eq_dfdx, eq_dfdy])()

    # Need to trim the data to remove the edges where stencils go out of
    # bounds
    dfdx_check = xmsh
    dfdy_check = -5 + 2*ymsh

    mask = np.logical_and(np.abs(dfdx.data) <= np.finfo(np.float32).eps,
                          np.abs(dfdy.data) <= np.finfo(np.float32).eps)
    dfdx_check[mask] = 0
    dfdy_check[mask] = 0

    dfdx_err = np.amax(np.abs(dfdx.data[1:-1, 1:-1]
                              - dfdx_check[1:-1, 1:-1]))
    dfdy_err = np.amax(np.abs(dfdy.data[1:-1, 1:-1]
                              - dfdy_check[1:-1, 1:-1]))

    return dfdx_err, dfdy_err


class TestVariableCoefficient:
    """Tests for variable coefficient evaluation"""
    @pytest.mark.parametrize('ndims', [2, 3])
    def test_coefficient_evaluation(self, ndims):
        """Check that values are correctly pulled off a grid"""

        grid = dv.Grid(shape=tuple([11 for d in range(ndims)]),
                       extent=tuple([100. for d in range(ndims)]))
        f = dv.Function(name='f', grid=grid)
        g = dv.Function(name='g', grid=grid)
        # Crosswise gradient fills
        f.data[:] = np.linspace(0., 10., 11)
        g.data[:] = np.linspace(0., 10., 11)[:, np.newaxis]

        coeff_0, coeff_1, coeff_2 = sp.symbols('coeff_0, coeff_1, coeff_2')
        symbols = (coeff_0, coeff_1, coeff_2)
        expr_map = {coeff_0: f+1, coeff_1: g+1, coeff_2: f+2*g+3}

        indices = tuple([np.random.randint(1, high=9, size=10)
                         for d in range(ndims)])
        points = tuple([10.*inds for inds in indices])

        vals = extract_values(grid, symbols, expr_map, points)

        # Checks calculated with NumPy
        check_0 = f.data[indices] + 1.
        check_1 = g.data[indices] + 1.
        check_2 = f.data[indices] + 2.*g.data[indices] + 3.

        assert np.all(np.isclose(check_0, vals[0]))
        assert np.all(np.isclose(check_1, vals[1]))
        assert np.all(np.isclose(check_2, vals[2]))


class TestFullStack:
    """Tests for the full stack with coefficient evaluation"""
    def test_convergence(self):
        """
        Check that stencils based on variable coefficients are correctly
        constructed and converge as expected.
        """
        resolutions = [51, 61, 71, 81, 91, 101]
        dfdx_errs = []
        dfdy_errs = []
        for res in resolutions:
            x_err, y_err = derivative_error_w_normals(res)
            dfdx_errs.append(x_err)
            dfdy_errs.append(y_err)

        spacings = [10./(res-1.) for res in resolutions]
        x_gradient = np.polyfit(np.log(spacings), np.log(dfdx_errs), 1)
        y_gradient = np.polyfit(np.log(spacings), np.log(dfdy_errs), 1)

        assert x_gradient[0] > 1.65
        assert y_gradient[0] > 1.65
