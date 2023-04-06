"""Convergence tests for the stencils generated"""

import devito as dv
import numpy as np
import sympy as sp
import pytest

from test_geometry import read_sdf
from schism import BoundaryGeometry, BoundaryConditions, Boundary


def sinusoid(func, x, ppw, deriv=0):
    """
    Return a sinusoidal function to test against.

    Parameters
    ----------
    func : str
        'sin' or 'cos'
    x : ndarray
        Values to take the function of
    ppw : int
        Points per wavelength
    deriv : int
        Derivative to take
    """
    xind = sp.Symbol('x')
    if func == 'sin':
        expr = sp.sin(2*sp.pi*(xind-50.5)/ppw)
    elif func == 'cos':
        expr = sp.cos(2*sp.pi*(xind-50.5)/ppw)
    if deriv != 0:
        diff_expr = sp.diff(expr, *[xind for i in range(deriv)])
    else:  # Need to sidestep default behaviour to take single deriv
        diff_expr = expr

    lam = sp.lambdify(xind, diff_expr)
    return lam(x)


def diag_sinusoid(func, x, y, ppw, deriv=0):
    """
    Returns a diagonal sinusoid to test against. The place wave is aligned with
    the x=y line.

    Parameters
    ----------
    func : str
        'sin' or 'cos'
    x, y : ndarray
        Values to take the function of
    ppw : int
        Points per wavelength
    deriv : tuple
        Derivative to take
    """
    xind = sp.Symbol('x')
    yind = sp.Symbol('y')
    if func == 'sin':
        expr = sp.sin(2*sp.pi*(xind-yind)/ppw)
    elif func == 'cos':
        expr = sp.cos(2*sp.pi*(xind-yind)/ppw)
    if deriv != (0, 0):
        diff_expr = sp.diff(expr, *[xind for i in range(deriv[0])],
                            *[yind for i in range(deriv[1])])
    else:  # Need to sidestep default behaviour to take single deriv
        diff_expr = expr

    lam = sp.lambdify([xind, yind], diff_expr)
    return lam(x, y)


class TestScalar:
    """Simple tests based on a horizontal boundary"""

    @pytest.mark.parametrize('s_o', [2, 4, 6])
    def test_horizontal_convergence(self, s_o):
        """
        Convergence test for immersed boundary stencils at a horizontal surface
        0.5 dy above the last interior point.
        """
        # Load the flat 2D sdf
        sdf = read_sdf('horizontal', 2)
        # Create a geometry from it
        bg = BoundaryGeometry(sdf)
        grid = bg.grid

        f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
        # Deriv will be dy2
        deriv = (f.dy2,)  # Wants to be tuple

        # Pressure free-surface bcs
        if s_o == 2:
            bcs = BoundaryConditions([dv.Eq(f, 0),
                                      dv.Eq(f.dx2+f.dy2, 0)])
        elif s_o == 4:
            bcs = BoundaryConditions([dv.Eq(f, 0),
                                      dv.Eq(f.dx2+f.dy2, 0),
                                      dv.Eq(f.dx4 + 2*f.dx2dy2 + f.dy4, 0)])
        elif s_o == 6:
            bcs = BoundaryConditions([dv.Eq(f, 0),
                                      dv.Eq(f.dx2+f.dy2, 0),
                                      dv.Eq(f.dx4 + 2*f.dx2dy2 + f.dy4, 0),
                                      dv.Eq(f.dx6 + 3*f.dx4dy2
                                            + 3*f.dx2dy4 + f.dy6, 0)])

        boundary = Boundary(bcs, bg)
        subs = boundary.substitutions(deriv)

        # Fill f with sinusoid
        yinds = np.arange(grid.shape[-1])

        eq = dv.Eq(f.forward, subs[f.dy2])
        op = dv.Operator(eq)

        errs = []

        if s_o == 2 or s_o == 4:
            refinements = [1, 2, 4, 8, 16]
        elif s_o == 6:
            # Tighter range as you hit the noise floor otherwise
            refinements = [1, 2, 3, 4]
        for refinement in refinements:
            f.data[:] = 0  # Reset the data
            f.data[0] = sinusoid('sin', yinds, refinement*10, deriv=0)
            op.apply(time_M=1)

            # Scaling factor
            # (as I'm expanding the function, not shrinking the grid)
            err = f.data[-1]*10**2
            err -= sinusoid('sin', yinds, refinement*10, deriv=2)

            # Trim down to interior and exclude edges
            err_trimmed = err[s_o//2:-s_o//2, s_o//2:50]
            errs.append(float(np.amax(np.abs(err_trimmed))))

        grad = np.polyfit(np.log10(refinements), np.log10(errs), 1)[0]

        assert grad <= -s_o

    @pytest.mark.parametrize('s_o', [2, 4, 6])
    def test_x_deriv_convergence(self, s_o):
        """
        Convergence test for cross-derivatives featuring a diagonal
        boundary.
        """
        # Read the diagonal SDF
        sdf = read_sdf('45', 2)
        # Create a geometry from it
        bg = BoundaryGeometry(sdf)
        origin = (sp.core.numbers.Zero(), sp.core.numbers.Zero())
        interior_mask = bg.interior_mask[origin]
        interior_mask = interior_mask[s_o//2:-s_o//2, s_o//2:-s_o//2]

        grid = bg.grid

        f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
        # Deriv will be dy2
        deriv = (f.dxdy,)  # Wants to be tuple

        # Pressure free-surface bcs
        if s_o == 2:
            bcs = BoundaryConditions([dv.Eq(f, 0),
                                      dv.Eq(f.dx2+f.dy2, 0)])
        elif s_o == 4:
            bcs = BoundaryConditions([dv.Eq(f, 0),
                                      dv.Eq(f.dx2+f.dy2, 0),
                                      dv.Eq(f.dx4 + 2*f.dx2dy2 + f.dy4, 0)])
        elif s_o == 6:
            bcs = BoundaryConditions([dv.Eq(f, 0),
                                      dv.Eq(f.dx2+f.dy2, 0),
                                      dv.Eq(f.dx4 + 2*f.dx2dy2 + f.dy4, 0),
                                      dv.Eq(f.dx6 + 3*f.dx4dy2
                                            + 3*f.dx2dy4 + f.dy6, 0)])

        boundary = Boundary(bcs, bg)
        subs = boundary.substitutions(deriv)

        # Fill f with sinusoid
        xinds, yinds = np.meshgrid(np.arange(grid.shape[0]),
                                   np.arange(grid.shape[1]),
                                   indexing='ij')

        eq = dv.Eq(f.forward, subs[f.dxdy])
        op = dv.Operator(eq)

        errs = []

        if s_o == 2 or s_o == 4:
            refinements = [1, 2, 4, 8, 16]
        elif s_o == 6:
            # Tighter range as you hit the noise floor otherwise
            refinements = [1, 2, 3, 4]
        for refinement in refinements:
            f.data[:] = 0  # Reset the data
            f.data[0] = diag_sinusoid('sin', xinds, yinds,
                                      refinement*10, deriv=(0, 0))
            op.apply(time_M=0)

            # Scaling factor
            # (as I'm expanding the function, not shrinking the grid)
            err = f.data[-1]*10**2
            err -= diag_sinusoid('sin', xinds, yinds,
                                 refinement*10, deriv=(1, 1))

            # Trim off the edges
            err_trimmed = err[s_o//2:-s_o//2, s_o//2:-s_o//2]

            # Apply the interior mask
            err_trimmed = err_trimmed[interior_mask]
            errs.append(float(np.amax(np.abs(err_trimmed))))

        grad = np.polyfit(np.log10(refinements), np.log10(errs), 1)[0]

        assert grad <= -s_o


class TestVector:
    @pytest.mark.parametrize('s_o', [2, 4, 6])
    def test_velocity_convergence(self, s_o):
        """
        Convergence test for velocity stencils applied to a diagonal plane
        wave.
        """
        sdf = read_sdf('45', 2)
        sdf_x = read_sdf('45_x', 2)
        sdf_y = read_sdf('45_y', 2)

        sdfs = (sdf, sdf_x, sdf_y)

        # Create a geometry from it
        bg = BoundaryGeometry(sdfs)

        origin = (sp.core.numbers.Zero(), sp.core.numbers.Zero())
        interior_mask = bg.interior_mask[origin]
        interior_mask = interior_mask[1+s_o//2:-1-s_o//2, 1+s_o//2:-1-s_o//2]

        grid = bg.grid
        x, y = grid.dimensions

        v = dv.VectorTimeFunction(name='v', grid=grid, space_order=s_o)

        derivs = (v[0].dx(x0=x), v[1].dy(x0=y))

        # Velocity free-surface bcs
        if s_o == 2:
            bcs = BoundaryConditions([dv.Eq(v[0].dx + v[1].dy, 0)])
        elif s_o == 4:
            bcs = BoundaryConditions([dv.Eq(v[0].dx + v[1].dy, 0),
                                      dv.Eq(v[0].dx3 + v[1].dx2dy
                                            + v[0].dxdy2 + v[1].dy3, 0)])
        elif s_o == 6:
            bcs = BoundaryConditions([dv.Eq(v[0].dx + v[1].dy, 0),
                                      dv.Eq(v[0].dx3 + v[1].dx2dy
                                            + v[0].dxdy2 + v[1].dy3, 0),
                                      dv.Eq(v[0].dx5 + v[1].dx4dy
                                            + 2*v[0].dx3dy2 + 2*v[1].dx2dy3
                                            + v[0].dxdy4 + v[1].dy5)])

        boundary = Boundary(bcs, bg)
        subs = boundary.substitutions(derivs)

        eq_x = dv.Eq(v[0].forward, subs[v[0].dx(x0=x)])
        eq_y = dv.Eq(v[1].forward, subs[v[1].dy(x0=y)])
        # eq_x = dv.Eq(v[0].forward, v[0].dx(x0=x))
        # eq_y = dv.Eq(v[1].forward, v[1].dy(x0=y))
        op = dv.Operator([eq_x, eq_y])

        # Fill f with sinusoid
        xinds, yinds = np.meshgrid(np.arange(grid.shape[0]),
                                   np.arange(grid.shape[1]),
                                   indexing='ij')

        errs_div = []
        refinements = [1, 1.1, 1.2, 1.3, 1.4]

        for refinement in refinements:
            v[0].data[:] = 0  # Reset the data
            v[1].data[:] = 0
            v[0].data[0] = diag_sinusoid('cos', xinds+0.5, yinds,
                                         refinement*10, deriv=(0, 0))
            v[1].data[0] = -diag_sinusoid('cos', xinds, yinds+0.5,
                                          refinement*10, deriv=(0, 0))
            op.apply(time_M=1)

            # Scaling factor
            # (as I'm expanding the function, not shrinking the grid)
            # Check the error in the divergence, as this is what we will use
            err_div = v[0].data[-1]*10 + v[1].data[-1]*10
            err_div -= 2*diag_sinusoid('cos', xinds, yinds,
                                       refinement*10, deriv=(1, 0))

            # Trim off the edges
            err_div = err_div[1+s_o//2:-1-s_o//2, 1+s_o//2:-1-s_o//2]

            # Apply the interior mask
            err_div = err_div[interior_mask]

            errs_div.append(float(np.linalg.norm(err_div)))

        grad = np.polyfit(np.log10(refinements), np.log10(errs_div), 1)[0]

        assert grad <= -s_o
