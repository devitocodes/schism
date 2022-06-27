"""Convergence tests for the stencils generated"""

import devito as dv
import numpy as np
import sympy as sp

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


class TestHorizontalSetup:
    """Simple tests based on a horizontal boundary"""

    def test_horizontal_convergence(self):
        """
        Convergence test for immersed boundary stencils at a horizontal surface
        0.5 dy above the last interior point.
        """
        # Load the flat 2D sdf
        sdf = read_sdf('horizontal', 2)
        # Create a geometry from it
        bg = BoundaryGeometry(sdf)
        grid = bg.grid
        # Currently hardcoded to fourth order
        f = dv.TimeFunction(name='f', grid=grid, space_order=4)
        # Deriv will be dy2
        deriv = (f.dy2,)  # Wants to be tuple
        # Pressure free-surface bcs
        bcs = BoundaryConditions([dv.Eq(f, 0),
                                  dv.Eq(f.dx2+f.dx2, 0),
                                  dv.Eq(f.dx4 + 2*f.dx2dy2 + f.dy4, 0)])

        boundary = Boundary(bcs, bg)
        subs = boundary.substitutions(deriv)

        # Fill f with sinusoid
        yinds = np.arange(grid.shape[-1])

        eq = dv.Eq(f.forward, subs[f.dy2])
        op = dv.Operator(eq)

        errs = []
        refinements = [1, 2, 4, 8, 16]
        for refinement in refinements:
            f.data[:] = 0  # Reset the data
            f.data[0] = sinusoid('sin', yinds, refinement*10, deriv=0)
            op.apply(time_M=1)

            # Scaling factor
            # (as I'm expanding the function, not shrinking the grid)
            err = f.data[-1]*10**2
            err -= sinusoid('sin', yinds, refinement*10, deriv=2)
            # Trim down to interior and exclude edges
            err_trimmed = err[2:-2, 2:50]
            errs.append(float(np.amax(np.abs(err_trimmed))))

        grad = np.polyfit(np.log10(refinements), np.log10(errs), 1)[0]
        assert grad <= -4
