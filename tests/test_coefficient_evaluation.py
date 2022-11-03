"""
Tests to check that values of variable coefficients are correctly pulled off
the grid.
"""

import devito as dv
import sympy as sp
import numpy as np
import pytest

from schism.finite_differences.tools import extract_values


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
