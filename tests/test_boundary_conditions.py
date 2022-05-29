"""Tests for boundary conditions"""
import pytest
import devito as dv
import sympy as sp

from schism.conditions import BoundaryConditions
from collections import Counter


class TestBCs:
    """Tests for the BoundaryConditions object"""

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    f = dv.TimeFunction(name='f', grid=grid)
    v = dv.VectorTimeFunction(name='v', grid=grid)
    tau = dv.TensorTimeFunction(name='tau', grid=grid)

    @pytest.mark.parametrize('eqs, ans',
                             [([dv.Eq(f, 0), dv.Eq(f, 0)],
                               [dv.Eq(f, 0)]),
                              ([dv.Eq(dv.div(v), 0)],
                               [dv.Eq(v[0].dx + v[1].dy, 0)]),
                              ([dv.Eq(tau*v, sp.Matrix([0., 0.]))],
                               [dv.Eq(v[0]*tau[0, 0] + v[1]*tau[0, 1], 0),
                                dv.Eq(v[0]*tau[1, 0] + v[1]*tau[1, 1], 0)])])
    def test_eq_flattening(self, eqs, ans):
        """
        Check that equations are correctly flattened with duplicates removed.
        """
        bcs = BoundaryConditions(eqs)
        # Uses a counter as bc order can be inconsistent
        assert Counter(bcs.equations) == Counter(ans)
