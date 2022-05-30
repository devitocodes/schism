"""Tests for boundary conditions"""
import pytest
import devito as dv
import sympy as sp

from schism.conditions import BoundaryCondition, BoundaryConditions
from collections import Counter


class TestBCs:
    """Tests for the BoundaryConditions object"""

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    f = dv.TimeFunction(name='f', grid=grid, space_order=2)
    v = dv.VectorTimeFunction(name='v', grid=grid)
    tau = dv.TensorTimeFunction(name='tau', grid=grid)

    @pytest.mark.parametrize('eqs, ans',
                             [([dv.Eq(f, 0), dv.Eq(f, 0)],
                               [dv.Eq(f, 0)]),
                              ([dv.Eq(dv.div(v), 0)],
                               [dv.Eq(v[0].dx + v[1].dy, 0)]),
                              ([dv.Eq(tau*v, sp.Matrix([0., 0.]))],
                               [dv.Eq(v[0]*tau[0, 0] + v[1]*tau[0, 1], 0),
                                dv.Eq(v[0]*tau[1, 0] + v[1]*tau[1, 1], 0)]),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0)],
                               [dv.Eq(f, 0), dv.Eq(f.laplace, 0)]),
                              ([dv.Eq(f, 0), dv.Eq(f, 0),
                                dv.Eq(f.dx2, 0), dv.Eq(f.dy2, 0)],
                               [dv.Eq(f, 0), dv.Eq(f.dx2, 0),
                                dv.Eq(f.dy2, 0)])])
    def test_eq_flattening(self, eqs, ans):
        """
        Check that equations are correctly flattened with duplicates removed.
        """
        bcs = BoundaryConditions(eqs)
        # Uses a counter as bc order can be inconsistent
        assert Counter(bcs.equations) == Counter(ans)


class TestBC:
    """Tests for the BoundaryCondition object"""

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    f = dv.TimeFunction(name='f', grid=grid, space_order=2)
    g = dv.Function(name='g', grid=grid, space_order=2)
    v = dv.VectorTimeFunction(name='v', grid=grid)
    tau = dv.TensorTimeFunction(name='tau', grid=grid)

    @pytest.mark.parametrize('bc, funcs, ans',
                             [(dv.Eq(f, 0), None, (f,)),
                              (dv.Eq(dv.div(v), 0), None, (v[0], v[1])),
                              (dv.Eq(v[0]*tau[0, 0] + v[1]*tau[0, 1], 0),
                               None,
                               (v[0], v[1], tau[0, 0], tau[0, 1])),
                              (dv.Eq(v[0]*tau[0, 0] + v[1]*tau[0, 1], 0),
                               (tau[0, 0], tau[0, 1]),
                               (tau[0, 0], tau[0, 1])),
                              (dv.Eq(v[0]*tau[0, 0] + v[1]*tau[0, 1], 0),
                               (f, tau[0, 0], tau[0, 1]),
                               (tau[0, 0], tau[0, 1])),
                              (dv.Eq(f+g, 0), None, (f,)),
                              (dv.Eq(f+g, 0), (f, g), (f, g)),
                              (dv.Eq(dv.div(v), 0),
                               (v[0], v[1], f, g),
                               (v[0], v[1])),
                              (dv.Eq(f.laplace, 0), None, (f,)),
                              (dv.Eq(3*f+2*f.dx+f.dx2+2*f.dy+f.dy2, 0),
                               None,
                               (f,)),
                              (dv.Eq(3*f+2*f.dx+f.dx2+2*f.dy+f.dy2, 0),
                               (f, g),
                               (f,))])
    def test_function_id(self, bc, funcs, ans):
        """
        Check that functions within the boundary condition are correctly
        identified.
        """
        condition = BoundaryCondition(bc, funcs=funcs)

        assert Counter(condition.functions) == Counter(ans)
