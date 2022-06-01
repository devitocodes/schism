"""Tests for boundary conditions"""
import pytest
import devito as dv
import sympy as sp

from schism import BoundaryConditions
from schism.conditions.boundary_conditions import SingleCondition
from collections import Counter
from itertools import combinations


class TestBCs:
    """Tests for the BoundaryConditions object"""

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    f = dv.TimeFunction(name='f', grid=grid, space_order=4)
    v = dv.VectorTimeFunction(name='v', grid=grid, space_order=4)
    tau = dv.TensorTimeFunction(name='tau', grid=grid, space_order=4)

    @pytest.mark.parametrize('funcs, ans',
                             [((v,), (v[0], v[1])),
                              ((tau,), (tau[0, 0],
                                        tau[1, 0], tau[1, 1])),
                              ((v, tau), (v[0], v[1], tau[0, 0],
                                          tau[1, 0], tau[1, 1])),
                              ((v, f), (v[0], v[1], f))])
    def test_function_flattening(self, funcs, ans):
        """Check that functions supplied by user are correctly flattened"""
        grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
        f = dv.TimeFunction(name='f', grid=grid, space_order=2)
        bcs = BoundaryConditions([dv.Eq(f, 0)], funcs=funcs)
        # Uses a counter as unordered sets used
        assert Counter(bcs.funcs) == Counter(ans)

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

    @pytest.mark.parametrize('eqs, funcs, ans',
                             [([dv.Eq(f, 0), dv.Eq(f.laplace, 0)],
                               None,
                               (SingleCondition(dv.Eq(f, 0)),
                                SingleCondition(dv.Eq(f.laplace, 0)))),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                                dv.Eq(dv.div(v), 0)],
                               None,
                               (SingleCondition(dv.Eq(f, 0)),
                                SingleCondition(dv.Eq(f.laplace, 0)),
                                SingleCondition(dv.Eq(dv.div(v), 0)))),
                              ([dv.Eq(tau*v, sp.Matrix([0., 0.]))],
                               (tau,),
                               (SingleCondition(dv.Eq(v[0]*tau[0, 0]
                                                  + v[1]*tau[0, 1], 0),
                                                  funcs=(tau[0, 0], tau[0, 1],
                                                         tau[1, 1])),
                                SingleCondition(dv.Eq(v[0]*tau[1, 0]
                                                  + v[1]*tau[1, 1], 0),
                                                  funcs=(tau[0, 0], tau[0, 1],
                                                         tau[1, 1]))))])
    def test_bc_setup(self, eqs, funcs, ans):
        """Check that BCs are correctly set up"""
        bcs = BoundaryConditions(eqs, funcs=funcs)
        # Need to assert that both match -> goes both ways
        for bc in bcs.bcs:
            assert bc in ans
        for a in ans:
            assert a in bcs.bcs

    @pytest.mark.parametrize('eqs, funcs, same, sep',
                             [([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                               dv.Eq(dv.div(v), 0)], None,
                               (v[0], v[1]), (v[0], f)),
                              ([dv.Eq(tau*v, sp.Matrix([0., 0.]))], (tau,),
                               (tau[0, 0], tau[0, 1], tau[1, 1]), ()),
                              ([dv.Eq(f, 0), dv.Eq(f.dx2+f.dy2, 0),
                                dv.Eq(f.dx4+2*f.dx2dy2+f.dy4, 0),
                                dv.Eq(v[0].dx+v[1].dy, 0),
                                dv.Eq(v[0].dx3+v[1].dx2dy
                                      + v[0].dxdy2+v[1].dx3, 0)], None,
                               (v[0], v[1]), (v[0], f))])
    def test_condition_grouping(self, eqs, funcs, same, sep):
        """Check that BCs are correctly grouped"""
        # same = functions that should be together
        # sep = functions that should be separate
        bcs = BoundaryConditions(eqs, funcs=funcs)
        for a, b in combinations(same, 2):
            assert bcs.get_group(a) is bcs.get_group(b)
        for a, b in combinations(sep, 2):
            assert bcs.get_group(a) is not bcs.get_group(b)


class TestBC:
    """Tests for the SingleCondition object"""

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    x, y = grid.dimensions
    f = dv.TimeFunction(name='f', grid=grid, space_order=2)
    g = dv.Function(name='g', grid=grid, space_order=2)
    h = dv.TimeFunction(name='h', grid=grid, space_order=2)
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
        condition = SingleCondition(bc, funcs=funcs)

        assert Counter(condition.funcs) == Counter(ans)

    @pytest.mark.parametrize('bc, ans',
                             [(dv.Eq(f, 0), None),
                              (dv.Eq(f+h, 0), None),
                              (dv.Eq(f+2*f.dx+3*f.dx2, 0), (x,)),
                              (dv.Eq(f.dx+h.dx, 0), (x,)),
                              (dv.Eq(f.laplace, 0), (x, y)),
                              (dv.Eq(f.laplace+h.laplace, 0), (x, y)),
                              (dv.Eq(f.dxdy, 0), (x, y)),
                              (dv.Eq(f+h.dxdy, 0), (x, y))])
    def test_derivative_dims(self, bc, ans):
        """Check that derivative directions correctly identified"""
        condition = SingleCondition(bc)

        assert Counter(condition.dims) == Counter(ans)
