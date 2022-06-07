"""Tests for the Boundary object and its components"""

import pytest
import devito as dv
import sympy as sp

from schism import BoundaryConditions, Boundary
from collections import Counter


class TestSubstitutions:
    """Tests for components of Boundary.substitutions()"""
    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    f = dv.TimeFunction(name='f', grid=grid, space_order=2)
    v = dv.VectorTimeFunction(name='v', grid=grid, space_order=2)
    tau = dv.TensorTimeFunction(name='tau', grid=grid, space_order=2)

    @pytest.mark.parametrize('bcs, has_1D_basis, deriv, funcs',
                             [([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                               dv.Eq(dv.div(v), 0)], False, f.dx, None),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                                dv.Eq(dv.div(v), 0)], False, f.dy, None),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                                dv.Eq(dv.div(v), 0)], False, f.dx2, None),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                                dv.Eq(dv.div(v), 0)], False, f.dxdy, None),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                                dv.Eq(dv.div(v), 0)], False, v[0].dx, None),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0),
                                dv.Eq(dv.div(v), 0)], False, v[1].dy, None),
                              ([dv.Eq(f, 0), dv.Eq(f.dx2, 0)],
                               True, f.dx, None),
                              ([dv.Eq(f, 0), dv.Eq(f.dx2, 0)],
                               True, f.dx2, None),
                              ([dv.Eq(f, 0), dv.Eq(f.dy2, 0)],
                               True, f.dy, None),
                              ([dv.Eq(tau*v, sp.Matrix([0., 0.]))],
                               False, tau[0, 0].dx, (tau,))])
    def test_basis_map(self, bcs, has_1D_basis, deriv, funcs):
        """
        Check that _get_basis_map() returns a mapping containing the correct
        basis functions.
        """
        # Create BoundaryConditions
        bcs = BoundaryConditions(bcs, funcs)
        # Create a Boundary (use a dummy BoundaryGeometry)
        boundary = Boundary(bcs, [], has_1D_basis=has_1D_basis)
        # Get the group
        group = boundary._get_filtered_group(deriv)
        # Get the basis map
        basis_map = boundary._get_basis_map(deriv, group)
        print(basis_map)
        # Check that all functions are present
        assert Counter(group.funcs) == Counter(basis_map.keys())
        # Check that each basis function maps between the expected dimensions
        # and order
        for func in basis_map:
            assert basis_map[func].order == func.space_order
            if has_1D_basis:
                assert basis_map[func].dims == deriv.dims
            else:
                assert basis_map[func].dims == func.space_dimensions

    def test_filtered_group(self):
        """
        Check that _get_filtered_group() returns the correctly filtered group.
        """
        return 0  # Placeholder
