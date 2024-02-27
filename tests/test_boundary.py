"""Tests for the Boundary object and its components"""

import pytest
import devito as dv
import sympy as sp
import numpy as np

from test_geometry import read_sdf
from schism import BoundaryConditions, Boundary, BoundaryGeometry
from collections import Counter
from devito.tools.utils import filter_sorted


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

    @pytest.mark.parametrize('bcs, has_1D_basis, deriv',
                             [([dv.Eq(f, 0), dv.Eq(f.dx2, 0), dv.Eq(f.dy2, 0)],
                               True, f.dx2),
                              ([dv.Eq(f, 0), dv.Eq(f.dx2, 0), dv.Eq(f.dy2, 0)],
                               True, f.dy2),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0)],
                               False, f.dx2),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0)],
                               False, f.dx2),
                              ([dv.Eq(f, 0), dv.Eq(f.laplace, 0)],
                               False, f.dy2)])
    def test_filtered_group(self, bcs, has_1D_basis, deriv):
        """
        Check that _get_filtered_group() returns the correctly filtered group.
        """
        # Create BoundaryConditions
        bcs = BoundaryConditions(bcs)
        # Create a Boundary (use a dummy BoundaryGeometry)
        boundary = Boundary(bcs, [], has_1D_basis=has_1D_basis)
        # Get the group
        group = boundary._get_filtered_group(deriv)
        unfiltered_group = boundary.conditions.get_group(deriv.expr)
        filtered_group = unfiltered_group.filter(deriv.dims[0])

        # Check that setups with 1D basis produce filtered results
        if has_1D_basis:
            assert group.conditions != unfiltered_group.conditions
            assert group.conditions == filtered_group.conditions
        # Check that setups with ND basis produce unfiltered results
        else:
            assert group.conditions != filtered_group.conditions
            assert group.conditions == unfiltered_group.conditions

    @pytest.mark.parametrize('s_o', [2, 4, 6])
    def test_forward_timestep(self, s_o):
        """
        Check that stencils for the forward timestep are the same as those for
        the current timestep.
        """

        # Load the flat 2D sdf
        sdf = read_sdf('horizontal', 2)
        # Create a geometry from it
        bg = BoundaryGeometry(sdf)
        grid = bg.grid

        f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
        # Compare second y deriv stencil at forward and current timestep
        derivs = (f.dy2, f.forward.dy2)

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
        subs = boundary.substitutions(derivs)

        t = f.time_dim

        current = subs[f.dy2].subs(t, t+t.spacing)
        forward = subs[f.forward.dy2]

        assert str(current) == str(forward)

        current_weights = dv.symbolics.retrieve_functions(current)
        forward_weights = dv.symbolics.retrieve_functions(forward)

        # Sort to ensure that weights are in the same order
        current_weights = filter_sorted(current_weights)
        forward_weights = filter_sorted(forward_weights)

        for crt, fwd in zip(current_weights, forward_weights):
            assert np.all(np.isclose(crt.data, fwd.data))
