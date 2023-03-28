"""Tests for the Substitution object"""

import pytest
import devito as dv
import numpy as np
import sympy as sp
import os

from test_interpolation import setup_geom, setup_f, DummyGroup, \
    DummyGeometry, DummySkin
from test_geometry import read_sdf
from schism.geometry.skin import ModifiedSkin
from schism.basic.basis import Basis
from schism.conditions.boundary_conditions import SingleCondition
from schism.finite_differences.substitution import Substitution
from schism import BoundaryGeometry, BoundaryConditions


def weights_test_setup(use_x_derivs=False):
    """Perform setup for the weight tests"""
    # Load the flat 2D sdf
    sdf = read_sdf('horizontal', 2)
    # Create a geometry from it
    bg = BoundaryGeometry(sdf)
    grid = bg.grid
    f = dv.TimeFunction(name='f', grid=grid, space_order=4)
    if use_x_derivs:
        # Use a cross-derivative
        deriv = f.dxdy
    else:
        # Use 2nd derivative wrt y
        deriv = f.dy2
    # Pressure free-surface bcs
    bcs = BoundaryConditions([dv.Eq(f, 0),
                              dv.Eq(f.dx2+f.dx2, 0),
                              dv.Eq(f.dx4 + 2*f.dx2dy2 + f.dy4, 0)])
    group = bcs.get_group(f)
    # Create a skin from that
    skin = ModifiedSkin(deriv, bg)
    # Create a basis map
    basis_map = {f: Basis(name=f.name,
                          dims=f.space_dimensions,
                          order=f.space_order)}

    # Create the Substitution
    subs = Substitution(deriv, group, basis_map, 'expand', skin)
    return subs


class TestSubstitution:
    """Tests for the Substitution object"""

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    @pytest.mark.parametrize('deriv_type', ['dx', 'dy2', 'dxf'])
    def test_get_stencils_expand(self, setup, func_type, deriv_type):
        """
        Check that _get_stencils() obtains correct stencils for all points.
        Note that this test is for the 'expand' strategy only.
        """
        grid = dv.Grid(shape=(3, 3), extent=(10., 10.))
        x, y = grid.dimensions
        geom, skin = setup_geom(setup, grid)
        f, group = setup_f(func_type, grid)

        if deriv_type == 'dx':
            deriv = group.funcs[0].dx
        elif deriv_type == 'dy2':
            deriv = group.funcs[0].dy2
        elif deriv_type == 'dxf':
            deriv = group.funcs[0].dx(x0=x+x.spacing/2)

        basis_map = {func: Basis(func.name, grid.dimensions, func.space_order)
                     for func in group.funcs}

        substitution = Substitution(deriv, group, basis_map, 'expand', skin)

        interpolants = substitution.interpolants.interpolants
        projections = substitution.projections.projections

        # Check that all points have stencils generated for them
        points = sum([i.pinv.shape[0] for i in interpolants])
        assert points == skin.npts

        # Check support region is incrementally increased
        for func in group.funcs:
            radii = [i.support.radius_map[func] for i in interpolants]
            check = func.space_order//2 + np.arange(len(radii))
            assert np.all(radii == check)

        # Check interpolants have matching projections
        assert len(interpolants) == len(projections)

    @pytest.mark.parametrize('deriv_type', ['dx', 'dy2'])
    def test_get_stencils_reduce(self, deriv_type):
        """
        Check that _get_stencils() obtains correct stencils for all points.
        Note that this test is for the 'reduce' strategy only.
        """
        grid = dv.Grid(shape=(5, 4), extent=(6., 5.))
        x, y = grid.dimensions
        origin = (sp.core.numbers.Zero(), sp.core.numbers.Zero())

        int_points = (np.array([2, 1, 2, 3, 0, 1, 2, 3, 4]),
                      np.array([2, 1, 1, 1, 0, 0, 0, 0, 0]))
        interior_mask = {origin: np.zeros(grid.shape, dtype=bool)}
        interior_mask[origin][int_points] = True

        bou_points = (np.array([1, 2, 3, 0, 1, 3, 4, 0, 4]),
                      np.array([3, 3, 3, 2, 2, 2, 2, 1, 1]))
        boundary_mask = np.zeros(grid.shape, dtype=bool)
        boundary_mask[bou_points] = True

        bou_pts_x = (np.array([1, 3, 0, 4]),
                     np.array([2, 2, 1, 1]))
        bou_msk_x = np.zeros(grid.shape, dtype=bool)
        bou_msk_x[bou_pts_x] = True
        bou_pts_y = (np.array([2, 1, 3, 0, 4]),
                     np.array([3, 2, 2, 1, 1]))
        bou_msk_y = np.zeros(grid.shape, dtype=bool)
        bou_msk_y[bou_pts_y] = True
        b_mask_1D = (bou_msk_x, bou_msk_y)

        r2 = np.sqrt(2)/2
        pos = (np.array([r2, 0, -r2, r2, -r2, r2, -r2, -r2, r2]),
               np.array([-r2, 0.5, -r2, -r2, r2, r2, -r2, r2, r2]))
        dense_pos = [np.zeros(grid.shape) for dim in range(2)]
        for dim in range(2):
            dense_pos[dim][bou_points] = pos[dim]
        geom = DummyGeometry(grid=grid, interior_mask=interior_mask,
                             boundary_mask=boundary_mask, dense_pos=dense_pos,
                             b_mask_1D=b_mask_1D)

        f = dv.TimeFunction(name='f', grid=grid, space_order=4)
        if deriv_type == 'dx':
            deriv = f.dx
            conditions = (SingleCondition(dv.Eq(f, 0)),)

            points = (np.array([2, 1, 2, 3]),
                      np.array([2, 1, 1, 1]))
            skin = DummySkin(geometry=geom, points=points)
        elif deriv_type == 'dy2':
            deriv = f.dy2
            conditions = (SingleCondition(dv.Eq(f, 0)),
                          SingleCondition(dv.Eq(f.dy2, 0)),
                          SingleCondition(dv.Eq(f.dy4, 0)))
            points = (np.array([2, 1, 2, 3, 0, 1, 3, 4]),
                      np.array([2, 1, 1, 1, 0, 0, 0, 0]))
            skin = DummySkin(geometry=geom, points=points)

        group = DummyGroup(funcs=(f,), conditions=conditions)

        basis_map = {func: Basis(func.name, deriv.dims, func.space_order)
                     for func in group.funcs}

        substitution = Substitution(deriv, group, basis_map, 'reduce', skin)

        interpolants = substitution.interpolants.interpolants

        # Check that all points have stencils generated for them
        points = sum([i.pinv.shape[0] for i in interpolants])
        assert points == skin.npts

        # Check order is incrementally reduced for the dx case
        if deriv_type == 'dx':
            assert len(interpolants) == 2  # Reduced once
            assert interpolants[0].pinv.shape == (3, 5, 10)
            assert interpolants[1].pinv.shape == (1, 3, 10)
            # Check order is never reduced in the dy2 case
        elif deriv_type == 'dy2':
            assert len(interpolants) == 1  # Never reduced
            assert interpolants[0].pinv.shape == (8, 5, 20)

    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    @pytest.mark.parametrize('deriv_type', ['dx', 'dy2', 'dxf'])
    def test_create_weight_functions(self, func_type, deriv_type):
        """
        Check that the weight functions are correctly generated for a given
        RHS vector.
        """
        grid = dv.Grid(shape=(3, 3), extent=(10., 10.))
        x, y = grid.dimensions
        geom, skin = setup_geom(0, grid)
        f, group = setup_f(func_type, grid)

        if deriv_type == 'dx':
            deriv = group.funcs[0].dx
        elif deriv_type == 'dy2':
            deriv = group.funcs[0].dy2
        elif deriv_type == 'dxf':
            deriv = group.funcs[0].dx(x0=x+x.spacing/2)

        basis_map = {func: Basis(func.name, grid.dimensions, func.space_order)
                     for func in group.funcs}

        substitution = Substitution(deriv, group, basis_map, 'expand', skin)

        wnames = [f.name for f in substitution.wfuncs]
        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/substitution_test_results/' \
            + 'create_weight_function/' + func_type + deriv_type + '.dat'

        with open(fname, 'r') as f:
            names = f.read()
            check = names.split(',')[:-1]  # Reads in extra comma

        assert wnames == check

    def test_unique_weight_names(self):
        """Check that generated weight functions have unique names"""
        grid = dv.Grid(shape=(3, 3), extent=(10., 10.))
        x, y = grid.dimensions
        geom, skin = setup_geom(0, grid)
        f, group = setup_f('vector', grid)
        basis_map = {func: Basis(func.name, grid.dimensions, func.space_order)
                     for func in group.funcs}

        subs_x = Substitution(group.funcs[0].dx, group, basis_map,
                              'expand', skin)
        subs_y = Substitution(group.funcs[1].dy, group, basis_map,
                              'expand', skin)

        wnames_x = [f.name for f in subs_x.wfuncs]
        wnames_y = [f.name for f in subs_y.wfuncs]

        # These two sets of names should have no overlap (no reused names)

        assert len(set(wnames_x).intersection(wnames_y)) == 0

    def test_fill_weights_coverage(self):
        """Check that the stencils get filled everywhere"""
        subs = weights_test_setup()
        bg = subs.geometry

        # Check that the weighting of the centre stencil point is nonzero
        # throughout the interior
        # Also assert that it is zeroed on the exterior
        feps = np.finfo(float).eps
        for term in subs.weight_map:
            func = subs.weight_map[term]
            if func.name == 'w_f_0_0':
                assert np.all(np.abs(func.data[bg.interior_mask]) >= feps)
                not_interior = np.logical_not(bg.interior_mask)
                assert np.all(np.abs(func.data[not_interior]) <= feps)

    def test_fill_weights_consistency(self):
        """Check that stencils are filled consistently"""
        subs = weights_test_setup()

        feps = np.finfo(float).eps
        for term in subs.weight_map:
            func = subs.weight_map[term]
            maxvals = np.amax(func.data[2:-2], axis=0)
            minvals = np.amin(func.data[2:-2], axis=0)
            assert np.all(np.abs(maxvals - minvals) <= feps)

    def test_returned_expr(self):
        """Check that the correct expression is returned"""
        subs = weights_test_setup()
        ans = 'f(t, x, y)*w_f_f_dy2_0_0(x, y) ' \
            + '+ f(t, x, y - 2*h_y)*w_f_f_dy2_0_m2(x, y) ' \
            + '+ f(t, x, y - h_y)*w_f_f_dy2_0_m1(x, y) ' \
            + '+ f(t, x, y + h_y)*w_f_f_dy2_0_1(x, y) ' \
            + '+ f(t, x, y + 2*h_y)*w_f_f_dy2_0_2(x, y) ' \
            + '+ f(t, x - 2*h_x, y)*w_f_f_dy2_m2_0(x, y) ' \
            + '+ f(t, x - 2*h_x, y - h_y)*w_f_f_dy2_m2_m1(x, y) ' \
            + '+ f(t, x - 2*h_x, y + h_y)*w_f_f_dy2_m2_1(x, y) ' \
            + '+ f(t, x - h_x, y)*w_f_f_dy2_m1_0(x, y) ' \
            + '+ f(t, x - h_x, y - 2*h_y)*w_f_f_dy2_m1_m2(x, y) ' \
            + '+ f(t, x - h_x, y - h_y)*w_f_f_dy2_m1_m1(x, y) ' \
            + '+ f(t, x - h_x, y + h_y)*w_f_f_dy2_m1_1(x, y) ' \
            + '+ f(t, x - h_x, y + 2*h_y)*w_f_f_dy2_m1_2(x, y) ' \
            + '+ f(t, x + h_x, y)*w_f_f_dy2_1_0(x, y) ' \
            + '+ f(t, x + h_x, y - 2*h_y)*w_f_f_dy2_1_m2(x, y) ' \
            + '+ f(t, x + h_x, y - h_y)*w_f_f_dy2_1_m1(x, y) ' \
            + '+ f(t, x + h_x, y + h_y)*w_f_f_dy2_1_1(x, y) ' \
            + '+ f(t, x + h_x, y + 2*h_y)*w_f_f_dy2_1_2(x, y) ' \
            + '+ f(t, x + 2*h_x, y)*w_f_f_dy2_2_0(x, y) ' \
            + '+ f(t, x + 2*h_x, y - h_y)*w_f_f_dy2_2_m1(x, y) ' \
            + '+ f(t, x + 2*h_x, y + h_y)*w_f_f_dy2_2_1(x, y)'

        assert str(subs.expr) == ans

    def test_x_derivatives(self):
        """
        Check that cross-derivatives produce substitutions in the
        expected manner
        """
        subs = weights_test_setup(use_x_derivs=True)
        assert False
