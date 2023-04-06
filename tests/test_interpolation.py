"""
Tests for fitting of the basis to boundary conditions and interior function
values.
"""

import pytest
import os
import devito as dv
import numpy as np
import sympy as sp

from schism.conditions.boundary_conditions import SingleCondition
from schism.basic.basis import Basis
from schism.geometry.support_region import SupportRegion
from schism.finite_differences.interpolate_project import Interpolant, \
    Projection


class DummyGroup:
    """Dummy class as a placeholder for ConditionGroup"""
    def __init__(self, funcs, conditions):
        self.funcs = funcs
        self.conditions = conditions


class DummyGeometry:
    """Dummy class as a placeholder for Geometry"""
    def __init__(self, **kwargs):
        self.grid = kwargs.get('grid', None)
        self.interior_mask = kwargs.get('interior_mask', None)
        self.boundary_mask = kwargs.get('boundary_mask', None)
        self.dense_pos = kwargs.get('dense_pos', None)
        self.b_mask_1D = kwargs.get('b_mask_1D', None)


class DummySkin:
    """Dummy class as a placeholder for ModifiedSkin"""
    def __init__(self, **kwargs):
        self.geometry = kwargs.get('geometry', None)
        self.points = kwargs.get('points', None)
        self.npts = len(self.points[0])


def setup_geom(setup, grid):
    """Set up dummy geometry and skin objects"""
    assert grid.shape == (3, 3)

    origin = (sp.core.numbers.Zero(), sp.core.numbers.Zero())
    origin_x = (grid.dimensions[0].spacing/2, sp.core.numbers.Zero())
    origin_y = (sp.core.numbers.Zero(), grid.dimensions[1].spacing/2)
    origins = (origin, origin_x, origin_y)

    interior_mask = {orig: np.zeros(grid.shape, dtype=bool)
                     for orig in origins}
    boundary_mask = np.zeros(grid.shape, dtype=bool)
    dense_pos = [np.zeros(grid.shape) for dim in grid.dimensions]
    if setup == 0:  # Flat surface
        skin_points = (np.array([0, 1, 2]), np.array([1, 1, 1]))
        interior_mask[origin][:, :2] = True
        interior_mask[origin_x][:, :2] = True
        interior_mask[origin_y][:, :2] = True
        boundary_mask[:, 2] = True
        dense_pos[1][:, 2] = 0.4
        b_mask_1D = (np.zeros(grid.shape, dtype=bool), boundary_mask.copy())

    elif setup == 1:  # Tilted surface
        skin_points = (np.array([0, 0, 1, 1, 2]),
                       np.array([2, 1, 1, 0, 0]))
        interior = (np.array([0, 0, 1, 0, 1, 2]),
                    np.array([2, 1, 1, 0, 0, 0]))
        interior_mask[origin][interior] = True
        interior_mask[origin_x][interior] = True
        interior_mask[origin_y][interior] = True
        boundary = (np.array([1, 2, 2]), np.array([2, 2, 1]))
        boundary_mask[boundary] = True
        positions = np.array([np.sqrt(2)/2, -np.sqrt(2)/2, np.sqrt(2)/2])
        dense_pos[0][boundary] = positions
        dense_pos[1][boundary] = positions
        b_mask_template = np.zeros(grid.shape)
        b_mask_1D_points = (np.array([1, 2]), np.array([2, 1]))
        b_mask_template[b_mask_1D_points] = True
        b_mask_1D = (b_mask_template.copy(), b_mask_template.copy())

    geometry = DummyGeometry(grid=grid, interior_mask=interior_mask,
                             boundary_mask=boundary_mask, dense_pos=dense_pos,
                             b_mask_1D=b_mask_1D)
    skin = DummySkin(geometry=geometry, points=skin_points)

    return geometry, skin


def setup_f(func_type, grid):
    """Set up a function and a dummy group"""
    if func_type == 'scalar':
        f = dv.TimeFunction(name='f', grid=grid, space_order=2)
        conditions = (SingleCondition(dv.Eq(f, 0)),
                      SingleCondition(dv.Eq(f.laplace, 0)))
        group = DummyGroup((f,), conditions)
    else:
        f = dv.VectorTimeFunction(name='f', grid=grid, space_order=2)
        conditions = (SingleCondition(dv.Eq(dv.div(f), 0)),)
        group = DummyGroup((f[0], f[1]), conditions)

    return f, group


def mask_test_setup(setup, func_type):
    """Set up the interpolant for the boundary and interior mask tests"""
    grid = dv.Grid(shape=(3, 3), extent=(10., 10.))
    f, group = setup_f(func_type, grid)

    basis_map = {func: Basis(func.name, grid.dimensions, 2)
                 for func in group.funcs}

    radius_map = {func: 1 for func in group.funcs}

    # Create a SupportRegion
    deriv = f.dx
    support = SupportRegion(basis_map, radius_map, deriv)

    geometry, skin = setup_geom(setup, grid)

    # Create the Interpolant
    interpolant = Interpolant(support, group, basis_map,
                              skin.geometry, skin.points)

    return interpolant


class TestInterpolant:
    """Tests for the Interpolant object"""

    @pytest.mark.parametrize('ndims, basis_dim',
                             [(1, None), (2, None), (2, 0), (2, 1),
                              (3, None), (3, 0)])
    @pytest.mark.parametrize('nfuncs',
                             [1, 2])
    @pytest.mark.parametrize('s_o',
                             [2, 4, 6])
    def test_interior_vector(self, ndims, basis_dim, nfuncs, s_o):
        """
        Check that the interior vector is of the correct size (weaker check but
        more general).
        """
        # Need to create a grid with some dimensionality
        shape = tuple([3 for dim in range(ndims)])
        extent = tuple([10. for dim in range(ndims)])
        grid = dv.Grid(shape=shape, extent=extent)
        origin = tuple([sp.core.numbers.Zero() for dim in range(ndims)])
        # Need to create one or two functions
        funcs = [dv.TimeFunction(name='f'+str(i), grid=grid, space_order=s_o)
                 for i in range(nfuncs)]
        conditions = tuple([SingleCondition(dv.Eq(func, 0)) for func in funcs])
        group = DummyGroup(tuple(funcs), conditions)
        if basis_dim is None:
            basis_map = {func: Basis(func.name, grid.dimensions, s_o)
                         for func in funcs}
            deriv = dv.Derivative(funcs[0], grid.dimensions[0])
        else:
            basis_map = {func: Basis(func.name,
                                     (grid.dimensions[basis_dim],),
                                     s_o)
                         for func in funcs}
            deriv = dv.Derivative(funcs[0], grid.dimensions[basis_dim])
        radius_map = {func: int(s_o//2) for func in funcs}
        # Create a SupportRegion
        support = SupportRegion(basis_map, radius_map, deriv)

        geometry = DummyGeometry(grid=grid,
                                 interior_mask={origin:
                                                np.full(grid.shape, True,
                                                        dtype=bool)},
                                 boundary_mask=np.full(grid.shape, False,
                                                       dtype=bool),
                                 dense_pos=tuple([np.full(grid.shape, 0)
                                                  for dim in range(ndims)]),
                                 b_mask_1D=tuple([np.full(grid.shape, False,
                                                          dtype=bool)
                                                  for dim in range(ndims)]))
        skin = DummySkin(geometry=geometry,
                         points=tuple([np.array([], dtype=int)
                                       for dim in range(ndims)]))

        # Create the Interpolant
        interpolant = Interpolant(support, group, basis_map,
                                  skin.geometry, skin.points)
        # Check the interior vector has the correct length
        if basis_dim is None:
            check_len = sum([len(support.footprint_map[func][0])
                             for func in funcs])
        else:
            check_len = nfuncs*(s_o+1)
        assert len(interpolant.interior_vector) == check_len

    answers = ['[f0(t, x - h_x) f0(t, x) f0(t, x + h_x)]',
               '[f0(t, x - h_x) f0(t, x) f0(t, x + h_x) f1(t, x - h_x) '
               + 'f1(t, x)\n f1(t, x + h_x)]',
               '[f0(t, x - h_x, y - h_y) f0(t, x - h_x, y) '
               + 'f0(t, x - h_x, y + h_y)\n f0(t, x, y - h_y) f0(t, x, y) '
               + 'f0(t, x, y + h_y) f0(t, x + h_x, y - h_y)\n '
               + 'f0(t, x + h_x, y) f0(t, x + h_x, y + h_y)]',
               '[f0(t, x - h_x, y) f0(t, x, y) f0(t, x + h_x, y)]',
               '[f0(t, x, y - h_y) f0(t, x, y) f0(t, x, y + h_y)]',
               '[f0(t, x - h_x, y, z) f0(t, x, y, z) f0(t, x + h_x, y, z)]']

    @pytest.mark.parametrize('ndims, basis_dim, nfuncs, ans',
                             [(1, None, 1, answers[0]),
                              (1, None, 2, answers[1]),
                              (2, None, 1, answers[2]),
                              (2, 0, 1, answers[3]),
                              (2, 1, 1, answers[4]),
                              (3, 0, 1, answers[5])])
    def test_interior_vector_contents(self, ndims, basis_dim, nfuncs, ans):
        """Check that the interior vector is correctly generated"""
        # Create a grid
        shape = tuple([11 for dim in range(ndims)])
        extent = tuple([10. for dim in range(ndims)])
        grid = dv.Grid(shape=shape, extent=extent)
        origin = tuple([sp.core.numbers.Zero() for dim in range(ndims)])
        funcs = [dv.TimeFunction(name='f'+str(i), grid=grid, space_order=2)
                 for i in range(nfuncs)]
        conditions = tuple([SingleCondition(dv.Eq(func, 0)) for func in funcs])
        group = DummyGroup(tuple(funcs), conditions)
        if basis_dim is None:
            deriv = dv.Derivative(funcs[0], grid.dimensions[0])
            basis_map = {func: Basis(func.name, grid.dimensions, 2)
                         for func in group.funcs}
        else:
            deriv = dv.Derivative(funcs[0], grid.dimensions[basis_dim])
            basis_map = {func: Basis(func.name,
                                     (grid.dimensions[basis_dim],),
                                     2)
                         for func in group.funcs}
        radius_map = {func: 1 for func in group.funcs}
        # Create a SupportRegion
        support = SupportRegion(basis_map, radius_map, deriv)
        geometry = DummyGeometry(grid=grid,
                                 interior_mask={origin:
                                                np.full(grid.shape, True,
                                                        dtype=bool)},
                                 boundary_mask=np.full(grid.shape, False,
                                                       dtype=bool),
                                 dense_pos=tuple([np.full(grid.shape, 0)
                                                  for dim in range(ndims)]),
                                 b_mask_1D=tuple([np.full(grid.shape, False,
                                                          dtype=bool)
                                                  for dim in range(ndims)]))
        skin = DummySkin(geometry=geometry,
                         points=tuple([np.array([], dtype=int)
                                       for dim in range(ndims)]))
        # Create the Interpolant
        interpolant = Interpolant(support, group, basis_map,
                                  skin.geometry, skin.points)

        # Check the interior vector matches the answer
        assert str(interpolant.interior_vector) == ans

    @pytest.mark.parametrize('basis_dim, func_type, s_o',
                             [(None, 'scalar', 2),
                              (None, 'vector', 2),
                              (0, 'scalar', 4)])
    def test_interior_matrix(self, basis_dim, func_type, s_o):
        """Check that the master interior matrix is correctly generated"""
        # Create a grid and the specified function type
        grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
        origin = (sp.core.numbers.Zero(), sp.core.numbers.Zero())
        interior_mask = {origin: np.full(grid.shape, True, dtype=bool)}
        if func_type == 'vector':
            origin_x = (grid.dimensions[0].spacing/2, sp.core.numbers.Zero())
            origin_y = (sp.core.numbers.Zero(), grid.dimensions[1].spacing/2)
            extra_mask = {origin_x: np.full(grid.shape, True, dtype=bool),
                          origin_y: np.full(grid.shape, True, dtype=bool)}
            interior_mask.update(extra_mask)

        if func_type == 'scalar':
            f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
            conditions = (SingleCondition(dv.Eq(f, 0)),)
            group = DummyGroup((f,), conditions)
            deriv_func = f
        else:
            f = dv.VectorTimeFunction(name='f', grid=grid, space_order=s_o)
            conditions = (SingleCondition(dv.Eq(f[0], 0)),
                          SingleCondition(dv.Eq(f[1], 0)))
            group = DummyGroup((f[0], f[1]), conditions)
            deriv_func = f[0]

        if basis_dim is None:
            deriv = dv.Derivative(deriv_func, grid.dimensions[0])
            basis_map = {func: Basis(func.name, grid.dimensions, s_o)
                         for func in group.funcs}
        else:
            deriv = dv.Derivative(deriv_func, grid.dimensions[basis_dim])
            basis_map = {func: Basis(func.name,
                                     (grid.dimensions[basis_dim],),
                                     s_o)
                         for func in group.funcs}

        radius_map = {func: s_o//2 for func in group.funcs}

        # Create a SupportRegion
        support = SupportRegion(basis_map, radius_map, deriv)

        geometry = DummyGeometry(grid=grid,
                                 interior_mask=interior_mask,
                                 boundary_mask=np.full(grid.shape, False,
                                                       dtype=bool),
                                 dense_pos=tuple([np.full(grid.shape, 0)
                                                  for dim in range(2)]),
                                 b_mask_1D=tuple([np.full(grid.shape, False,
                                                          dtype=bool)
                                                  for dim in range(2)]))
        skin = DummySkin(geometry=geometry,
                         points=tuple([np.array([], dtype=int)
                                       for dim in range(2)]))
        # Create the Interpolant
        interpolant = Interpolant(support, group, basis_map,
                                  skin.geometry, skin.points)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/interpolation_test_results/interior_matrix/' \
            + str(basis_dim) + func_type + str(s_o) + '.npy'

        check = np.load(fname)

        assert np.all(np.isclose(interpolant.interior_matrix, check))

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    def test_interior_mask(self, setup, func_type):
        """Check that the interior mask is correctly generated"""
        # Create the Interpolant
        interpolant = mask_test_setup(setup, func_type)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/interpolation_test_results/interior_mask/' \
            + str(setup) + func_type + '.npy'

        check = np.load(fname)

        assert np.all(interpolant.interior_mask == check)

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    def test_boundary_mask(self, setup, func_type):
        """Check that the boundary mask is correctly generated"""
        interpolant = mask_test_setup(setup, func_type)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/interpolation_test_results/boundary_mask/' \
            + str(setup) + func_type + '.npy'

        check = np.load(fname)

        assert np.all(interpolant.boundary_mask == check)

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    def test_boundary_matrices(self, setup, func_type):
        """Check that the boundary matrices are correctly generated"""
        interpolant = mask_test_setup(setup, func_type)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/interpolation_test_results/boundary_mats/' \
            + str(setup) + func_type + '.npy'

        check = np.load(fname)
        assert np.all(np.isclose(np.array(interpolant.boundary_matrices),
                                 check))

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    def test_rank_check(self, setup, func_type):
        """Check that the rank of each matrix is correctly calculated"""
        interpolant = mask_test_setup(setup, func_type)

        path = os.path.dirname(os.path.abspath(__file__))
        # Filename for the ranks
        fnamer = path + '/results/interpolation_test_results/ranks/' \
            + str(setup) + func_type + '.npy'

        # Filename for the rank mask
        fnamem = path + '/results/interpolation_test_results/rank_mask/' \
            + str(setup) + func_type + '.npy'

        checkr = np.load(fnamer)
        checkm = np.load(fnamem)

        assert np.all(checkr == interpolant.rank)
        assert np.all(checkm == interpolant.rank_mask)

        if setup == 0 and func_type == 'vector':
            assert interpolant.all_full_rank is False
        else:
            assert interpolant.all_full_rank is True

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type', ['scalar', 'vector'])
    def test_pinv(self, setup, func_type):
        """Check that the pseudoinverse is correctly calculated"""
        interpolant = mask_test_setup(setup, func_type)

        id = np.matmul(interpolant.pinv,
                       interpolant.matrix[interpolant.rank_mask])

        assert np.all(np.isclose(id, np.eye(interpolant.matrix.shape[-1])))

    @pytest.mark.parametrize('setup', [0, 1])
    @pytest.mark.parametrize('func_type, deriv_type',
                             [('scalar', 'dx'),
                              ('scalar', 'dy2'),
                              ('scalar', 'dxf'),
                              ('vector', 'dx'),
                              ('vector', 'dxb')])
    def test_projection(self, setup, func_type, deriv_type):
        """
        Check that the interpolant is correctly projected onto the interior
        stencil.
        """
        interpolant = mask_test_setup(setup, func_type)
        group = interpolant.group
        basis_map = interpolant.basis_map

        x, y = group.funcs[0].space_dimensions

        if deriv_type == 'dx':
            deriv = group.funcs[0].dx
        elif deriv_type == 'dy2':
            deriv = group.funcs[0].dy2
        elif deriv_type == 'dxf':
            deriv = group.funcs[0].dx(x0=x+x.spacing/2)
        elif deriv_type == 'dxb':
            deriv = group.funcs[0].dx(x0=x)

        projection = Projection(deriv, group, basis_map)

        stencil = interpolant.project(projection)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/interpolation_test_results/projection/' \
            + str(setup) + func_type + deriv_type + '.npy'

        check = np.load(fname)
        assert np.all(np.isclose(stencil, check))
