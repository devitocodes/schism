"""
Tests for fitting of the basis to boundary conditions and interior function
values.
"""

import pytest
import os
import devito as dv
import numpy as np

from schism.basic.basis import Basis
from schism.geometry.support_region import SupportRegion
from schism.finite_differences.interpolate_project import Interpolant


class DummyGroup:
    """Dummy class as a placeholder for ConditionGroup"""
    def __init__(self, funcs):
        self.funcs = funcs


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
        shape = tuple([11 for dim in range(ndims)])
        extent = tuple([10. for dim in range(ndims)])
        grid = dv.Grid(shape=shape, extent=extent)
        # Need to create one or two functions
        funcs = [dv.TimeFunction(name='f'+str(i), grid=grid, space_order=s_o)
                 for i in range(nfuncs)]
        group = DummyGroup(tuple(funcs))
        if basis_dim is None:
            basis_map = {func: Basis(func.name, grid.dimensions, s_o)
                         for func in funcs}
        else:
            basis_map = {func: Basis(func.name,
                                     (grid.dimensions[basis_dim],),
                                     s_o)
                         for func in funcs}
        radius_map = {func: int(s_o//2) for func in funcs}
        # Create a SupportRegion
        support = SupportRegion(basis_map, radius_map)
        # Create the Interpolant
        interpolant = Interpolant(support, group)
        # Check the interior vector has the correct length
        if basis_dim is None:
            check_len = sum([len(support.footprint_map[func][0])
                             for func in funcs])
        else:
            check_len = nfuncs*(s_o+1)
        assert len(interpolant.interior_vector) == check_len

    answers = ['Matrix([[f0[t, x - 1]], [f0[t, x]], [f0[t, x + 1]]])',
               'Matrix([[f0[t, x - 1]], [f0[t, x]], [f0[t, x + 1]], '
               + '[f1[t, x - 1]], [f1[t, x]], [f1[t, x + 1]]])',
               'Matrix([[f0[t, x - 1, y - 1]], [f0[t, x, y - 1]], '
               + '[f0[t, x + 1, y - 1]], [f0[t, x - 1, y]], [f0[t, x, y]], '
               + '[f0[t, x + 1, y]], [f0[t, x - 1, y + 1]], '
               + '[f0[t, x, y + 1]], [f0[t, x + 1, y + 1]]])',
               'Matrix([[f0[t, x - 1, y]], [f0[t, x, y]], [f0[t, x + 1, y]]])',
               'Matrix([[f0[t, x, y - 1]], [f0[t, x, y]], [f0[t, x, y + 1]]])',
               'Matrix([[f0[t, x - 1, y, z]], [f0[t, x, y, z]], '
               + '[f0[t, x + 1, y, z]]])']

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
        funcs = [dv.TimeFunction(name='f'+str(i), grid=grid, space_order=2)
                 for i in range(nfuncs)]
        group = DummyGroup(tuple(funcs))
        if basis_dim is None:
            basis_map = {func: Basis(func.name, grid.dimensions, 2)
                         for func in group.funcs}
        else:
            basis_map = {func: Basis(func.name,
                                     (grid.dimensions[basis_dim],),
                                     2)
                         for func in group.funcs}
        radius_map = {func: 1 for func in group.funcs}
        # Create a SupportRegion
        support = SupportRegion(basis_map, radius_map)
        # Create the Interpolant
        interpolant = Interpolant(support, group)
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
        if func_type == 'scalar':
            f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
            group = DummyGroup((f,))
        else:
            f = dv.VectorTimeFunction(name='f', grid=grid, space_order=s_o)
            group = DummyGroup((f[0], f[1]))

        if basis_dim is None:
            basis_map = {func: Basis(func.name, grid.dimensions, s_o)
                         for func in group.funcs}
        else:
            basis_map = {func: Basis(func.name,
                                     (grid.dimensions[basis_dim],),
                                     s_o)
                         for func in group.funcs}

        radius_map = {func: s_o//2 for func in group.funcs}

        # Create a SupportRegion
        support = SupportRegion(basis_map, radius_map)
        # Create the Interpolant
        interpolant = Interpolant(support, group, basis_map)

        path = os.path.dirname(os.path.abspath(__file__))
        fname = path + '/results/interpolation_test_results/interior_matrix' \
            + str(basis_dim) + func_type + str(s_o) + '.npy'

        check = np.load(fname)

        assert np.all(np.isclose(interpolant.interior_matrix, check))
