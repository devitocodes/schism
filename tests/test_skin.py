"""Tests for modified point identification"""

import pytest
import devito as dv
import numpy as np
import sympy as sp

from schism.geometry.skin import stencil_footprint, ModifiedSkin


class DummyGeometry:
    """Dummy class to replace BoundaryGeometry for testing"""
    def __init__(self, grid):
        self.grid = grid
        dims = grid.dimensions
        # Only works with square/cube grids

        imask = np.zeros((len(dims)+1,) + grid.shape, dtype=bool)
        zero = sp.core.numbers.Zero()

        # Ugly but will do for now
        if len(self.grid.shape) == 2:  # 2D
            origins = [(zero, zero), (dims[0].spacing/2, zero),
                       (zero, dims[1].spacing/2)]
            bpoints = ([], [])
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if i == j:
                        bpoints[0].append(i)
                        bpoints[1].append(j)
                        imask[1, i, j] = True
                    elif i > j:
                        imask[:, i, j] = True
        else:  # 3D
            origins = [(zero, zero, zero), (dims[0].spacing/2, zero, zero),
                       (zero, dims[1].spacing/2, zero),
                       (zero, zero, dims[2].spacing/2)]
            bpoints = ([], [], [])
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    for k in range(grid.shape[2]):
                        if i == k:
                            bpoints[0].append(i)
                            bpoints[1].append(j)
                            bpoints[2].append(k)
                            imask[1, i, j, k] = True
                        elif i > k:
                            imask[:, i, j, k] = True

        self.boundary_points = tuple([np.array(bpoints[i])
                                      for i in range(len(bpoints))])

        self.interior_mask = {}
        for i in range(len(origins)):
            self.interior_mask[origins[i]] = imask[i]


class TestMisc:
    """Tests for misc functions used for modified point id"""
    grid1d = dv.Grid(shape=(11,), extent=(10.,))
    grid2d = dv.Grid(shape=(11, 11), extent=(10., 10.))
    grid3d = dv.Grid(shape=(11, 11, 11), extent=(10., 10., 10.))

    f1d = dv.TimeFunction(name='f', grid=grid1d, space_order=4)
    f2d = dv.TimeFunction(name='f', grid=grid2d, space_order=4)
    f3d = dv.TimeFunction(name='f', grid=grid3d, space_order=4)

    @pytest.mark.parametrize('deriv',
                             [f1d.dx, f1d.dx2, f2d.dx, f2d.dy,
                              f2d.dxdy, f2d.dx2, f3d.dx, f3d.dxdy,
                              f3d.dxdydz])
    def test_stencil_footprint(self, deriv):
        """Check that stencil footprints correctly determined"""
        fp = stencil_footprint(deriv)
        # Check it has the right size
        assert fp.shape[-1] == (deriv.expr.space_order+1)**len(deriv.dims)
        # Check the dimensions correspond correctly
        for n in range(len(deriv.expr.space_dimensions)):
            if deriv.expr.space_dimensions[n] not in deriv.dims:
                assert np.all(fp[n] == 0)


class TestSkin:
    """Tests for the ModifiedSkin object"""
    grid2d = dv.Grid(shape=(4, 4), extent=(10., 10.))
    grid3d = dv.Grid(shape=(4, 4, 4), extent=(10., 10., 10.))
    x2d, y2d = grid2d.dimensions
    x3d, y3d, z3d = grid3d.dimensions

    f2d = dv.TimeFunction(name='f', grid=grid2d, space_order=4)
    f3d = dv.TimeFunction(name='f', grid=grid3d, space_order=2)

    v2d = dv.VectorTimeFunction(name='v', grid=grid2d, space_order=4)
    v3d = dv.VectorTimeFunction(name='v', grid=grid3d, space_order=2)

    geom0 = DummyGeometry(grid2d)
    geom1 = DummyGeometry(grid3d)

    # Answer for first two
    ans0 = (np.array([1, 2, 2, 3, 3]), np.array([0, 0, 1, 1, 2]))
    # Answer for third
    ans1 = (np.array([1, 2, 2, 3, 3, 3]), np.array([0, 0, 1, 0, 1, 2]))
    # Answer for fourth
    ans2 = (np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
            np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))
    # Another answer
    ans3 = (np.array([0, 1, 1, 2, 2, 2, 3, 3, 3]),
            np.array([0, 0, 1, 0, 1, 2, 1, 2, 3]))

    @pytest.mark.parametrize('deriv, geometry, ans',
                             [(f2d.dx, geom0, ans0),
                              (f2d.dy, geom0, ans0),
                              (f2d.dxdy, geom0, ans1),
                              (f3d.dx, geom1, ans2),
                              (v2d[0].dx(x0=x2d), geom0, ans0),
                              (v2d[1].dy(x0=y2d), geom0, ans0),
                              (v3d[0].dx(x0=x3d), geom1, ans2),
                              (f2d.dx(x0=x2d+x2d.spacing/2), geom0, ans3),
                              (f2d.dy(x0=y2d+y2d.spacing/2), geom0, ans0)])
    def test_modified_points(self, deriv, geometry, ans):
        """Check that modified points are correctly identified"""
        skin = ModifiedSkin(deriv, geometry)

        for i in range(len(skin.points)):
            assert np.all(skin.points[i] == ans[i])
