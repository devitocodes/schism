"""Tests for modified point identification"""

import pytest
import devito as dv
import numpy as np

from schism.geometry.skin import stencil_footprint


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
