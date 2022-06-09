"""Tests for the support region object and associate functions"""

import pytest
import numpy as np
import devito as dv

from schism.basic import Basis
from schism.geometry.support_region import SupportRegion


class TestSupport:
    """Tests for the SupportRegion class"""
    @pytest.mark.parametrize('ndims, selected',
                             [(1, 0), (2, 0), (2, 1),
                              (3, 0), (3, 1), (3, 2)])
    @pytest.mark.parametrize('s_o, radius',
                             [(2, 1), (2, 2), (4, 2), (4, 3), (6, 3)])
    def test_footprint_linear(self, ndims, selected, s_o, radius):
        """Check that the correct footprint is returned for a 1D basis"""
        # Create a grid with some dimensionality
        shape = tuple([11 for dim in range(ndims)])
        extent = tuple([10. for dim in range(ndims)])
        grid = dv.Grid(shape=shape, extent=extent)
        # Create a function on this grid
        f = dv.TimeFunction(name='f', grid=grid)
        # Create a basis in some particular dimension
        basis = Basis('f', (grid.dimensions[selected],), s_o)
        # Create a basis map
        basis_map = {f: basis}
        # Create the radius map (several radii to test)
        radius_map = {f: radius}
        # Create the support region
        sr = SupportRegion(basis_map, radius_map)

        # Check the footprint
        # Check that it has a length equal to 1+2*radius
        assert len(sr.footprint_map[f][0]) == 1+2*radius
        # Check that all of the non-specified dims are full of zeros
        # Check that the other dim contains correct range
        for dim in range(ndims):
            footprint = sr.footprint_map[f][dim]
            if dim != selected:
                assert np.all(footprint == 0)
            else:
                assert np.all(footprint == np.arange(-radius, radius+1))

    @pytest.mark.parametrize('ndims',
                             [1, 2, 3])
    @pytest.mark.parametrize('s_o',
                             [2, 4, 6, 8])
    @pytest.mark.parametrize('inc',
                             [0, 1, 2])
    def test_footprint_circle(self, ndims, s_o, inc):
        """Check that the correct footprint is returned for an N-D basis"""
        # Create a grid with some dimensionality
        shape = tuple([11 for dim in range(ndims)])
        extent = tuple([10. for dim in range(ndims)])
        grid = dv.Grid(shape=shape, extent=extent)
        # Create multiple functions on this grid
        v = dv.VectorTimeFunction(name='v', grid=grid)
        # Create the basis map and radius map
        basis_map = {}
        radius_map = {}
        for dim in range(ndims):
            basis_map[v[dim]] = Basis('v_'+str(dim), grid.dimensions, s_o)
            radius_map[v[dim]] = s_o//2 + inc
        # Create the support region
        sr = SupportRegion(basis_map, radius_map)

        for dim in range(ndims):
            fp = sr.footprint_map[v[dim]]
            rad = sr.radius_map[v[dim]]
            # Check for correct max extent in each dimension
            for i in range(ndims):
                assert np.amin(fp[i]) == -rad
                assert np.amax(fp[i]) == rad
            # Check that footprint all lies within radius + 0.5
            assert np.all(np.sqrt(sum([fp[i]**2 for i in range(ndims)]))
                          < s_o//2 + inc + 0.5)

    # Test mismatched orders + radii
