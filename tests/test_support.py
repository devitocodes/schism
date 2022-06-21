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

    @pytest.mark.parametrize('s_o',
                             [2, 4, 6, 8])
    def test_mismatched_sizes(self, s_o):
        """
        Test to check that using basis functions of different spans works as
        expected.
        """
        grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
        f = dv.TimeFunction(name='f', grid=grid, space_order=s_o)
        g = dv.TimeFunction(name='g', grid=grid, space_order=s_o+2)

        basis_map = {f: Basis('f', grid.dimensions, f.space_order),
                     g: Basis('g', grid.dimensions, g.space_order)}

        radius_map = {f: f.space_order//2, g: g.space_order//2}

        sr = SupportRegion(basis_map, radius_map)

        for dim in range(2):
            assert np.amin(sr.footprint_map[f]) == -s_o//2
            assert np.amax(sr.footprint_map[f]) == s_o//2
            assert np.amin(sr.footprint_map[g]) == -1-s_o//2
            assert np.amax(sr.footprint_map[g]) == 1+s_o//2

    def test_expand_radius(self):
        """
        Test to check that incrementing the radius of the support region
        results in the correct footprint
        """
        grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
        f = dv.TimeFunction(name='f', grid=grid, space_order=4)
        g = dv.TimeFunction(name='g', grid=grid, space_order=4)

        basis_map = {f: Basis('f', grid.dimensions, f.space_order),
                     g: Basis('g', grid.dimensions, g.space_order)}

        radius_map = {f: f.space_order//2, g: g.space_order//2}
        larger_radius_map = {f: 1+f.space_order//2, g: 1+g.space_order//2}

        support = SupportRegion(basis_map, radius_map)
        large_support = SupportRegion(basis_map, larger_radius_map)

        expanded = support.expand_radius(1)

        for func in (f, g):
            for i in range(2):
                result = expanded.footprint_map[func][i]
                check = large_support.footprint_map[func][i]
            assert np.all(result == check)
