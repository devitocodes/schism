"""
Tests for fitting of the basis to boundary conditions and interior function
values.
"""

import pytest
import devito as dv

from schism.basic.basis import Basis
from schism.geometry.support_region import SupportRegion
from schism.finite_differences.interpolate_project import Interpolant


class TestInterpolation:
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
        interpolant = Interpolant(support)
        # Check the interior vector has the correct length
        if basis_dim is None:
            check_len = sum([len(support.footprint_map[func][0])
                             for func in funcs])
        else:
            check_len = nfuncs*(s_o+1)
        assert len(interpolant.interior_vector) == check_len

    def test_interior_vector_contents(self):
        """Check that the interior vector is correctly generated"""
        # Create a grid
        # Create one or two functions
        # Create a SupportRegion
        # Create the Interpolant
        # Check the interior vector matches the answer
        return 0  # Placeholder
