"""Tests for the Substitution object"""

import pytest
import devito as dv

from test_interpolation import setup_geom, setup_f
from schism.basic.basis import Basis
from schism.finite_differences.substitution import Substitution


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

        assert False
