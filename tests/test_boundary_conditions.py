"""Tests for boundary conditions"""
import pytest
import devito as dv

from schism.conditions import BoundaryConditions


class TestBCs:
    """Tests for the BoundaryConditions object"""

    grid = dv.Grid(shape=(11,), extent=(10.,))
    f = dv.TimeFunction(name='f', grid=grid)

    @pytest.mark.parametrize('eqs, ans',
                             [([dv.Eq(f, 0), dv.Eq(f, 0)],
                               [dv.Eq(f, 0)])])
    def test_eq_flattening(self, eqs, ans):
        """
        Check that equations are correctly flattened with duplicates removed.
        """
        bcs = BoundaryConditions(eqs)
        assert bcs.equations == tuple(ans)
