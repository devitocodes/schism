"""Tests for utilites and misc tools"""

import pytest
import devito as dv

from schism.utils import retrieve_derivatives
from collections import Counter


class TestSearch:
    """Tests for search functions"""

    grid = dv.Grid(shape=(11, 11), extent=(10., 10.))
    f = dv.TimeFunction(name='f', grid=grid, space_order=2)
    g = dv.TimeFunction(name='g', grid=grid, space_order=2)

    @pytest.mark.parametrize('expr, ans',
                             [(f, ()),
                              (f+g, ()),
                              (f+f.dx+f.dx2, (f.dx, f.dx2)),
                              (f.laplace, (f.dx2, f.dy2)),
                              (f.dx2 + g.dx2, (f.dx2, g.dx2)),
                              (f+g+f.dx+g.dy+g.dx, (f.dx, g.dx, g.dy))])
    def test_retrieve_derivs(self, expr, ans):
        """Check that derivatives are retrieved correctly"""
        retrieved = retrieve_derivatives(expr)

        assert Counter(retrieved) == Counter(ans)
