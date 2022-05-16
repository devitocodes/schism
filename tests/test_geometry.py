"""Tests for geometry objects"""
import pytest
import pickle
import os

import numpy as np

from schism import BoundaryGeometry


def read_sdf(surface, dims):
    """Unpickle an sdf"""
    path = os.path.dirname(os.path.abspath(__file__))
    fname = path + '/sdfs/' + surface + '_' + str(dims) + 'd.dat'
    with open(fname, 'rb') as f:
        sdf = pickle.load(f)
    return sdf


class TestBoundaryGeometry:
    """Tests for the BoundaryGeometry object"""
    @pytest.mark.parametrize('surface', ['45', '45_mirror',
                                         'horizontal', 'vertical'])
    @pytest.mark.parametrize('dims', [2, 3])
    def test_unit_normal_magnitude(self, surface, dims):
        """Check that unit normals have correct magnitude"""
        rtol = 0.1  # Allow up to 10% deviation
        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)
        # Check that boundary normal magnitude == 1 where the sdf<=0.5*spacing
        spacing = sdf.grid.spacing
        max_dist = np.sqrt(sum([(inc/2)**2 for inc in spacing]))

        # Trim edges off data, as normal calculation in corners is imperfect
        slices = tuple([slice(1, -1) for dim in sdf.grid.dimensions])
        data = sdf.data[slices]

        mask = np.abs(data) <= max_dist

        normals = [bg.n[i].data[slices][mask] for i in range(len(spacing))]

        n_mag = np.sqrt(sum([normals[i]**2 for i in range(len(spacing))]))

        assert np.all(np.isclose(n_mag, 1, rtol=rtol))
