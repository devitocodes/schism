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

    r2o2 = np.sqrt(2)/2  # Used repeatedly in next test

    @pytest.mark.parametrize('surface, dims, answer',
                             [('45', 2, (-r2o2, r2o2)),
                              ('45', 3, (-r2o2, 0, r2o2)),
                              ('45_mirror', 2, (r2o2, r2o2)),
                              ('45_mirror', 3, (r2o2, 0, r2o2)),
                              ('horizontal', 2, (0., 1.)),
                              ('horizontal', 3, (0., 0., 1.)),
                              ('vertical', 2, (1., 0.)),
                              ('vertical', 3, (1., 0., 0.))])
    def test_unit_normal_direction(self, surface, dims, answer):
        """Check that unit normals point in the correct direction"""
        rtol = 0.1  # Allow up to 10% deviation

        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)
        # Check boundary normals where the sdf<=0.5*spacing
        spacing = sdf.grid.spacing
        max_dist = np.sqrt(sum([(inc/2)**2 for inc in spacing]))

        # Trim edges off data, as normal calculation in corners is imperfect
        slices = tuple([slice(1, -1) for dim in sdf.grid.dimensions])
        data = sdf.data[slices]

        mask = np.abs(data) <= max_dist

        normals = [bg.n[i].data[slices][mask] for i in range(len(spacing))]

        for i in range(len(normals)):
            assert np.all(np.isclose(normals[i], answer[i], rtol=rtol))

    @pytest.mark.parametrize('surface, dims', [('45', 2),
                                               ('45_mirror', 2),
                                               ('horizontal', 2),
                                               ('vertical', 2),
                                               ('45', 3),
                                               ('45_mirror', 3),
                                               ('horizontal', 3),
                                               ('vertical', 3)])
    def test_boundary_mask(self, surface, dims):
        """Check that the boundary points are correctly identified"""

        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)

        # Check boundary mask size
        assert bg.boundary_mask.shape == bg.grid.shape

        # Trim edges off data, as normal calculation in corners is imperfect
        slices = tuple([slice(2, -2) for dim in sdf.grid.dimensions])
        data = bg.boundary_mask[slices]

        check_mask = np.zeros(data.shape, dtype=bool)
        if surface == '45':
            # Diagonal indices
            diag_indices = np.arange(data.shape[0])
            # Below the diagonal
            diag_indices_n1 = np.arange(1, data.shape[0])
            # Above the diagonal
            diag_indices_1 = np.arange(0, data.shape[0]-1)
            # Fill the diagonals
            if dims == 2:
                check_mask[diag_indices, diag_indices] = True
                check_mask[diag_indices_n1, diag_indices_n1-1] = True
                check_mask[diag_indices_1, diag_indices_1+1] = True
            elif dims == 3:
                check_mask[diag_indices, :, diag_indices] = True
                check_mask[diag_indices_n1, :, diag_indices_n1-1] = True
                check_mask[diag_indices_1, :, diag_indices_1+1] = True

        elif surface == '45_mirror':
            # Diagonal indices
            diag_indices = np.arange(data.shape[0])
            # Above the diagonal
            diag_indices_1 = np.arange(0, data.shape[0]-1)
            # Two above the diagonal
            diag_indices_2 = np.arange(0, data.shape[0]-2)
            # Fill the diagonals
            if dims == 2:
                check_mask[diag_indices, diag_indices] = True
                check_mask[diag_indices_1, diag_indices_1+1] = True
                check_mask[diag_indices_2, diag_indices_2+2] = True
            elif dims == 3:
                check_mask[diag_indices, :, diag_indices] = True
                check_mask[diag_indices_1, :, diag_indices_1+1] = True
                check_mask[diag_indices_2, :, diag_indices_2+2] = True

            check_mask = check_mask[::-1]

        elif surface == 'horizontal':
            if dims == 2:
                check_mask[:, 48:50] = True
            elif dims == 3:
                check_mask[:, :, 48:50] = True

        elif surface == 'vertical':
            if dims == 2:
                check_mask[48:50, :] = True
            elif dims == 3:
                check_mask[48:50, :, :] = True

        assert np.all(data == check_mask)

    @pytest.mark.parametrize('surface, dims', [('45', 2),
                                               ('45_mirror', 2),
                                               ('horizontal', 2),
                                               ('vertical', 2),
                                               ('45', 3),
                                               ('45_mirror', 3),
                                               ('horizontal', 3),
                                               ('vertical', 3)])
    def test_boundary_points(self, surface, dims):
        """Check that boundary points are correctly identified"""
        # Create the SDF
        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)
        # Check that the boundary points recovers the mask for that sdf
        check_mask = np.zeros(bg.grid.shape, dtype=bool)
        check_mask[bg.boundary_points] = True
        assert np.all(check_mask == bg.boundary_mask)
        # TODO: Wants to test some other surfaces too
        # TODO: Like sines, eggboxes, etc

    @pytest.mark.parametrize('surface, dims', [('45', 2),
                                               ('horizontal', 2)])
    def test_positions(self, surface, dims):
        """Check that calculated point positions are correct"""
        # Create the SDF
        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)

        # For 45, they're going to be (0, 0), (0.5, -0.5), (-0.5, 0.5)
        if surface == '45':
            on = np.logical_and(np.isclose(bg.positions[0], 0),
                                np.isclose(bg.positions[1], 0))
            above = np.logical_and(np.isclose(bg.positions[0], 0.5),
                                   np.isclose(bg.positions[1], -0.5))
            below = np.logical_and(np.isclose(bg.positions[0], -0.5),
                                   np.isclose(bg.positions[1], 0.5))

            assert np.all(np.logical_or.reduce((on, above, below)))
        # For horizontal, they're going to be (0, 0.5), (0.5, 0)
        elif surface == 'horizontal':
            above = np.logical_and(np.isclose(bg.positions[0], 0),
                                   np.isclose(bg.positions[1], -0.5))
            below = np.logical_and(np.isclose(bg.positions[0], 0),
                                   np.isclose(bg.positions[1], 0.5))

    @pytest.mark.parametrize('surface, dims', [('45_mirror', 2),
                                               ('horizontal', 2),
                                               ('horizontal', 3)])
    def test_interior_mask(self, surface, dims):
        """Check that the interior mask is correct"""
        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)

        # Trim edges off data, as normal calculation in corners is imperfect
        slices = tuple([slice(2, -2) for dim in sdf.grid.dimensions])

        # Create a meshgrid of indices
        if dims == 2:
            x, z = np.meshgrid(np.arange(bg.grid.shape[0]),
                               np.arange(bg.grid.shape[1]), indexing='ij')
        elif dims == 3:
            x, y, z = np.meshgrid(np.arange(bg.grid.shape[0]),
                                  np.arange(bg.grid.shape[1]),
                                  np.arange(bg.grid.shape[2]), indexing='ij')

        if surface == '45_mirror':
            check_mask = x + z < 100
        elif surface == 'horizontal':
            check_mask = z < 50

        assert np.all(bg.interior_mask[slices] == check_mask[slices])
