"""Tests for geometry objects"""
import pytest
import pickle
import os

import numpy as np
import sympy as sp

from schism import BoundaryGeometry
from devito.tools.data_structures import frozendict


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

    @pytest.mark.parametrize('surface', ['45', 'horizontal'])
    def test_dense_pos(self, surface):
        """
        Check that the dense version of the point positions is correctly
        constructed.
        """
        sdf = read_sdf(surface, 2)
        bg = BoundaryGeometry(sdf)
        # Trim edges off data, as normal calculation in corners is imperfect
        slices = tuple([slice(2, -2) for dim in sdf.grid.dimensions])
        data = tuple([bg.dense_pos[dim][slices] for dim in range(2)])

        if surface == '45':
            assert np.all(np.isclose(np.diagonal(data[0], offset=1), 0.5))
            assert np.all(np.isclose(np.diagonal(data[0], offset=0), 0))
            assert np.all(np.isclose(np.diagonal(data[0], offset=-1), -0.5))
            assert np.all(np.isclose(np.diagonal(data[1], offset=1), -0.5))
            assert np.all(np.isclose(np.diagonal(data[1], offset=0), 0))
            assert np.all(np.isclose(np.diagonal(data[1], offset=-1), 0.5))

        elif surface == 'horizontal':
            assert np.all(np.isclose(data[0][:, 48], 0))
            assert np.all(np.isclose(data[0][:, 49], 0))
            assert np.all(np.isclose(data[1][:, 48], 0.5))
            assert np.all(np.isclose(data[1][:, 49], -0.5))

    @pytest.mark.parametrize('surface, dims', [('45_mirror', 2),
                                               ('horizontal', 2),
                                               ('horizontal', 3)])
    def test_interior_mask_unstaggered(self, surface, dims):
        """Check that the interior mask is correct in unstaggered case"""
        sdf = read_sdf(surface, dims)
        bg = BoundaryGeometry(sdf)
        origin = tuple([sp.core.numbers.Zero() for dim in range(dims)])

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

        assert np.all(bg.interior_mask[origin][slices] == check_mask[slices])

    @pytest.mark.parametrize('surface', ['45_mirror', 'horizontal'])
    @pytest.mark.parametrize('setup', [0, 1])
    def test_interior_mask_staggered(self, surface, setup):
        """Check that the interior masks are correct in the staggered case"""
        sdf = read_sdf(surface, 2)
        sdf_x = read_sdf(surface + '_x', 2)
        sdf_y = read_sdf(surface + '_y', 2)
        sdfs = (sdf, sdf_x, sdf_y)

        # Trim edges off data, as normal calculation in corners is imperfect
        slices = tuple([slice(2, -2) for dim in sdf.grid.dimensions])

        x, y = sdf.grid.dimensions
        h_x = x.spacing
        h_y = y.spacing
        zero = sp.core.numbers.Zero()

        if setup == 0:
            cutoff = None
        elif setup == 1:
            cutoff = cutoff = {(h_x/2, zero): 0, (zero, h_y/2): 0}

        bg = BoundaryGeometry(sdfs, cutoff=cutoff)

        xmsh, ymsh = np.meshgrid(np.arange(bg.grid.shape[0]),
                                 np.arange(bg.grid.shape[1]), indexing='ij')

        if surface == '45_mirror' and setup == 0:
            check_mask = xmsh + ymsh < 100
            for origin in bg.interior_mask:
                check = bg.interior_mask[origin][slices] == check_mask[slices]
                assert np.all(check)
        elif surface == '45_mirror' and setup == 1:
            check_mask = xmsh + ymsh < 100
            check_mask_stagger = xmsh + ymsh < 101
            for origin in bg.interior_mask:
                if origin == (zero, zero):
                    check = bg.interior_mask[origin][slices] \
                        == check_mask[slices]
                else:
                    check = bg.interior_mask[origin][slices] \
                        == check_mask_stagger[slices]
                assert np.all(check)
        elif surface == 'horizontal' and setup == 0:
            check_mask = ymsh < 50
            for origin in bg.interior_mask:
                check = bg.interior_mask[origin][slices] == check_mask[slices]
                assert np.all(check)
        elif surface == 'horizontal' and setup == 1:
            check_mask = ymsh < 50
            check_mask_stagger = ymsh < 51
            for origin in bg.interior_mask:
                if origin == (zero, zero):
                    check = bg.interior_mask[origin][slices] \
                        == check_mask[slices]
                else:
                    check = bg.interior_mask[origin][slices] \
                        == check_mask_stagger[slices]
                assert np.all(check)

    def test_1d_boundary_masks(self):
        """Test the 1D versions of the boundary masks"""
        sdf = read_sdf('45_mirror', 2)
        bg = BoundaryGeometry(sdf)

        # Trim edges as geometry calculation gets weird here
        slices = tuple([slice(2, -2) for dim in sdf.grid.dimensions])

        # In the 45 degree cases, both masks should be the same
        assert np.all(bg.b_mask_1D[0][slices] == bg.b_mask_1D[1][slices])
        assert np.count_nonzero(bg.b_mask_1D[0][slices]) == 97

    @pytest.mark.parametrize('setup',
                             [0, 1, 2, 3, 4])
    def test_cutoff(self, setup):
        """Test that the dictionary of cutoffs is correctly constructed"""
        sdf = read_sdf('45_mirror', 2)
        staggered_setups = [2, 3, 4]
        if setup in staggered_setups:
            sdf_x = read_sdf('45_mirror_x', 2)
            sdf_y = read_sdf('45_mirror_y', 2)
            sdfs = (sdf, sdf_x, sdf_y)
        else:
            sdfs = sdf
        grid = sdf.grid
        x, y = grid.dimensions

        h_x = x.spacing
        h_y = y.spacing
        zero = sp.core.numbers.Zero()

        # Need to have the dimensions, so done inside test rather than
        # parameterization
        if setup == 0:
            cutoff = None
            answer = {(zero, zero): 0.5}
        elif setup == 1:
            cutoff = {(zero, zero): 0.5, (h_x/2, zero): 0, (zero, h_y/2): 0}
            answer = {(zero, zero): 0.5, (h_x/2, zero): 0, (zero, h_y/2): 0}
        elif setup == 2:
            cutoff = None
            answer = {(zero, zero): 0.5, (h_x/2, zero): 0.5,
                      (zero, h_y/2): 0.5}
        elif setup == 3:
            cutoff = {(h_x/2, zero): 0, (zero, h_y/2): 0}
            answer = {(zero, zero): 0.5, (h_x/2, zero): 0,
                      (zero, h_y/2): 0}
        elif setup == 4:
            cutoff = {(zero, zero): 0, (h_x/2, zero): 0,
                      (zero, h_y/2): 0}
            answer = {(zero, zero): 0, (h_x/2, zero): 0,
                      (zero, h_y/2): 0}

        bg = BoundaryGeometry(sdfs, cutoff=cutoff)

        assert bg.cutoff == frozendict(answer)
