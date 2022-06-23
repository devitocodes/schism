"""
Objects concerning the geometry of the surface in an immersed boundary problem.
"""

import numpy as np
import devito as dv

from schism.utils.environment import get_geometry_feps
from functools import cached_property

__all__ = ['BoundaryGeometry']

_feps = get_geometry_feps()


class BoundaryGeometry:
    """
    Geometry for an immersed boundary. Contains information on the boundary
    position relative to the finite difference grid, alongside other metadata.

    Parameters
    ----------
    sdf : devito Function
        A function whose data contains the signed distance function of the
        boundary discretized to the finite difference grid. Note that the
        space order of this function is used for the operators used to
        calculate normal vectors to the boundary.

    Attributes
    ----------
    sdf : Function
        The signed distance function used to generate the boundary geometry
    grid : Grid
        The grid to which the geometry is attached
    n : VectorFunction
        Symbolic expression for the outward unit normal
    interior_mask : ndarray
        Boolean mask for interior points
    boundary_mask : ndarray
        Boolean mask for boundary points
    b_mask_1D : tuple
        Tuple of boolean masks for boundary points. When using 1D stencils,
        only the innermost point should be used for extrapolation
    boundary_points : tuple
        Boundary point indices as a length N tuple of arrays
    positions : tuple
        Distances from each boundary point to the boundary as a length N tuple
        of arrays.
    n_boundary_points : int
        Number of boundary points associated with the geometry
    """

    def __init__(self, sdf):
        self._sdf = sdf
        self._grid = self.sdf.grid

        self._get_boundary_normals()
        self._get_boundary_points()
        self._get_interior_mask()

    def _get_boundary_normals(self):
        """Get normal direction and distance from each point to the boundary"""
        # Size of padding (required to avoid odd normals at grid edge)
        pad = int(self.sdf.space_order//2)

        padded_grid = self._padded_grid(pad)
        # Create a padded version of the signed distance function
        pad_sdf = dv.Function(name='pad_sdf', grid=padded_grid,
                              space_order=self.sdf.space_order)
        pad_sdf.data[:] = np.pad(self.sdf.data, (pad,), 'edge')

        # Normal vectors
        n = dv.VectorFunction(name='n', grid=padded_grid,
                              space_order=self.sdf.space_order,
                              staggered=(None, None, None))

        # Negative here as normals will be outward
        normal_eq = dv.Eq(n, -dv.grad(pad_sdf))
        dv.Operator(normal_eq, name='normals')()

        self._normals = dv.VectorFunction(name='n', grid=self.sdf.grid,
                                          space_order=self.sdf.space_order,
                                          staggered=(None, None, None))

        # Trim the padding off
        slices = tuple([slice(pad, -pad) for dim in self.grid.dimensions])
        for i in range(len(self.grid.dimensions)):
            self._normals[i].data[:] = n[i].data[slices]

    def _padded_grid(self, pad):
        """
        Return a grid with an additional M/2 nodes of padding on each side vs
        the main grid.
        """
        # Get updated origin position
        origin = np.array([value for value in self.grid.origin_map.values()])
        origin -= pad*np.array(self.grid.spacing)
        origin = tuple(origin)

        # Get size and extent of the padded grid
        extent = np.array(self.grid.extent, dtype=float)
        extent += 2*pad*np.array(self.grid.spacing)
        extent = tuple(extent)

        shape = np.array(self.grid.shape)+2*pad
        shape = tuple(shape)

        grid = dv.Grid(shape=shape, extent=extent, origin=origin,
                       dimensions=self.grid.dimensions)
        return grid

    def _get_boundary_points(self):
        """
        Get indices of boundary points and their distances to the boundary.
        """
        spacing = self.grid.spacing
        dims = self.grid.dimensions
        max_dist = np.sqrt(sum([(inc/2)**2 for inc in spacing]))

        # Normalise positions by grid increment
        positions = [self.n[i].data*self.sdf.data/spacing[i]
                     for i in range(len(spacing))]

        # Allows some slack to cope with floating point error
        masks = [np.abs(positions[i]) <= 0.5 + _feps
                 for i in range(len(spacing))]
        mask = np.logical_and.reduce(masks)

        close = np.abs(self.sdf.data) <= max_dist

        self._boundary_mask = np.logical_and(close, mask)

        self._boundary_points = np.where(self.boundary_mask)

        # May want to be an array rather than a tuple of arrays in future
        self._positions = tuple([positions[i][self.boundary_points]
                                 for i in range(len(dims))])

        dense_pos = [np.zeros(self.grid.shape) for dim in dims]
        for dim in range(len(dims)):
            dense_pos[dim][self.boundary_points] = self.positions[dim]

        self._dense_pos = tuple(dense_pos)

        self._n_boundary_points = self._boundary_points[0].shape[0]

    def _get_interior_mask(self):
        """Get the mask for interior points"""
        not_boundary = np.logical_not(self.boundary_mask)
        interior = self.sdf.data > 0
        self._interior_mask = np.logical_and(interior, not_boundary)

    @property
    def sdf(self):
        """The signed distance function"""
        return self._sdf

    @property
    def grid(self):
        """The grid on which the geometry is defined"""
        return self._grid

    @property
    def interior_mask(self):
        """Boolean mask for interior points"""
        return self._interior_mask

    @property
    def boundary_mask(self):
        """Boolean mask for boundary points"""
        return self._boundary_mask

    @property
    def boundary_points(self):
        """Boundary points"""
        return self._boundary_points

    @property
    def positions(self):
        """Relative offsets corresponding with each boundary point"""
        return self._positions

    @property
    def dense_pos(self):
        """Dense version of positions"""
        return self._dense_pos

    @property
    def n_boundary_points(self):
        """Number of boundary points associated with this geometry"""
        return self._n_boundary_points

    @property
    def n(self):
        """Boundary unit normal vector"""
        return self._normals

    @cached_property
    def b_mask_1D(self):
        """
        Tuple of boolean masks for boundary points. When using 1D stencils,
        only the innermost point should be used for extrapolation.
        """
        ndims = len(self.grid.dimensions)
        masks = []
        for dim in range(ndims):
            shift_plus = np.full(self.grid.shape, True, dtype=bool)
            shift_minus = np.full(self.grid.shape, True, dtype=bool)

            plus_slices = tuple([slice(None) if d != dim else slice(1, None)
                                 for d in range(ndims)])
            min_slices = tuple([slice(None) if d != dim else slice(0, -1)
                                for d in range(ndims)])

            shift_plus[min_slices] = self.interior_mask[plus_slices]
            shift_minus[plus_slices] = self.interior_mask[min_slices]

            interior_adjacent = np.logical_or(shift_plus, shift_minus)
            masks.append(np.logical_and(interior_adjacent, self.boundary_mask))

        self._b_mask_1D = tuple(masks)
        return self._b_mask_1D
