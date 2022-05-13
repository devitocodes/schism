"""
Objects concerning the geometry of the surface in an immersed boundary problem.
"""

import numpy as np
import devito as dv


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
        pad = int(self.sdf.order//2)

        padded_grid = self._padded_grid(pad)

        # Normal vectors
        n = dv.VectorFunction(name='n', grid=padded_grid,
                              space_order=self.sdf.order,
                              staggered=(None, None, None))

        normal_eq = dv.Eq(n, dv.div(self.sdf))
        dv.Operator(normal_eq, name='normals')()

        self._normals = dv.VectorFunction(name='n', grid=self.sdf.grid,
                                          space_order=self.sdf.order,
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
        extent = np.array(self.grid.extent)
        extent += 2*pad*np.array(self.grid.spacing)
        extent = tuple(extent)

        shape = np.array(self.grid.shape)+2*pad
        shape = tuple(shape)

        grid = dv.Grid(shape=shape, extent=extent, origin=origin)
        return grid

    def _get_boundary_points(self):
        """
        Get indices of boundary points and their distances to the boundary.
        """
        spacing = self.grid.spacing

        positions = [self.n[i].data*self.sdf.data for i in range(len(spacing))]

        masks = [np.abs(positions[i]) <= spacing[i]/2
                 for i in range(len(spacing))]

        self._boundary_mask = np.logical_and.reduce(masks)

        self._boundary_points = np.where(self.boundary_mask)

        self._positions = np.array(positions)[self.boundary_points]

    def _get_interior_mask(self):
        """Get the mask for interior points"""
        # TODO: Finish this. Should just be sdf is positive and not a boundary
        # point

    @property
    def sdf(self):
        """The signed distance function"""
        return self._sdf

    @property
    def grid(self):
        """The grid on which the geometry is defined"""
        return self._grid

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
    def n(self):
        """Boundary unit normal vector"""
        return self._normals
