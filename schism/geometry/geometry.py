"""
Objects concerning the geometry of the surface in an immersed boundary problem.
"""

import numpy as np
import devito as dv

from schism.utils.environment import get_geometry_feps
from functools import cached_property
from devito.tools.data_structures import frozendict

__all__ = ['BoundaryGeometry']

_feps = get_geometry_feps()


class BoundaryGeometry:
    """
    Geometry for an immersed boundary. Contains information on the boundary
    position relative to the finite difference grid, alongside other metadata.

    Parameters
    ----------
    sdf : devito Function or list of Function
        A function whose data contains the signed distance function of the
        boundary discretized to the finite difference grid. Note that the
        space order of this function is used for the operators used to
        calculate normal vectors to the boundary. For staggered grids,
        multiple signed distance functions should be supplied as a list. The
        subgrids to which they pertain are determined from the staggering of
        each Function. Note that at least one SDF must be associated with the
        unstaggered grid.

    cutoff : dict, optional
        The cutoff determining how close points in each subgrid can be to the
        boundary before being excluded from the numerical scheme. Dict should
        be formatted {origin: cutoff}. If none supplied then defaults to 0.5
        (half a grid increment).

    Attributes
    ----------
    sdf : dict
        Signed distance functions for each subgrid
    sdf_ref : Function
        The unstaggered SDF
    cutoff : dict
        The cutoff determining how close points for each subgrid can be to the
        boundary.
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
    dense_pos : tuple
        Boundary position according to the sdf
    n_boundary_points : int
        Number of boundary points associated with the geometry
    """

    def __init__(self, sdf, cutoff=None):
        self._process_sdfs(sdf)
        self._get_cutoff(cutoff)
        self._get_boundary_normals()
        self._get_boundary_points()
        self._get_interior_mask()

    def _process_sdfs(self, sdf):
        """Process the SDF argument according to how it is supplied"""
        if isinstance(sdf, dv.Function):
            # Check that this is on the unstaggered grid
            if np.any([stagger != 0 for stagger in sdf.origin]):
                raise ValueError("Single SDF not on unstaggered grid")
            self._sdf = frozendict({sdf.origin: sdf})
            self._sdf_ref = sdf  # Attribute for the reference sdf
            # This reference is used for normal calculation etc
            self._grid = self.sdf_ref.grid
        else:  # Iterable of sdfs supplied
            sdf_dict = {}
            self._grid = sdf[0].grid
            ref_sdf_set = False
            for func in sdf:
                if func.grid != self.grid:
                    raise ValueError("SDFs do not share a grid")
                sdf_dict[func.origin] = func
                if np.all([stagger == 0 for stagger in func.origin]):
                    self._sdf_ref = func
                    ref_sdf_set = True
            self._sdf = frozendict(sdf_dict)
            if not ref_sdf_set:
                raise ValueError("No SDF supplied on unstaggered grid")

    def _get_cutoff(self, cutoff):
        """Get the cutoff for each subgrid"""
        if cutoff is None:
            self._cutoff = frozendict({origin: 0.5 for origin in self.sdf})
        else:
            # Get sdf keys not in cutoff
            leftover_keys = set(self.sdf.keys()) - set(cutoff.keys())
            default_cutoff = {key: 0.5 for key in leftover_keys}
            # Append them to cutoff
            all_cutoffs = cutoff.update(default_cutoff)
            # Turn this into a frozendict
            self._cutoff = frozendict(all_cutoffs)

    def _get_boundary_normals(self):
        """Get normal direction and distance from each point to the boundary"""
        # Size of padding (required to avoid odd normals at grid edge)
        pad = int(self.sdf_ref.space_order//2)

        padded_grid = self._padded_grid(pad)
        # Create a padded version of the signed distance function
        pad_sdf = dv.Function(name='pad_sdf', grid=padded_grid,
                              space_order=self.sdf_ref.space_order)
        pad_sdf.data[:] = np.pad(self.sdf_ref.data, (pad,), 'edge')

        # Normal vectors
        n = dv.VectorFunction(name='n', grid=padded_grid,
                              space_order=self.sdf_ref.space_order,
                              staggered=(None, None, None))

        # Negative here as normals will be outward
        normal_eq = dv.Eq(n, -dv.grad(pad_sdf))
        dv.Operator(normal_eq, name='normals')()

        self._normals = dv.VectorFunction(name='n', grid=self.sdf_ref.grid,
                                          space_order=self.sdf_ref.space_order,
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
        max_dist = np.sqrt(sum([(inc/2)**2 for inc in spacing]))

        # Normalise positions by grid increment
        positions = [self.n[i].data*self.sdf_ref.data/spacing[i]
                     for i in range(len(spacing))]

        # Allows some slack to cope with floating point error
        masks = [np.abs(positions[i]) <= 0.5 + _feps
                 for i in range(len(spacing))]
        mask = np.logical_and.reduce(masks)

        # Needed to allow sdfs that are flat after some radius
        close = np.abs(self.sdf_ref.data) <= max_dist

        self._boundary_mask = np.logical_and(close, mask)

        self._dense_pos = tuple(positions)

        self._boundary_points = np.where(self.boundary_mask)

        self._n_boundary_points = np.count_nonzero(self.boundary_mask)

    def _get_interior_mask(self):
        """Get the mask for interior points"""
        interior_masks = {}
        dims = self.grid.dimensions
        spacing = self.grid.spacing
        for origin in self.sdf:
            cutoff = self.cutoff[origin]
            max_dist = np.sqrt(sum([(spacing[dim]*cutoff)**2
                                    for dim in range(len(dims))]))
            # Convert h_x/2 to 0.5 etc
            stagger = [float(origin[i].subs(dims[i].spacing, 1))
                       for i in range(len(dims))]
            # Allows some slack to cope with floating point error
            masks = [np.abs(self.dense_pos[i] - stagger[i]) > cutoff + _feps
                     for i in range(len(dims))]

            # Needed to allow sdfs that are flat after some radius
            far = np.abs(self.sdf[origin].data) > max_dist
            masks.append(far)
            # Points outside the cutoff
            not_excluded = np.logical_or.reduce(masks)
            # On the interior according to the SDF
            interior = self.sdf[origin].data > 0
            interior_masks[origin] = np.logical_and(interior, not_excluded)

        self._interior_mask = frozendict(interior_masks)

    @property
    def sdf(self):
        """The signed distance functions"""
        return self._sdf

    @property
    def sdf_ref(self):
        """The unstaggered signed distance function (the reference grid)"""
        return self._sdf_ref

    @property
    def cutoff(self):
        """
        The cutoff point at which points are considered too close to the
        boundary. Points closer than this will be excluded from the numerical
        scheme.
        """
        return self._cutoff

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
        interior_mask = self.interior_mask[self.sdf_ref.origin]
        masks = []
        for dim in range(ndims):
            shift_plus = np.full(self.grid.shape, False, dtype=bool)
            shift_minus = np.full(self.grid.shape, False, dtype=bool)

            plus_slices = tuple([slice(None) if d != dim else slice(1, None)
                                 for d in range(ndims)])
            min_slices = tuple([slice(None) if d != dim else slice(0, -1)
                                for d in range(ndims)])

            shift_plus[min_slices] = interior_mask[plus_slices]
            shift_minus[plus_slices] = interior_mask[min_slices]

            interior_adjacent = np.logical_or(shift_plus, shift_minus)
            masks.append(np.logical_and(interior_adjacent, self.boundary_mask))

        self._b_mask_1D = tuple(masks)
        return self._b_mask_1D
