"""
Objects relating to the support region of the basis used to impose the boundary
conditions.
"""

import numpy as np
from devito.tools.data_structures import frozendict
from functools import reduce


def get_points_and_oob(support_points, modified_points, geometry):
    """
    Get the points used by each stencil and a mask indicating where these are
    out of bounds.

    Parameters
    ----------
    support_points : tuple
        Points in the support region of the stencil
    modified_points : tuple
        Points where modified stencils are required
    geometry : BoundaryGeometry
        Geometry of the boundary. Used to obtain the Grid.

    Returns
    -------
    points : tuple
        Points accessed by the stencil when applied at the modified points
    oob : ndarray
        Boolean mask for points. True where points are out of bounds
    """
    grid = geometry.grid
    ndims = len(grid.dimensions)
    points = tuple([support_points[dim][:, np.newaxis]
                    + modified_points[dim][np.newaxis, :]
                    for dim in range(ndims)])

    # Out of bounds points
    oob = [np.logical_or(points[dim] < 0, points[dim] >= grid.shape[dim])
           for dim in range(ndims)]

    # If a point is out of bounds in any dimension, then label as oob
    oob_msk = reduce(np.logical_or, oob)

    return points, oob_msk


class SupportRegion:
    """
    The support region for a set of basis functions.

    Parameters
    ----------
    basis_map : dict
        Mapping between functions and their respective basis functions
    radius_map : dict
        Mapping between functions and the radius of their basis. Note that this
        is not a true radius, so much as a convenient measure of extent
        measured in grid increments.

    Attributes
    ----------
    footprint_map : dict
        Mapping between functions and the points within their support region
    npts_map : dict
        Mapping between functions and the number of points within their support
        region.
    max_span_func : Function
        The function with the largest span

    Methods
    -------
    expand_radius(inc)
        Return a support region with an expanded radius
    """
    def __init__(self, basis_map, radius_map):
        self._basis_map = basis_map
        self._radius_map = radius_map
        self._max_span_func = max(self.radius_map, key=self.radius_map.get)

        self._get_footprint()

    def _get_footprint(self):
        """Get the stencil footprint for each function"""
        footprints = {}
        npts_map = {}
        if self.basis_map.keys() != self.radius_map.keys():
            # Should never end up here
            raise ValueError("Mismatch in functions supplied")

        for func in self.basis_map:
            if self.basis_map[func].dims == func.space_dimensions:
                # N-D basis so N-D support region
                footprint = self._get_circle_support(func)
            else:
                if len(self.basis_map[func].dims) != 1:
                    # Should never end up here
                    raise ValueError("Basis neither 1D or N-D")
                # 1D basis
                footprint = self._get_linear_support(func)
            footprints[func] = footprint
            npts_map[func] = footprint[0].shape[0]
        self._footprint_map = frozendict(footprints)
        self._npts_map = frozendict(npts_map)

    def _get_circle_support(self, func):
        """Get the footprint of a circular support region"""
        # Essentially makes a square then cookie-cutters it
        radius = self.radius_map[func]
        dims = func.space_dimensions
        ndims = len(dims)
        # Make a meshgrid of indices (of int type)
        # Indexing type changes order of points but not points overall
        # 'ij' results in most logical ordering however
        msh = np.meshgrid(*[np.arange(-radius, radius+1, dtype=int)
                            for dim in dims],
                          indexing='ij')
        # Mask it by radius
        mask = np.sqrt(sum(msh[i]**2 for i in range(ndims))) < radius + 0.5
        # Do np.where to get meshgrid indices
        locs = np.where(mask)
        # Use indices to get the physical indices from the meshgrid
        footprint = [msh[i][locs].flatten() for i in range(ndims)]
        # Return these as a tuple of arrays
        return tuple(footprint)

    def _get_linear_support(self, func):
        """Get the footprint of a 1D support region"""
        footprint = []
        basis = self.basis_map[func]
        radius = self.radius_map[func]
        for dim in func.space_dimensions:
            if dim in basis.dims:
                footprint.append(np.arange(-radius, radius+1, dtype=int))
            else:  # No offset in other dimensions
                footprint.append(np.zeros(1+2*radius, dtype=int))
        return tuple(footprint)

    def expand_radius(self, inc):
        """
        Return another support region with radius expanded by the increment
        specified

        Parameters
        ----------
        inc : int
            The amount by which the radius should be incremented

        Returns
        -------
        expanded : SupportRegion
            The expanded support region
        """
        new_radius_map = {func: rad + inc
                          for func, rad in self.radius_map.items()}
        return self.__class__(self.basis_map, new_radius_map)

    @property
    def basis_map(self):
        """Mapping between functions and respective basis functions"""
        return self._basis_map

    @property
    def radius_map(self):
        """Mapping between functions and the radius of their basis"""
        return self._radius_map

    @property
    def max_span_func(self):
        """The function with the largest support region span"""
        return self._max_span_func

    @property
    def footprint_map(self):
        """
        Mapping between functions and the footprint of their support region.
        """
        return self._footprint_map

    @property
    def npts_map(self):
        """
        Mapping between functions and the number of points in their support
        regions.
        """
        return self._npts_map
