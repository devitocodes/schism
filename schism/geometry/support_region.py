"""
Objects relating to the support region of the basis used to impose the boundary
conditions.
"""

import numpy as np
from devito.tools.data_structures import frozendict


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
    """
    def __init__(self, basis_map, radius_map):
        self._basis_map = basis_map
        self._radius_map = radius_map

        self._get_footprint()

    def _get_footprint(self):
        """Get the stencil footprint for each function"""
        footprints = {}
        if self.basis_map.keys() != self.radius_map.keys():
            # Should never end up here
            raise ValueError("Mismatch in functions supplied")

        for func in self.basis_map:
            if self.basis_map[func].dims == func.grid.space_dimensions:
                # N-D basis so N-D support region
                footprint = self._get_circle_support(func)
            else:
                if len(self.basis_map[func].dims) != 1:
                    # Should never end up here
                    raise ValueError("Basis neither 1D or N-D")
                # 1D basis
                footprint = self._get_linear_support(func)
            footprints[func] = footprint
        self._footprint_map = frozendict(footprints)

    def _get_circle_support(self, func):
        """Get the footprint of a circular support region"""
        # Essentially makes a square then cookie-cutters it
        radius = self.radius_map[func]
        dims = func.grid.space_dimensions
        ndims = len(dims)
        # Make a meshgrid of indices (of int type)
        msh = np.meshgrid(*[np.arange(-radius, radius+1, dtype=int)
                            for dim in dims],
                          indexing='xy')
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
        for dim in func.grid.space_dimensions:
            if dim in basis.dims:
                footprint.append(np.arange(-radius, radius+1))
            else:  # No offset in other dimensions
                footprint.append(np.zeros(1+2*radius))
        return tuple(footprint)

    @property
    def basis_map(self):
        """Mapping between functions and respective basis functions"""
        return self._basis_map

    @property
    def radius_map(self):
        """Mapping between functions and the radius of their basis"""
        return self._radius_map
