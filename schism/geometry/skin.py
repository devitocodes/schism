"""Geometry for the skin of modified operators surrounding the boundary"""

import numpy as np
import sympy as sp


def stencil_footprint(deriv):
    """Get the stencil footprint for a derivative"""
    dims = deriv.expr.space_dimensions
    s_o = deriv.expr.space_order
    spans = []
    for dim in dims:
        if dim in deriv.dims:
            spans.append(np.arange(-s_o//2, s_o//2+1))
        else:
            spans.append(np.zeros(1))
    msh = np.meshgrid(*spans, indexing='ij')
    stack = [m.flatten() for m in msh]
    return np.vstack(stack).astype(int)


class ModifiedSkin:
    """
    The skin of boundary-adjacent points where stencils are to be modified.

    Parameters
    ----------
    deriv : Derivative
        The Devito derivative expression
    geometry : BoundaryGeometry
        The geometry of the boundary

    Attributes
    ----------
    deriv : Derivative
        The Devito derivative expression
    geometry : BoundaryGeometry
        The geometry of the boundary
    points : ndarray
        Points at which modified operators are required
    npts : int
        Number of points at which modified operators are required
    """
    def __init__(self, deriv, geometry):
        self._deriv = deriv
        self._geometry = geometry

        self._get_points()

    def _get_points(self):
        """Get the points at which modified operators are required"""
        bp = np.array(self.geometry.boundary_points)
        fp = stencil_footprint(self.deriv)

        # Get all the modified points from stencil span and boundary points
        mp = bp[:, :, np.newaxis] + fp[:, np.newaxis, :]
        mp = mp.reshape((bp.shape[0], -1))

        # Filter duplicates
        mp = np.unique(mp, axis=1)

        # Trim off points outside domain
        grid = self.geometry.grid
        for dim in range(mp.shape[0]):
            mask = np.logical_and(mp[dim] >= 0, mp[dim] < grid.shape[dim])
            mp = mp[:, mask]

        # Get the intersection with the interior
        all_points = tuple([mp[i] for i in range(mp.shape[0])])

        # Needs to use the interior mask of the target subgrid
        deriv_stagger = []
        for d in range(len(grid.dimensions)):
            try:
                deriv_stagger.append(self.deriv.x0[grid.dimensions[d]]
                                     - grid.dimensions[d])
            except KeyError:
                deriv_stagger.append(sp.core.numbers.Zero())
        origin = tuple(deriv_stagger)

        interior_mask = self.geometry.interior_mask[origin][all_points]
        mp = mp[:, interior_mask]

        self._mod_points = tuple([mp[i] for i in range(mp.shape[0])])
        self._npts = self.points[0].shape[0]

    @property
    def deriv(self):
        """The Devito derivative expression"""
        return self._deriv

    @property
    def geometry(self):
        """The geometry of the boundary"""
        return self._geometry

    @property
    def points(self):
        """Points where the derivative stencil will be modified"""
        return self._mod_points

    @property
    def npts(self):
        """Number of points where the derivative stencil will be modified"""
        return self._npts
