"""Geometry for the skin of modified operators surrounding the boundary"""

import numpy as np


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
    msh = np.meshgrid(*spans, indexing='xy')
    stack = [m.flatten() for m in msh]
    return np.vstack(stack)


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
    """
    def __init__(self, deriv, geometry):
        self._deriv = deriv
        self._geometry = geometry

        self._get_points()

    def _get_points(self):
        """Get the points at which modified operators are required"""
        b_points = self.geometry.boundary_points

    @property
    def deriv(self):
        """The Devito derivative expression"""
        return self._deriv

    @property
    def geometry(self):
        """The geometry of the boundary"""
        return self._geometry
