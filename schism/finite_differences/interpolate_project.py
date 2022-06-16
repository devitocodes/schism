"""
Objects for fitting of polynomial basis funtions and projection onto the
interior stencil.
"""

import sympy as sp
import numpy as np

from devito.tools.data_structures import frozendict
from schism.basic import row_from_expr
from schism.geometry.support_region import get_points_and_oob


class MultiInterpolant:
    """
    A group of interpolants fitting function values and boundary conditions
    for various stencil spans or polynomial orders.

    Attributes
    ----------
    interpolants : tuple
        Interpolant objects attached to the MultiInterpolant

    Methods
    -------
    add(interpolant)
        Add an interpolant
    project(projections)
        Project the interpolants onto the interior stencil using their
        respective projections.
    """
    def __init__(self):
        self._interpolants = []

    def add(self, interpolant):
        """Add an Interpolant"""
        # FIXME: Will need to do a bunch of other stuff in due course
        self._interpolants.append(interpolant)

    @property
    def interpolants(self):
        """Interpolant objects attached to the MultiInterpolant"""
        return tuple(self._interpolants)


class MultiProjection:
    """
    A group of projections projecting fitted polynomials onto the interior
    derivative stencil.

    Attributes
    ----------
    projections : tuple
        The Projection objects attached to the MultiProjection

    Methods
    -------
    add(projection)
        Add a Projection
    """
    def __init__(self):
        self._projections = []

    def add(self, projection):
        """Add a Projection"""
        # FIXME: will need to do a bunch of other stuff in due course
        self._projections.append(projection)

    @property
    def projections(self):
        """Projection objects attached to the MultiProjection"""
        return tuple(self._projections)


class Interpolant:
    """
    Encapsulates the fitting of a set of polynomial basis functions to interior
    values and boundary conditions.

    Parameters
    ----------
    support : SupportRegion
        The support region used to fit this basis
    group : ConditionGroup
        The group of boundary conditions which the function will be fitted to
    basis_map : dict
        Mapping between functions and approximating basis functions
    skin : ModifiedSkin
        The boundary-adjacent skin of points in which modified stencils are
        required.
    """
    def __init__(self, support, group, basis_map, skin):
        self._support = support
        self._group = group
        self._basis_map = basis_map
        self._skin = skin
        self._geometry = self.skin.geometry
        self._get_interior_vector()
        self._get_interior_matrix()
        self._get_stencil_points()
        self._get_interior_mask()
        self._get_boundary_mask()
        self._get_boundary_matrices()

    def _get_interior_vector(self):
        """
        Generate the vector of interior points corresponding with the support
        region.
        """
        footprint_map = self.support.footprint_map
        # Needs to be an index notation like f[t, x-1, y+1]
        vec = []
        # Loop over functions
        for func in self.group.funcs:
            # Get the space and time dimensions of that function
            t = func.time_dim
            dims = func.space_dimensions
            footprint = footprint_map[func]
            # Create entry for each point in the support region
            for point in range(len(footprint[0])):
                space_ind = [dims[dim]+footprint[dim][point]
                             for dim in range(len(dims))]
                ind = (t,) + tuple(space_ind)
                vec.append(func[ind])

        # Make this a sympy Matrix
        self._interior_vector = sp.Matrix(vec)

    def _get_interior_matrix(self):
        """
        Generate a master matrix for interior points corresponding with the
        support region.
        """
        submats = []  # Submatrices to be concatenated
        for func in self.group.funcs:
            basis = self.basis_map[func]
            row_func = row_from_expr(basis.expr, self.group.funcs,
                                     self.basis_map)
            submats.append(row_func(*self.support.footprint_map[func]))
        # Will need to do an axis swap in due course
        self._interior_matrix = np.concatenate(submats, axis=1)

    def _get_stencil_points(self):
        """
        Get the stencil points associated with each modified point and a mask
        indicating where these are out of bounds.
        """
        sten_pts = {}
        oob = {}
        for func in self.group.funcs:
            support_points = self.support.footprint_map[func]
            sten_pts[func], oob[func] = get_points_and_oob(support_points,
                                                           self.skin)
        self._stencil_points = frozendict(sten_pts)
        self._oob = frozendict(oob)

    def _get_interior_mask(self):
        """
        For each modified point, create a mask for points in its associated
        support region indicating interior points.
        """
        submasks = []
        for func in self.group.funcs:
            ndims = len(func.space_dimensions)

            interior_msk = np.zeros(self.support.npts_map[func],
                                    self.skin.npts, dtype=bool)
            interior_msk[self._oob] = True

            in_bounds = np.logical_not(self._oob)
            # Stencil points within bounds
            pts_ib = tuple([self._stencil_points[dim][in_bounds]
                            for dim in range(ndims)])
            interior_msk[in_bounds] = self.geometry.interior_mask[pts_ib]

            submasks.append(interior_msk)
        # (0 axis is support region points)
        self._interior_mask = np.concatenate(submasks)

    def _get_boundary_mask(self):
        """
        For each modified point, create a mask for points in its associated
        support region of the function with the largest support region
        indicating boundary points.
        """
        # Get the function with the largest support region
        func = self.support.max_span_func

        ndims = len(func.space_dimensions)

        boundary_msk = np.zeros(self.support.npts_map[func],
                                self.skin.npts, dtype=bool)
        boundary_msk[self._oob[func]] = False

        in_bounds = np.logical_not(self._oob[func])
        # Stencil points within bounds
        pts_ib = tuple([self._stencil_points[func][dim][in_bounds]
                        for dim in range(ndims)])
        boundary_msk[in_bounds] = self.geometry.boundary_mask[pts_ib]

        # (0 axis is support region points)
        self._boundary_mask = boundary_msk

    def _get_boundary_matrices(self):
        """Get the submatrices for each boundary condition to be applied"""
        # First need all the boundary points
        # Get the function with the largest support region
        func = self.support.max_span_func

        # Number of points in the support region and number of modified points
        nsten = self.support.npts_map[func]
        nmod = self.skin.npts
        ndims = len(func.space_dimensions)

        # _stencil_points refers to absolute indices
        bp = tuple([self._stencil_points[func][dim][self.boundary_mask]
                    for dim in range(ndims)])
        # Use these to access boundary offset from reference node
        # FIXME: All of these are tuples of ndims x npts
        pos = [self.geometry.dense_pos[dim][bp] for dim in range(ndims)]
        footprint = self.support.footprint_map[func]
        # Get the stencil footprint and cast it to the same size
        for dim in range(ndims):
            pos[dim] += np.broadcast_to(footprint[dim][:, np.newaxis],
                                        (nsten, nmod))[self.boundary_mask]

        pos = tuple(pos)

        submats = []
        for bc in self.group.conditions:
            submat = np.zeros(nsten, nmod)
            rowfunc = row_from_expr(bc, self.group.funcs, self.basis_map)
            submat[self.boundary_mask] = rowfunc(*pos)
            submats.append(submat)

        self._boundary_matrices = tuple(submats)

    @property
    def support(self):
        """The support region used to fit the basis"""
        return self._support

    @property
    def group(self):
        """The boundary condition group"""
        return self._group

    @property
    def basis_map(self):
        """Mapping between functions and approximating basis functions"""
        return self._basis_map

    @property
    def skin(self):
        """
        The boundary-adjacent skin of points in which modified stencils are
        required.
        """
        return self._skin

    @property
    def geometry(self):
        """The geometry of the boundary"""
        return self._geometry

    @property
    def interior_vector(self):
        """The vector of interior points corresponding to the support region"""
        return self._interior_vector

    @property
    def interior_matrix(self):
        """
        The master matrix of interior points corresponding to the support
        region.
        """
        return self._interior_matrix

    @property
    def interior_mask(self):
        """
        Mask for the master matrix. If True, then the row in the interior
        matrix corresponds with a stencil point on the interior
        """
        return self._interior_mask

    @property
    def boundary_mask(self):
        """Mask indicating boundary points"""
        return self._boundary_mask
