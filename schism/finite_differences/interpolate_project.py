"""
Objects for fitting of polynomial basis funtions and projection onto the
interior stencil.
"""

import sympy as sp
import numpy as np
import devito as dv

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
        self._assemble_matrix()
        self._check_rank()
        self._get_pinv()

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
            if func.staggered is None:  # No stagger
                stagger = tuple([0 for dim in func.space_dimensions])
            elif type(func.staggered) == dv.SpaceDimension:
                stagger = tuple([0.5 if dim == func.staggered else 0.
                                 for dim in func.space_dimensions])
            else:  # Staggering in multiple directions
                stagger = tuple([0.5 if dim in func.staggered else 0.
                                 for dim in func.space_dimensions])
            basis = self.basis_map[func]
            row_func = row_from_expr(basis.expr, self.group.funcs,
                                     self.basis_map)
            # Support points with stagger
            s_p = tuple([self.support.footprint_map[func][dim] + stagger[dim]
                         for dim in range(len(func.space_dimensions))])
            submats.append(row_func(*s_p))
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

            interior_msk = np.zeros((self.support.npts_map[func],
                                     self.skin.npts), dtype=bool)
            interior_msk[self._oob[func]] = True

            in_bounds = np.logical_not(self._oob[func])
            # Stencil points within bounds
            pts_ib = tuple([self._stencil_points[func][dim][in_bounds]
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

        boundary_msk = np.zeros((self.support.npts_map[func],
                                 self.skin.npts), dtype=bool)
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
        nterms = sum([self.basis_map[f].nterms for f in self.group.funcs])
        ndims = len(func.space_dimensions)

        # _stencil_points refers to absolute indices
        bp = tuple([self._stencil_points[func][dim][self.boundary_mask]
                    for dim in range(ndims)])
        # Use these to access boundary offset from reference node
        pos = [self.geometry.dense_pos[dim][bp] for dim in range(ndims)]
        footprint = self.support.footprint_map[func]
        # Get the stencil footprint and cast it to the same size
        for dim in range(ndims):
            pos[dim] += np.broadcast_to(footprint[dim][:, np.newaxis],
                                        (nsten, nmod))[self.boundary_mask]

        pos = tuple(pos)

        submats = []
        for bc in self.group.conditions:
            # Note, the first two axes will need swapping in due course
            submat = np.zeros((nterms, nsten, nmod))
            # Substitute the basis funcs into the boundary condition
            expr = bc.sub_basis(self.basis_map)
            rowfunc = row_from_expr(expr, self.group.funcs, self.basis_map)
            submat[:, self.boundary_mask] = rowfunc(*pos)
            submats.append(submat)

        self._boundary_matrices = tuple(submats)

    def _assemble_matrix(self):
        """
        Assemble the matrix from the masked interior matrix and the boundary
        matrices.
        """
        nmod = self.skin.npts
        # Initialise empty interior matrix
        interior = np.zeros(self.interior_matrix.shape+(nmod,))
        interior_bcst = np.broadcast_to(self.interior_matrix[..., np.newaxis],
                                        self.interior_matrix.shape+(nmod,))
        interior[:, self.interior_mask] = interior_bcst[:, self.interior_mask]

        matrix = np.concatenate((interior,)+self.boundary_matrices,
                                axis=1)
        # Need to reverse the order of axes at this point
        # Linalg exprects the stacking to be on the first axis
        # But we have it on the last
        self._matrix = np.moveaxis(matrix, (0, 1, 2), (2, 1, 0))

    def _check_rank(self):
        """
        Check the ranks of the matrices in the stack and assemble a mask for
        where it is not equal to the number of terms to solve for.
        """
        self._rank = np.linalg.matrix_rank(self.matrix)
        self._rank_mask = self._rank == self.matrix.shape[0]
        if np.all(self._rank_mask):
            self._is_sufficient_rank = True
        else:
            self._is_sufficient_rank = False

    def _get_pinv(self):
        """
        Get the Moore-Penrose pseudoinverse of the matrices in the stack which
        have rank sufficient to yield a unique pseudoinverse.
        """
        of_rank = self._matrix[self._rank_mask]
        self._pinv = np.linalg.pinv(of_rank)

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

    @property
    def boundary_matrices(self):
        """Boundary matrices"""
        return self._boundary_matrices

    @property
    def matrix(self):
        """The full matrix stack"""
        return self._matrix

    @property
    def rank(self):
        """The rank of each matrix in the stack"""
        return self._rank

    @property
    def rank_mask(self):
        """Mask for points where the matrix inverse is unique"""
        return self._rank_mask

    @property
    def pinv(self):
        """
        The Moore-Penrose pseudoinverse of matrices in the stack for matrices
        which have sufficient rank for this pseudoinverse to be unique.
        """
        return self._pinv
