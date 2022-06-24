"""The Substitution object used to produce a single derivative substitution"""

import numpy as np
import devito as dv

from schism.finite_differences.interpolate_project import MultiInterpolant, \
    MultiProjection, Interpolant, Projection
from schism.geometry.support_region import SupportRegion
from schism.geometry.skin import stencil_footprint
from schism.finite_differences.tools import get_sten_vector
from itertools import chain
from devito.tools.data_structures import frozendict


class Substitution:
    """
    A substitution for a particular derivate expression. Encapsulates the
    modified finite-difference stencils in the vincinity of the boundary.

    Parameters
    ----------
    deriv : Derivative
        The derivate to be substituted
    group : ConditionGroup
        The corresponding group of boundary conditions
    basis_map : frozendict
        Mapping between functions and corresponding basis functions
    strategy : str
        The strategy used when the matrix system has insufficient rank
    skin : ModifiedSkin
        The boundary-adjacent skin in which modified boundary stencils are
        required
    """
    def __init__(self, deriv, group, basis_map, strategy, skin):
        self._deriv = deriv
        self._group = group
        self._basis_map = basis_map
        self._strategy = strategy
        self._skin = skin
        self._geometry = self.skin.geometry

        self._setup_collections()
        self._get_stencils()
        self._setup_weight_funcs()
        self._fill_weight_funcs()

    def _setup_collections(self):
        """
        Set up MultiInterpolant and MultiProjection objects used to collect
        Interpolant objects and their respective Projection objects
        """
        self._interpolants = MultiInterpolant()
        self._projections = MultiProjection()

    def _get_stencils(self):
        """
        Get the stencils by generating the required Interpolant and Projection
        objects
        """
        ndims = len(self.geometry.grid.dimensions)
        radius_map = {func: func.space_order//2 for func in self.basis_map}
        support = SupportRegion(self.basis_map, radius_map)
        interpolant = Interpolant(support, self.group,
                                  self.basis_map, self.skin.geometry,
                                  self.skin.points)
        projection = Projection(self.deriv, self.group, self.basis_map)

        # Loop whilst there are points which don't yet have stencils
        maxloops = 3
        loops = 0  # Catch too many loops
        while not interpolant.all_full_rank:
            if loops >= maxloops:
                raise RuntimeError("Not all stencils formed in 3 expansions")
            self._interpolants.add(interpolant)
            self._projections.add(projection)
            not_rank_mask = np.logical_not(interpolant.rank_mask)
            masked_points = tuple([interpolant.points[dim][not_rank_mask]
                                   for dim in range(ndims)])
            if self.strategy == 'expand':
                # Increase support region radius by one and apply to remaining
                # points
                support = support.expand_radius(1)
                interpolant = Interpolant(support, self.group,
                                          self.basis_map,
                                          self.skin.geometry,
                                          masked_points)

            elif self.strategy == 'reduce':
                basis_map = {func: interpolant.basis_map[func].reduce_order(2)
                             for func in interpolant.basis_map}
                interpolant = Interpolant(support, self.group,
                                          basis_map,
                                          self.skin.geometry,
                                          masked_points)
                projection = Projection(self.deriv, self.group, basis_map)

            else:
                raise ValueError("Unrecognised strategy")

            loops += 1

        # Will need to append the final interpolant and projection upon exiting
        # the loop
        self._interpolants.add(interpolant)
        self._projections.add(projection)

    def _setup_weight_funcs(self):
        """
        Set up the functions which will be used to contain stencil weights.
        """
        # Get the interpolant with the largest support region
        interp = self.interpolants.largest_support
        rhs = interp.vector
        rhs_nonzero = rhs != 0

        grid = self.geometry.grid
        dims = grid.dimensions
        ndims = len(dims)

        wfuncs = []
        weight_map = {}
        for item in rhs[rhs_nonzero]:
            if isinstance(item, dv.types.Indexed):
                indices = [int((item.indices[1+d]
                               - dims[d]).as_coeff_Mul()[0])
                           for d in range(ndims)]
                underscores = ['_' for d in range(ndims)]
                indices_str = [str(i) for i in indices]
                index = list(chain(*zip(underscores, indices_str)))
                name = 'w_' + item.name
                # Swap minus signs for m in identifier
                id = ''.join(index).replace('-', 'm')
                name += id
                wfunc = dv.Function(name=name, grid=grid)
                wfuncs.append(wfunc)
                weight_map[item] = wfunc
            else:
                raise NotImplementedError("Non-Function RHS not implemented")

        self._wfuncs = tuple(wfuncs)
        self._weight_map = frozendict(weight_map)

    def _fill_weight_funcs(self):
        """
        Fill the data attributes of the stencil weight functions with their
        respective coefficients.
        """
        self._fill_standard_weights()
        self._fill_modified_weights()

    def _fill_standard_weights(self):
        """
        Do an initial fill of the weight functions at interor points with the
        standard stencil weights
        """
        footprint = stencil_footprint(self.deriv)
        interior_stencil = get_sten_vector(self.deriv, footprint)

        expr = self.deriv.expr
        t = expr.time_dim
        dims = expr.space_dimensions

        for i in footprint.shape[-1]:
            ind = footprint[:, i]
            func_index = (t,) + tuple([dims[i]+ind[i]
                                       for i in range(len(dims))])
            wfunc = self.weight_map[expr[func_index]]
            wfunc.data[self.geometry.interior_mask] = interior_stencil[i]

    def _fill_modified_weights(self):
        """Fill the weight functions with the modified weights"""
        interpolants = self.interpolants.interpolants
        projections = self.projections.projections

        for i in range(len(interpolants)):
            interp = interpolants[i]
            projec = projections[i]
            # Points for this interpolant
            mod_pts = interp.points[interp.rank_mask]
            # Get the stencils
            stencils = interp.project(projec)

            # Get the stencil points
            rhs = interp.vector
            rhs_nonzero = rhs != 0
            rhs_masked = rhs[rhs_nonzero]
            # Now loop over points in this stencil
            for j in range(len(rhs_masked)):
                item = rhs_masked[j]
                if isinstance(item, dv.types.Indexed):
                    wfunc = self.weight_map[item]
                    wfunc.data[mod_pts] = stencils[:, j]

                else:
                    errmsg = "Non-Function RHS not implemented"
                    raise NotImplementedError(errmsg)

    @property
    def deriv(self):
        """The derivative to be substituted"""
        return self._deriv

    @property
    def group(self):
        """Boundary conditions corresponding to the derivative"""
        return self._group

    @property
    def basis_map(self):
        """Mapping between functions and corresponding basis functions"""
        return self._basis_map

    @property
    def strategy(self):
        """Strategy used when matrix system is of insufficient rank"""
        return self._strategy

    @property
    def skin(self):
        """The boundary-adjacent skin where modified stencils are required"""
        return self._skin

    @property
    def geometry(self):
        """The boundary geometry"""
        return self._geometry

    @property
    def interpolants(self):
        """The collection of Interpolant objects"""
        return self._interpolants

    @property
    def projections(self):
        """The collection of Projection objects"""
        return self._projections

    @property
    def wfuncs(self):
        """The weight functions"""
        return self._wfuncs

    @property
    def weight_map(self):
        """Mapping between stencil positions and weights"""
        return self._weight_map
