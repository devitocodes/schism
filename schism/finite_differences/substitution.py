"""The Substitution object used to produce a single derivative substitution"""

import numpy as np

from schism.finite_differences.interpolate_project import (MultiInterpolant,
                                                           MultiProjection,
                                                           Interpolant,
                                                           Projection)
from schism.geometry.support_region import SupportRegion


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
                basis_map = {func: basis.reduce_order(2)
                             for func, basis in interpolant.function_map}
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
