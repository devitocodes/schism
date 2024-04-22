"""Immersed boundary object forming the core API"""

import devito as dv
from devito.tools.memoization import memoized_meth

from schism.geometry.skin import ModifiedSkin
from schism.basic.basis import Basis
from schism.finite_differences.substitution import Substitution
from devito.tools.data_structures import frozendict

__all__ = ['Boundary']


class Boundary:
    """
    An immersed surface on which specified boundary conditions are imposed

    Parameters
    ----------
    conditions : BoundaryConditions
        Boundary conditions to be imposed on the surface
    geometry : BoundaryGeometry
        Geometry data for the surface
    has_1D_basis : bool
        Use 1D basis functions to fit and project the field. Default is False.
        Note that attempting to use 1D basis functions with boundary conditions
        which are not valid in 1D will raise an error.
    strategy : str
        Strategy to use when insufficient information is available to fit the
        function at some point. Can be set to 'expand' to expand the support
        region in these cases, or 'reduce' to reduce the order of the
        polynomial basis. Default is 'expand'.

    Methods
    -------
    substitutions(derivs)
        Get the substitution for the specified derivative. This will return
        the modified stencils that should replace the standard ones.
    """
    def __init__(self, conditions, geometry, **kwargs):
        self._has_1D_basis = kwargs.get('has_1D_basis', False)
        self._strategy = kwargs.get('strategy', 'expand')
        self._conditions = conditions
        self._geometry = geometry

    @memoized_meth
    def substitutions(self, derivs):
        """
        Get the substitution for the specified derivative. This will return
        the modified stencils that should replace the standard ones.

        Parameters
        ----------
        derivs : tuple
            Derivatives to be substituted
        """
        subs = {}  # Dict of substitutions
        for deriv in derivs:
            print("Generating stencils for", deriv)
            if not isinstance(deriv.expr, dv.Function):
                raise TypeError("Substituted derivatives must be of functions")
            skin = ModifiedSkin(deriv, self.geometry)

            group = self._get_filtered_group(deriv)

            # Form the basis map given the group of bcs and the derivative
            # to be approximated
            basis_map = self._get_basis_map(deriv, group)

            substitution = Substitution(deriv, group, basis_map, self.strategy,
                                        skin)
            subs[deriv] = substitution.expr

        return frozendict(subs)

    def _get_filtered_group(self, deriv):
        """
        Get the BC group filtered to BCs corresponding with specified
        derivative.
        """
        group = self.conditions.get_group(deriv.expr.function)
        if self.has_1D_basis:
            if len(deriv.dims) != 1:
                errmsg = "Only 1D derivatives can be taken with 1D basis"
                raise ValueError(errmsg)
            group = group.filter(deriv.dims[0])
        return group

    def _get_basis_map(self, deriv, group):
        """
        Return a dict mapping functions onto their respective basis functions.
        """
        map = {}
        if self.has_1D_basis:
            for func in group.funcs:
                map[func] = Basis(name=func.name+'_'+deriv.dims[0].name,
                                  dims=deriv.dims,
                                  order=func.space_order)
        else:  # N-D basis
            for func in group.funcs:
                map[func] = Basis(name=func.name,
                                  dims=func.space_dimensions,
                                  order=func.space_order)

        # Return a frozendict as doesn't want to be mutable
        return frozendict(map)

    @property
    def has_1D_basis(self):
        """Does boundary use 1D basis functions?"""
        return self._has_1D_basis

    @property
    def strategy(self):
        """
        Strategy to use when insufficient information available to fit basis.
        """
        return self._strategy

    @property
    def conditions(self):
        """Boundary conditions attached to the surface"""
        return self._conditions

    @property
    def geometry(self):
        """Gemetry data for the boundary"""
        return self._geometry
