"""Immersed boundary object forming the core API"""

from schism.geometry.skin import ModifiedSkin

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
            skin = ModifiedSkin(deriv, self.geometry)

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
