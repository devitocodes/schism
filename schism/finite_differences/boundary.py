"""Immersed boundary object forming the core API"""

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
    """
    def __init__(self, conditions, geometry, **kwargs):
        self.has_1D_basis = kwargs.get('has_1D_basis', False)
        self.strategy = kwargs.get('strategy', 'expand')

        self._setup_basis()

    def _setup_basis(self):
        """
        Set up the basis functions given boundary conditions supplied and the
        specification.
        """
        return 0  # Dummy function for now until I decide how it works
