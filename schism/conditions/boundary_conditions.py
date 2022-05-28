"""Objects for the specification of boundary conditions"""

__all__ = ['BoundaryConditions']


class BoundaryConditions:
    """
    A set of specified boundary conditions which will be imposed on some
    immersed surface.

    Parameters
    ----------
    equations : list of Devito Eq
        List of equalities to be imposed on the surface
    functions : list of Devito Functions
        List of Functions on which boundary conditions are imposed. If none
        provided, then all time-dependent functions contained within the
        boundary conditions will be used.

    Attributes
    ----------
    equations : tuple
        Equalities to be imposed on the surface
    conditions : tuple
        BoundaryCondition objects for each boundary condition
    groups : tuple
        Groups of boundary conditions which are connected by the functions
        used within them
    functions : tuple
        The Functions relevant to the boundary conditions
    function_map : dict
        Mapping between functions and boundary condition groups
    """
    def __init__(self, eqs, funcs=None):
        # Create the equations tuple, flattening equations
        self._flatten_equations(eqs)

        # Set up the BoundaryCondition objects for each equation

        # Set up the groups using networkx

        # Set up the remaining attributes and any helpers etc

    def _flatten_equations(self, eqs):
        """Flatten the equations provided"""
        eq_list = []
        for eq in eqs:
            eq_list += eq._flatten

        print(eq_list)
        # Cull duplicates
        eq_set = set(eq_list)

        self._eqs = tuple(eq_set)

    @property
    def equations(self):
        """Equalities to be imposed on the surface"""
        return self._eqs
