"""Objects for the specification of boundary conditions"""

__all__ = ['BoundaryConditions', 'BoundaryCondition']


from devito.symbolics import retrieve_functions


class BoundaryCondition:
    """
    A single boundary condition imposed on some immersed surface

    Parameters
    ----------
    equation : Devito Eq
        The equality imposed on the surface
    functions : list of Devito Functions
        List of Functions on which boundary conditions are imposed. If none
        provided, then all time-dependent functions contained within the
        boundary condition will be used.

    Attributes
    ----------
    equation : Devito Eq
        The equality imposed on the surface
    functions : tuple
        The functions within the boundary condition on which boundary
        conditions are imposed
    dimensions : tuple
        Dimensions in which derivatives are taken within the boundary condition
    coeff_placeholders : tuple
        Placeholders used to replace coefficients in the LHS expression
    coeff_map : dict
        Map between coeffcient placeholders and their expressions within the
        LHS expression
    """
    def __init__(self, eq, funcs=None):
        self._eq = eq
        # Get the intersection between funcs in lhs and funcs supplied
        self._get_functions(funcs)
        # Get the dimensions in which derivatives are taken
        # Substitute the coefficients in the expression
        # Note: needs to complain if a function found in the expression

    def _get_functions(self, funcs):
        """
        Get the intersection between functions in the LHS and functions
        supplied by the user
        """
        if funcs is None:
            # Get all time-dependent functions in LHS
            all_funcs = retrieve_functions(self.lhs)
            # Cull duplicates in here as retrieve_functions works on a term-by-
            # term basis and will return duplicates if a function appears
            # multiple times in an expression.
            self._funcs = tuple(set([func for func in all_funcs
                                     if func.is_TimeDependent]))
        else:
            # Need intersection of functions in LHS and specified functions
            all_funcs = set(retrieve_functions(self.lhs))
            self._funcs = tuple(all_funcs.intersection(set(funcs)))

    @property
    def equation(self):
        """The equation imposed by the boundary condition"""
        return self._eq

    @property
    def lhs(self):
        """Alias for self.equation.lhs"""
        return self.equation.lhs

    @property
    def rhs(self):
        """Alias for self.equation.rhs"""
        return self.equation.rhs

    @property
    def functions(self):
        """Functions on which the BC is imposed"""
        return self._funcs


class BoundaryConditions:
    """
    A set of specified boundary conditions which will be imposed on some
    immersed surface.

    Parameters
    ----------
    equations : list of Devito Eq
        List of equalities to be imposed on the surface
    functions : list of Devito Functions
        List of Functions on which boundary conditions are imposed.

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
            try:
                eq_list += eq._flatten
            except KeyError:  # RHS doesn't match LHS size-wise
                errmsg = "Size of {} does not match {}"
                raise TypeError(errmsg.format(eq.lhs, eq.rhs))

        # Cull duplicates
        eq_set = set(eq_list)

        self._eqs = tuple(eq_set)

    @property
    def equations(self):
        """Equalities to be imposed on the surface"""
        return self._eqs
