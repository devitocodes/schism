"""Objects for the specification of boundary conditions"""

__all__ = ['BoundaryConditions']

import devito as dv
import sympy as sp
import networkx as nx

from devito.symbolics import retrieve_functions
from devito.tools.data_structures import frozendict


class SingleCondition:
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
    expr_map : dict
        Map between coefficient placeholders and their expressions within the
        LHS expression

    Methods
    -------
    sub_basis(basis_map)
        Substitute instances of functions in the boundary conditions with their
        respective basis functions.
    """
    def __init__(self, eq, funcs=None):
        self._eq = eq
        # Parse the rhs
        self._parse_rhs()
        # Get the intersection between funcs in lhs and funcs supplied
        self._get_functions(funcs)
        # Get the dimensions in which derivatives are taken
        self._get_derivative_dimensions()
        # Parse the lhs and determine the coefficients
        self._parse_lhs()
        # Note: needs to complain if a function found in the expression
        # TODO: Finish this

    def __eq__(self, other):
        lhs_equal = self.lhs - other.lhs == 0
        rhs_equal = self.rhs - other.rhs == 0
        if isinstance(other, self.__class__):
            return lhs_equal and rhs_equal
        else:
            return False

    def __str__(self):
        return "SingleCondition({})".format(str(self.equation))

    def __repr__(self):
        return "SingleCondition({})".format(str(self.equation))

    def _parse_lhs(self):
        """
        Parse and process the LHS of the equation. Replace any non-constant
        coefficients with placeholder symbols.
        """
        self._expr_map = {}
        coeff_inc = 0

        # Expansion necessary to avoid expressions which should be Add being
        # disguised as Mul. For example g*(f+1) -> g*f + g.
        exp_lhs = sp.expand(self.lhs)
        if isinstance(exp_lhs, dv.Function):
            self._mod_lhs = self.lhs  # No coefficients to substitute
        elif isinstance(exp_lhs, dv.Derivative):
            self._mod_lhs = self.lhs  # No coefficients to substitute
        elif isinstance(exp_lhs, sp.Mul):
            # LHS is some function/derivative with a coefficient
            derivs = self.lhs.find(dv.Derivative)
            funcs = self.lhs.find(dv.Function).intersection(set(self.funcs))
            derivs_funcs = tuple(derivs.union(funcs))

            mod_args = []
            # May loop more than once if LHS is a derivative
            for item in derivs_funcs:
                coeff = self.lhs.coeff(item)
                if not isinstance(coeff, sp.core.numbers.Number):
                    coeff_sym = sp.Symbol('coeff_' + str(coeff_inc))
                    coeff_inc += 1  # Increment the coefficient labels
                    self._expr_map[coeff_sym] = coeff
                    mod_args.append(coeff_sym*item)
                else:
                    mod_args.append(coeff*item)
            self._mod_lhs = sum(mod_args)
        elif isinstance(exp_lhs, sp.Add):
            # LHS contains a bunch of Functions, Derivatives, and coefficients
            args = exp_lhs._args
            mod_args = list(args)  # Will modifiy entries in this list
            for i in range(len(mod_args)):
                arg = mod_args[i]  # Doesn't actually acess directly
                derivs = arg.find(dv.Derivative)
                funcs = arg.find(dv.Function).intersection(set(self.funcs))
                derivs_funcs = tuple(derivs.union(funcs))
                # Would generally only expect this to loop if the arg contains
                # a derivative
                item_args = []
                for item in derivs_funcs:
                    coeff = arg.coeff(item)
                    # If we do have a number, then don't change the arg
                    if not isinstance(coeff, sp.core.numbers.Number):
                        coeff_sym = sp.Symbol('coeff_' + str(coeff_inc))
                        coeff_inc += 1  # Increment the coefficient labels
                        self._expr_map[coeff_sym] = coeff
                        item_args.append(coeff_sym*item)
                if len(derivs_funcs) == 0:
                    if len(arg.find(dv.Function)) != 0:
                        # Other functions present in the expression
                        coeff = arg
                        coeff_sym = sp.Symbol('coeff_' + str(coeff_inc))
                        self._expr_map[coeff_sym] = coeff
                        mod_args[i] = coeff_sym
                if len(item_args) != 0:  # If I have coefficients to modify
                    mod_args[i] = sum(item_args)  # Overwrite current arg
            self._mod_lhs = sum(mod_args)
        else:
            raise TypeError("LHS of unhandled type")

    def _parse_rhs(self):
        """Parse and process the RHS of the equation"""
        if self.rhs != 0:
            raise NotImplementedError("Nonzero RHS currently unsupported")

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

    def _get_derivative_dimensions(self):
        """Get the dimensions in which derivatives are taken"""
        # Get all the derivatives on the lhs
        derivs = self.lhs.find(dv.Derivative)
        dims = set()  # Dimensions in which derivs are taken
        for deriv in derivs:
            # Check that expression is a function
            if not isinstance(deriv.expr, dv.Function):
                errmsg = """Derivatives of non-Function expressions
                            unsupported. Note that currently f.dx.dy will not
                            be collapsed to f.dxdy, and should be amended as
                            such."""
                raise NotImplementedError(errmsg)

            # Check that derivative is of a function in .funcs
            if deriv.expr in self.funcs:
                dims.update(deriv.dims)
            # Derivatives of other fields currently unsupported
            else:
                errmsg = "Derivatives of coefficent functions unsupported"
                raise NotImplementedError(errmsg)

        if len(dims) == 0:  # If no derivatives present, then set to None
            self._dims = None
        else:  # Otherwise record the derivatives
            self._dims = tuple(dims)

    def sub_basis(self, basis_map):
        """
        Substitute functions in the boundary condition for their respective
        basis functions.
        """
        # FIXME: Needs to act on the modified LHS
        derivs = self.lhs.find(dv.Derivative)
        funcs = self.lhs.find(dv.Function).intersection(set(self.funcs))
        reps = {}
        for func in funcs:
            try:
                basis = basis_map[func]
            except KeyError:
                # Should never end up here
                raise ValueError("No basis generated for required function")
            reps[func] = basis.expr
        for deriv in derivs:
            try:
                basis = basis_map[deriv.expr]
            except KeyError:
                # Should never end up here
                raise ValueError("No basis generated for required function")
            if type(deriv.deriv_order) != dv.types.utils.DimensionTuple:
                d_o = (deriv.deriv_order,)
            else:
                d_o = deriv.deriv_order
            # Derivs to take of the basis
            b_derivs = tuple([(deriv.dims[i], d_o[i])
                              for i in range(len(deriv.dims))])
            reps[deriv] = basis.deriv(b_derivs)
        return sp.simplify(self._mod_lhs.subs(reps))

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
    def funcs(self):
        """Functions on which the BC is imposed"""
        return self._funcs

    @property
    def dims(self):
        """Dimensions in which derivatives are taken"""
        return self._dims

    @property
    def expr_map(self):
        """
        Mapping between coefficient symbols and the expressions they replace.
        """
        return frozendict(self._expr_map)


class ConditionGroup:
    """
    A group of boundary conditions with overlapping functions

    Parameters
    ----------
    conditions : tuple
        SingleCondition objects in the group
    funcs : tuple
        Functions that define the group

    Attributes
    ----------
    conditions : tuple
        SingleCondition objects in the group
    funcs : tuple
        Functions that define the group
    dimension_map : dict
        Mapping between dimensions and valid BCs for that dimension

    Methods
    -------
    filter(dim)
        Returns a ConditionGroup containing boundary conditions which only have
        derivatives in the specified dimension or none at all.
    """
    def __init__(self, conditions, funcs):
        self._conds = conditions
        self._funcs = funcs

    def __str__(self):
        return "ConditionGroup{}".format(str(self.funcs))

    def __repr__(self):
        return "ConditionGroup{}".format(str(self.funcs))

    def filter(self, dim):
        """
        Return a ConditionGroup containing boundary conditions which only
        contain derivatives in the specified dimension or no derivatives.
        """
        # Filtered conditions
        f_conds = [cond for cond in self.conditions
                   if cond.dims == (dim,) or cond.dims is None]
        # Functions don't need filtering as 1D bcs will end up in separate
        # groups
        return self.__class__(f_conds, self.funcs)

    @property
    def conditions(self):
        """SingleCondition objects in the group"""
        return self._conds

    @property
    def funcs(self):
        """Functions that define the group"""
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
        SingleCondition objects for each boundary condition
    groups : tuple
        Groups of boundary conditions which are connected by the functions
        used within them
    funcs : tuple
        Functions on which the boundary conditions are imposed (as specified)
        by user
    function_map : dict
        Mapping between functions and boundary condition groups

    Methods
    -------
    get_group(func)
        Returns the BC group for a given function
    """
    def __init__(self, eqs, funcs=None):
        # Flatten functions supplied and remove duplicates
        self._flatten_functions(funcs)
        # Create the equations tuple, flattening equations
        self._flatten_equations(eqs)
        # Set up the SingleCondition objects for each equation
        self._setup_bcs()
        # Set up the groups using networkx
        self._group_bcs()

    def _flatten_functions(self, funcs):
        """Flatten the functions provided where necessary"""
        if funcs is None:
            self._funcs = None
        else:
            flat_funcs = set()
            for func in funcs:
                try:
                    flat_funcs.update(func.flat())
                except AttributeError:
                    flat_funcs.add(func)
            self._funcs = tuple(flat_funcs)

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

    def _setup_bcs(self):
        """Set up a SingleCondition object for each equation"""
        self._bcs = [SingleCondition(eq, funcs=self.funcs)
                     for eq in self.equations]

    def _group_bcs(self):
        """Group the BCs supplied by their overlapping functions"""
        # Use networkx graph to find interconnectivity between functions
        f_graph = nx.Graph()
        for bc in self.bcs:
            f_graph.add_nodes_from(bc.funcs)
            # Set up edges between first function and all others
            # Will all end up linked through the first function
            edges = [(bc.funcs[0], func) for func in bc.funcs[1:]]
            f_graph.add_edges_from(edges)

        # Functions interlinked by boundary conditions
        f_groups = [tuple(group) for group in nx.connected_components(f_graph)]

        bc_groups = []
        f_map = {}
        for group in f_groups:
            conditions = tuple([bc for bc in self.bcs if bc.funcs[0] in group])
            new_group = ConditionGroup(conditions, group)
            bc_groups.append(new_group)
            for func in group:
                f_map[func] = new_group

        self._groups = tuple(bc_groups)
        # Dictionary does not want to be mutable from here on out
        self._f_map = frozendict(f_map)

    def get_group(self, func):
        """Get the boundary condition group for a given function"""
        return self.function_map[func]

    @property
    def equations(self):
        """Equalities to be imposed on the surface"""
        return self._eqs

    @property
    def bcs(self):
        """SingleCondition for each equation"""
        return self._bcs

    @property
    def funcs(self):
        """
        The functions on which Boundary conditions are imposed. Note these are
        what has been specified by the user, not what is actually contained
        within the boundary conditions
        """
        return self._funcs

    @property
    def groups(self):
        """
        Groups of boundary conditions which are connected by the functions
        used within them
        """
        return self._groups

    @property
    def function_map(self):
        """
        Mapping between functions and boundary condition groups
        """
        return self._f_map
