"""The symbolic basis"""

import sympy as sp
import numpy as np
from itertools import product
from functools import reduce

__all__ = ['Basis', 'row_from_expr']


def row_from_expr(expr, funcs, basis_map, additional_params=None):
    """
    Generate a matrix row corresponding with some expression, where the LHS
    vector contains the derivative expressions.

    Parameters
    ----------
    expr : Mul
        The symbolic expression
    funcs : tuple
        The functions in the expression
    basis_map : dict
        The basis functions corresponding with those functions. Used to get the
        derivative placeholders and terms
    additional_params : tuple
        Additional parameters on top of the dimensions of the function. These
        are generally coefficients introduced by boundary conditions.

    Returns
    -------
    rowfunc : function
        Python function evaluating the matrix row. Note that to work around
        a SymPy broadcasting bug, the first two arguments should be set to 0
        for scalar values or np.zeros(shape) for arrays.
    """
    prelim_row = []
    for func in funcs:
        basis = basis_map[func]
        for term in basis.terms:
            # Needs a sp.expand otherwise coeff gathering is too stupid
            prelim_row.append(sp.expand(expr).coeff(basis.d[term]))

    # Need to add a dummy symbol and pass this as an argument
    # This is a workaround for SymPy issue #5642 to make the generated
    # function broadcast correctly for constant expressions
    dumsym = sp.symbols('dumsym')
    row = sp.Matrix(prelim_row)
    row += sp.Matrix([dumsym for i in range(len(prelim_row))])
    # Would ideally just be row = sp.Matrix(prelim_row)
    # Parameters are the grid dimensions
    params = tuple([dim for dim in funcs[0].space_dimensions])

    if additional_params is not None:
        params += tuple(additional_params)

    # Lambdify generated function with dummy variable to force the casting
    # We will also squeeze the output to ensure that the output size matches
    # the input
    gen_func = sp.lambdify((dumsym,)+params, row, 'numpy')

    def rowfunc(*param_vals):
        # Shape of the input parameters
        param_shape = np.shape(param_vals[0])
        # Check that parameters all have same shape
        for param in param_vals:
            if np.shape(param) != param_shape:
                raise ValueError("Inconsistent parameter array sizes")

        # Generate the matrix rows
        mat_rows = gen_func(np.zeros(param_shape), *param_vals)
        return np.squeeze(mat_rows)

    return rowfunc


class Basis:
    """
    The symbolic representation of an N-D McLauren-series basis

    Parameters
    ----------
    name : str
        Label for the basis
    dims : tuple
        Basis dimensions
    order : int
        Maximum order of the series

    Attributes
    ----------
    name : str
        Label for the basis
    dims : tuple
        Basis dimensions
    order : int
        Maximum order of the series
    terms : tuple
        Derivative terms in the expression
    nterms : int
        The number of terms in the basis
    d : dict
        Dictionary of placeholder symbols for grid spacing x derivatives
    expr : Mul
        The symbolic expression of the basis

    Methods
    -------
    deriv(derivative)
        Return the specified derivative of this basis
    reduce_order(reduction)
        Return the basis with the maximum order reduced
    """
    def __init__(self, name, dims, order):
        self._name = name
        self._dims = dims
        self._order = order

        self._get_expression()

    def __str__(self):
        return "Basis({})".format(str(self.expr))

    def __repr__(self):
        return "Basis({})".format(str(self.expr))

    def _get_expression(self):
        """Get the symbolic expression for the basis"""
        # Generate the term orders
        self._terms = [term for term in product(range(self.order+1),
                                                repeat=self.ndims)
                       if sum(term) <= self.order]

        self._d = {term: sp.Symbol('d_'+self.name+str(term))
                   for term in self.terms}  # Grid increment x derivative

        expr_terms = []
        for term in self.terms:
            expr_terms.append(reduce(lambda a, b: a*b,
                              [self.dims[n]**term[n]/sp.factorial(term[n])
                               for n in range(self.ndims)])*self.d[term])

        expr = sum(expr_terms)
        self._expr = expr

    def deriv(self, derivative):
        """
        Take a derivative of the expression given a set of tuples containing
        dimensions and their respective derivatives
        """
        return self.expr.diff(*derivative)

    def reduce_order(self, reduce):
        """Return the basis with reduced order"""
        if self.order - reduce < 0:
            raise ValueError("Order cannot be reduced below zero")
        return self.__class__(self.name+'_r'+str(reduce), self.dims,
                              self.order-reduce)

    @property
    def name(self):
        """Label for the basis"""
        return self._name

    @property
    def dims(self):
        """Basis dimensions"""
        return self._dims

    @property
    def ndims(self):
        """Number of dimensions"""
        return len(self.dims)

    @property
    def order(self):
        """The order of the approximation"""
        return self._order

    @property
    def terms(self):
        """Derivative terms in the expression"""
        return self._terms

    @property
    def nterms(self):
        """The number of terms in the expression"""
        return len(self.terms)

    @property
    def d(self):
        """Placeholder symbol for grid spacing x derivatives"""
        return self._d

    @property
    def expr(self):
        """The symbolic expression of the basis"""
        return self._expr
