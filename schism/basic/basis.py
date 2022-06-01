"""The symbolic basis"""

import sympy as sp
from itertools import product
from functools import reduce

__all__ = ['Basis']


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
    dimensions : tuple
        Basis dimensions
    order : int
        Maximum order of the series
    terms : tuple
        Derivative terms in the expression
    n_terms : int
        The number of terms in the basis
    d : IndexedBase
        Placeholder symbol for grid spacing x derivatives
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

    def _get_expression(self):
        """Get the symbolic expression for the basis"""
        # Generate the term orders
        self._terms = [term for term in product(range(self.order+1),
                                                repeat=self.ndims)
                       if sum(term) <= self.order]

        self._d = sp.IndexedBase('d_'+self.name)  # Grid increment x derivative

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
        return self.expr.diff(derivative)

    def reduce_order(self, reduce):
        """Return the basis with reduced order"""
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
    def d(self):
        """Placeholder symbol for grid spacing x derivatives"""
        return self._d

    @property
    def expr(self):
        """The symbolic expression of the basis"""
        return self._expr
