"""Classes for expressing boundary conditions"""

import devito as dv
import sympy as sp
import itertools as it
from functools import reduce


class Basis:
    """
    N-d polynomial basis with N dimensions

    Parameters
    ----------
    order : int
        The order of the basis function
    ndim : int
        The number of dimensions of the basis
    """
    def __init__(self, order, ndim):
        self._order = order
        self._ndim = ndim
        self._check_parameters()
        self._get_orders()
        self._def_parameters()
        self._get_basis()

    def __str__(self):
        return "Basis({})".format(str(self.basis))

    def _check_parameters(self):
        """Check input parameters"""
        if type(self._order) != int:
            raise ValueError("order must be integer")
        if self._order < 1:
            raise ValueError("order must be positive")
        if type(self._ndim) != int:
            raise ValueError("ndim must be integer")
        if self._ndim < 1:
            raise ValueError("ndim must be positive")

    def _get_orders(self):
        """Get powers for all terms in the basis"""
        # Get all combinatorial powers
        all_pwrs = list(it.product(range(self._order+1), repeat=self._ndim))
        # Filter to valid powers only
        self._pwrs = [pwr for pwr in all_pwrs if sum(pwr) <= self._order]

    def _def_parameters(self):
        """Define the parameters used in the basis expression"""
        self._x = sp.IndexedBase('x')
        self._a = sp.IndexedBase('a')
        self._d = sp.IndexedBase('d')  # Derivative*spacing placeholder

    def _get_basis(self):
        """
        Returns a SymPy expression for the N-d polynomial basis of a given
        order approximating some function.
        """
        trms = []
        for pwr in self.powers:
            # Coefficient of the N-d Taylor series expansion for given power
            trm = reduce((lambda x0, x1: x0*x1),
                         [(self.x[i]-self.a[i])**pwr[i]/sp.factorial(pwr[i])
                          for i in range(self.ndim)])
            trms.append(trm*self.d[pwr])

        self._expr = sum(trms)

    def derivative(self, deriv_orders):
        """
        Get a derivative of the basis

        Parameters
        ----------
        deriv_orders : tuple
            Derivative order in each respective dimension
        """
        if type(deriv_orders) != tuple:
            raise TypeError("derivative orders must be given as tuple")
        if len(deriv_orders) != self.ndim:
            errmsg = "derivative orders must be given for each dimension"
            raise ValueError(errmsg)
        if sum(deriv_orders) > self.order:
            raise ValueError("derivative is higher-order than basis")

        derivs = [(self.x[i], deriv_orders[i]) for i in range(self.ndim)]

        try:
            return self.basis.diff(*derivs)
        except ValueError:
            raise ValueError("order of differention must be nonnegative")

    @property
    def basis(self):
        """Get the SymPy expression for the basis"""
        return self._expr

    @property
    def powers(self):
        """List of the powers of terms in the basis"""
        return self._pwrs

    @property
    def ndim(self):
        """Dimensionality of the basis function"""
        return self._ndim

    @property
    def order(self):
        """Order of the basis function"""
        return self._order

    @property
    def a(self):
        """Parameter 'a', representing expansion point"""
        return self._a

    @property
    def x(self):
        """Parameter 'x' representing position"""
        return self._x

    @property
    def d(self):
        """
        Parameter 'd': a placeholder for derivative at expansion point
        multiplied by grid increment.
        """
        return self._d


class BoundaryCondition:
    """
    Encapsulates a single equation specifying a boundary condition.

    Parameters
    ----------
    lhs : list
        A list of tuples representing the left-hand side of the equation. Each
        tuple contains (function, derivative order per dimension, coefficient).
    rhs : Function or float
        The right-hand side of the boundary condition
    """
    def __init__(self, lhs, rhs):
        self._lhs = lhs
        self._check_lhs()

        self._rhs = rhs

        self._setup_parameters()

        self._get_expr_dict()

    def _check_lhs(self):
        """Check that argument supplied for left-hand side is valid"""
        # Assert lhs is a list
        if type(self.lhs) != list:
            raise TypeError("lhs should be list")
        # Check that list has items
        if len(self.lhs) == 0:
            raise ValueError("empty list supplied to BoundaryCondition")

        first_term = self.lhs[0]  # Used to check against
        derivs_taken = []  # Used to check derivatives are not repeated
        for term in self.lhs:
            if type(term) != tuple:
                raise TypeError("terms should be specified as tuples")
            # Check that all terms have length 3
            if len(term) != 3:
                raise ValueError("terms should have 3 parameters")
            # Check that all items in tuples are of correct types
            if not issubclass(term[0], dv.Function):
                errmsg = "function should be supplied as Devito Function"
                raise TypeError(errmsg)
            if not type(term[1]) == tuple:
                errmsg = "derivative orders should be supplied as tuple"
                raise TypeError(errmsg)
            if not issubclass(term[2], dv.Function) or type(term[2]) != float:
                errmsg = "coefficient should be float or Devito Function"
                raise TypeError(errmsg)
            # Check all functions share a grid
            if term[0].grid is not first_term[0].grid:
                raise ValueError("functions supplied must share a grid")
            # Check derivative orders correspond with grid dimensions
            if len(term[1]) != len(term[0].grid.dimensions):
                errmsg = "derivative orders do not match grid dimensions"
                raise ValueError(errmsg)
            # Check that all functions share a space order
            if term[0].space_order != first_term[0].space_order:
                errmsg = "functions with differing space orders not supported"
                raise NotImplementedError(errmsg)
            # Check that all derivatives are lower than this space order
            if sum(term[1]) > term[0].space_order:
                errmsg = "derivative specified at too-high an order"
                raise ValueError(errmsg)
            # Check that derivatives are not repeated
            if str(term[0]) + str(term[1]) not in derivs_taken:
                derivs_taken.append(str(term[0]) + str(term[1]))
            else:
                errmsg = "derivative {1} specified twice for function {0}"
                raise ValueError(errmsg.format(*term))

    def _setup_parameters(self):
        """Misc parameter setup"""
        self._so = self.lhs[0][0].space_order
        self._grid = self.lhs[0][0].grid
        self._dim = self.grid.dimensions
        self._ndim = len(self.dim)

    def _get_expr_dict(self):
        """
        Generate a dictionary mapping functions within the boundary condition
        onto their respective expression
        """
        expr_dict = {}
        # Reuse one basis for all functions
        basis = Basis(self.order, self.ndim)
        self._a = basis.a
        self._x = basis.x
        self._d = basis.d
        self._alpha = sp.IndexedBase('alpha')
        # (function, derivative order per dimension, coefficient)
        for term in self.lhs:
            func, deriv_order, _ = term
            # Placeholder weighting used for now
            expr = self.alpha[deriv_order]*basis.derivative(deriv_order)

            try:
                expr_dict[func] += expr
            except KeyError:
                expr_dict[func] = expr

        self._expr_dict = expr_dict
        self._functions = tuple(self.expr_dict.keys())

    @property
    def lhs(self):
        """List of tuples representing the left-hand side of the equation"""
        return self._lhs

    @property
    def rhs(self):
        """Right-hand side of the equation"""
        return self._rhs

    @property
    def order(self):
        """
        The space order of the functions concerned by the boundary conditions.
        """
        return self._so

    @property
    def grid(self):
        """The grid on which the functions are defined"""
        return self._grid

    @property
    def dimensions(self):
        """Dimensions of the functions"""
        return self._dims

    @property
    def ndim(self):
        """Dimensionality of the functions"""
        return self._ndim

    @property
    def functions(self):
        """Functions on which the boundary condition is imposed"""
        return self._functions

    @property
    def expressions(self):
        """Left-hand side expressions for each function"""
        return self._expr_dict

    @property
    def a(self):
        """Parameter 'a', representing expansion point"""
        return self._a

    @property
    def x(self):
        """Parameter 'x' representing position"""
        return self._x

    @property
    def d(self):
        """
        Parameter 'd': a placeholder for derivative at expansion point
        multiplied by grid increment.
        """
        return self._d

    @property
    def alpha(self):
        """
        Parameter used as a coefficient placeholder in boundary condition
        expressions
        """
        return self._alpha


def main():
    """Main function for troubleshooting"""
    print(Basis(2, 1).derivative((1,)))
    print("\n")
    print(Basis(4, 1).derivative((2,)))
    print("\n")
    print(Basis(2, 2).derivative((0, 1)))
    print("\n")
    print(Basis(4, 2).derivative((1, 2)))
    print("\n")
    print(Basis(2, 3).derivative((0, 1, 0)))
    print("\n")
    print(Basis(2, 4).derivative((0, 1, 1, 0)))  # 4D basis for testing


if __name__ == "__main__":
    main()
