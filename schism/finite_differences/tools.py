"""Tools for generating finite difference stencils"""

import sympy as sp
import numpy as np

from functools import reduce
import devito as dv


def get_sten_vector(deriv, points):
    """Get the stencil coefficient vector for a specified derivative"""
    func = deriv.expr
    s_o = func.space_order

    # Create a dict mapping dimensions onto their derivatives
    if type(deriv.deriv_order) == int:  # Not always iterable
        deriv_orders = (deriv.deriv_order,)
    else:
        deriv_orders = deriv.deriv_order
    deriv_map = {deriv.dims[i]: deriv_orders[i]
                 for i in range(len(deriv.dims))}
    # Get the indices
    inds = np.arange(-s_o//2, s_o//2 + 1)

    # Account for the function staggering
    if func.staggered == dv.NODE or func.staggered is None:
        stagger = {dim: 0 for dim in func.space_dimensions}
    elif isinstance(func.staggered, dv.SpaceDimension):
        stagger = {dim: (dim.spacing/2 if dim == func.staggered else 0)
                   for dim in func.space_dimensions}
    else:
        stagger = {dim: (dim.spacing/2 if dim in func.staggered else 0)
                   for dim in func.space_dimensions}

    dim_coeffs = []  # List of coefficients per dimension
    for dim in func.space_dimensions:
        try:
            order = deriv_map[dim]
        except KeyError:
            order = 0
        try:
            # Get the relative offset (set dim to zero and spacing to 1)
            x0 = float((deriv.x0[dim]
                        - stagger[dim]).subs([(dim, 0), (dim.spacing, 1)]))
        except KeyError:
            x0 = 0
        dim_coeffs.append(np.array(sp.finite_diff_weights(order,
                                                          inds, x0)[-1][-1],
                                   dtype=float))

    msh = np.meshgrid(*dim_coeffs, indexing='ij')

    coeffs = reduce(lambda a, b: a*b, msh)

    # Offset the stencil indices to convert to array indices
    offset_inds = tuple([points[dim]+s_o//2
                         for dim in range(len(func.space_dimensions))])

    coeff_vec = coeffs[offset_inds]
    return coeff_vec


def extract_values(grid, symbols, expr_map, points):
    """
    Pull values of some expression off of the grid. Used in the evaluation of
    variable coefficients in boundary conditions.

    Parameters
    ----------
    grid : Devito Grid
        The grid on which the Functions are defined
    symbols : tuple
        Tuple containing the coefficient placeholder symbols
    expr_map : dict
        Mapping between coefficient placeholder symbols and their respective
        expressions
    points : tuple
        Tuple of arrays containing the x, y, z, etc values of points where
        values are to be extracted
    """
    interp_terms = []  # Interpolation terms
    sparse_map = {}
    npts = points[0].shape[0]
    for sym in symbols:
        expr = expr_map[sym]
        func = dv.SparseFunction(name=sym.name, grid=grid, npoint=npts)
        sparse_map[sym] = func
        for i in range(len(points)):
            # Set the coordinates
            func.coordinates.data[:, i] = points[i]

        # Create an interpolation term for this
        interp = func.interpolate(expr=expr)
        interp_terms.append(interp)

    dv.Operator(interp_terms, name='eval_coeffs')()

    return tuple([sparse_map[sym].data for sym in symbols])
