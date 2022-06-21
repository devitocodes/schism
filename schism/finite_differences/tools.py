"""Tools for finite difference stencils"""

import sympy as sp
import numpy as np

from functools import reduce


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

    dim_coeffs = []  # List of coefficients per dimension
    for dim in func.space_dimensions:
        try:
            order = deriv_map[dim]
        except KeyError:
            order = 0
        try:
            x0 = deriv.x0[dim]
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
