"""Tools for searching expressions"""

__all__ = ['retrieve_derivatives']

import devito as dv


def q_derivs(expr):
    return isinstance(expr, dv.Derivative)


def retrieve_derivatives(exprs, mode='all'):
    """Shorthand to retrieve all the Derivatives in ``exprs``"""
    return dv.symbolics.search(exprs, q_derivs, mode, 'dfs')
