"""The Substitution object used to produce a single derivative substitution"""

import numpy as np
import devito as dv
import sympy as sp

from schism.finite_differences.interpolate_project import MultiInterpolant, \
    MultiProjection, Interpolant, Projection
from schism.geometry.support_region import SupportRegion
from schism.geometry.skin import stencil_footprint
from schism.finite_differences.tools import get_sten_vector
from itertools import chain
from devito.tools.data_structures import frozendict
from functools import reduce


class Substitution:
    """
    A substitution for a particular derivate expression. Encapsulates the
    modified finite-difference stencils in the vincinity of the boundary.

    Parameters
    ----------
    deriv : Derivative
        The derivate to be substituted
    group : ConditionGroup
        The corresponding group of boundary conditions
    basis_map : frozendict
        Mapping between functions and corresponding basis functions
    strategy : str
        The strategy used when the matrix system has insufficient rank
    skin : ModifiedSkin
        The boundary-adjacent skin in which modified boundary stencils are
        required
    """
    def __init__(self, deriv, group, basis_map, strategy, skin):
        self._deriv = deriv
        self._group = group
        self._basis_map = basis_map
        self._strategy = strategy
        self._skin = skin
        self._geometry = self.skin.geometry

        self._setup_collections()
        self._get_stencils()
        self._setup_weight_funcs()
        self._fill_weight_funcs()
        self._form_expression()

    def _setup_collections(self):
        """
        Set up MultiInterpolant and MultiProjection objects used to collect
        Interpolant objects and their respective Projection objects
        """
        self._interpolants = MultiInterpolant()
        self._projections = MultiProjection()

    def _get_stencils(self):
        """
        Get the stencils by generating the required Interpolant and Projection
        objects
        """
        # Check for a time index in the function which is being differentiated
        # If the index exists, use it in place of the time dimension
        if self.deriv.expr.dimensions[0].is_Time:
            time = self.deriv.expr.indices[0]
        else:
            time = None
        ndims = len(self.geometry.grid.dimensions)
        radius_map = {func: func.space_order//2 for func in self.basis_map}
        support = SupportRegion(self.basis_map, radius_map)
        interpolant = Interpolant(support, self.group,
                                  self.basis_map, self.skin.geometry,
                                  self.skin.points, time=time)
        projection = Projection(self.deriv, self.group, self.basis_map)

        # Loop whilst there are points which don't yet have stencils
        maxloops = 3
        loops = 0  # Catch too many loops
        while not interpolant.all_full_rank:
            if loops >= maxloops:
                raise RuntimeError("Not all stencils formed in 3 expansions")
            self._interpolants.add(interpolant)
            self._projections.add(projection)
            not_rank_mask = np.logical_not(interpolant.rank_mask)
            masked_points = tuple([interpolant.points[dim][not_rank_mask]
                                   for dim in range(ndims)])
            if self.strategy == 'expand':
                # Increase support region radius by one and apply to remaining
                # points
                support = support.expand_radius(1)
                interpolant = Interpolant(support, self.group,
                                          self.basis_map,
                                          self.skin.geometry,
                                          masked_points, time=time)

            elif self.strategy == 'reduce':
                basis_map = {func: interpolant.basis_map[func].reduce_order(2)
                             for func in interpolant.basis_map}
                interpolant = Interpolant(support, self.group,
                                          basis_map,
                                          self.skin.geometry,
                                          masked_points, time=time)
                projection = Projection(self.deriv, self.group, basis_map)

            else:
                raise ValueError("Unrecognised strategy")

            loops += 1

        # Will need to append the final interpolant and projection upon exiting
        # the loop
        self._interpolants.add(interpolant)
        self._projections.add(projection)

    def _setup_weight_funcs(self):
        """
        Set up the functions which will be used to contain stencil weights.
        """
        # Tag for the derivative
        deriv_order = self.deriv.deriv_order
        if type(deriv_order) == int:
            deriv_order = (deriv_order,)  # Put in a tuple
        deriv_dims = self.deriv.dims
        deriv_tag = ''.join([d.name+str(o)
                             for d, o in zip(deriv_dims, deriv_order)])

        # Get the interpolant with the largest support region
        interp = self.interpolants.largest_support
        rhs = interp.vector
        rhs_nonzero = rhs != 0

        grid = self.geometry.grid
        dims = grid.dimensions
        ndims = len(dims)

        wfuncs = []
        weight_map = {}
        data_map = {item: np.zeros(grid.shape) for item in rhs[rhs_nonzero]}
        for item in rhs[rhs_nonzero]:
            # Check that the RHS is a Devito Function
            if isinstance(item, dv.Function):
                # Turn the offset in space into a tag
                # First reduce f(t, x+h_x, y-h_y) to [1, -1]
                # Substitution of dimension spacings with 1
                sp_subs = {dim.spacing: 1 for dim in dims}
                indices = [(ind-f_ind).subs(sp_subs) for dim, ind, f_ind
                           in zip(item.dimensions, item.indices,
                                  item.function.indices)
                           if dim.is_Space]
                underscores = ['_' for d in range(ndims)]
                indices_str = [str(i) for i in indices]
                index = list(chain(*zip(underscores, indices_str)))
                name = 'w_' + item.name + '_d' + deriv_tag
                # Swap minus signs for m in identifier
                id = ''.join(index).replace('-', 'm')
                name += id
                wfunc = dv.Function(name=name, grid=grid)
                wfuncs.append(wfunc)
                weight_map[item] = wfunc
            else:
                raise NotImplementedError("Non-Function RHS not implemented")

        self._wfuncs = tuple(wfuncs)
        self._data_map = frozendict(data_map)
        self._weight_map = frozendict(weight_map)

    def _fill_weight_funcs(self):
        """
        Fill the data attributes of the stencil weight functions with their
        respective coefficients.
        """
        self._fill_standard_weights()
        self._fill_modified_weights()

        # Fill weight data from data arrays
        # Can't currently directly fill them due to bug when indexing function
        # data with numpy arrays whilst filling from another array
        for term in self.weight_map:
            self.weight_map[term].data[:] = self.data_map[term]

    def _fill_standard_weights(self):
        """
        Do an initial fill of the weight functions at interor points with the
        standard stencil weights
        """
        footprint = stencil_footprint(self.deriv)
        interior_stencil = get_sten_vector(self.deriv, footprint)

        expr = self.deriv.expr
        dims = expr.space_dimensions

        # Needs to use the interior mask of the target subgrid
        deriv_stagger = []
        for d in range(len(expr.space_dimensions)):
            try:
                deriv_stagger.append(self.deriv.x0[expr.space_dimensions[d]]
                                     - expr.space_dimensions[d])
            except KeyError:
                deriv_stagger.append(sp.core.numbers.Zero())
        origin = tuple(deriv_stagger)

        # P for stencil point, d for dimension
        for p in range(len(footprint[0])):
            ind_subs = {dims[d]: dims[d] + footprint[d][p]*dims[d].spacing
                        for d in range(len(dims))}

            wdata = self.data_map[expr.subs(ind_subs)]
            wdata[self.geometry.interior_mask[origin]] = interior_stencil[p]

    def _fill_modified_weights(self):
        """Fill the weight functions with the modified weights"""
        interpolants = self.interpolants.interpolants
        projections = self.projections.projections
        ndims = len(self.geometry.grid.dimensions)

        for i in range(len(interpolants)):
            interp = interpolants[i]
            projec = projections[i]
            # Points for this interpolant
            mod_pts = tuple([interp.points[d][interp.rank_mask]
                             for d in range(ndims)])
            # Get the stencils
            stencils = interp.project(projec)

            # Get the stencil points
            rhs = interp.vector
            rhs_nonzero = rhs != 0
            rhs_masked = rhs[rhs_nonzero]
            # Now loop over points in this stencil
            for j in range(len(rhs_masked)):
                item = rhs_masked[j]
                if isinstance(item, dv.Function):
                    wdata = self.data_map[item]
                    wdata[mod_pts] = stencils[:, j]

                else:
                    errmsg = "Non-Function RHS not implemented"
                    raise NotImplementedError(errmsg)

    def _form_expression(self):
        """Form the expression for the derivative"""
        # Get the denominator
        order = self.deriv.deriv_order
        if type(order) == int:  # For derivs wrt one dim, deriv order is int
            order = (order,)
        dims = self.deriv.dims
        denom_args = [dim.spacing**ord for dim, ord in zip(dims, order)]
        denom = reduce(lambda a, b: a*b, denom_args)

        grid = self.geometry.grid
        spacing_subs = {dim.spacing: spa for dim, spa in zip(grid.dimensions,
                                                             grid.spacing)}
        denom = denom.subs(spacing_subs)
        # Divide weight data through by denominator instead of applying
        # it symbolically (faster operator compilation)
        for w in self.weight_map.values():
            w.data[:] /= float(denom)

        # Get the numerator
        # Multiply each weight by its corresponding expression
        func_args = {f.function: [] for f in self.weight_map.keys()}
        for f, w in zip(self.weight_map.keys(), self.weight_map.values()):
            func_args[f.function].append(w*f)

        evalderivs = []
        for f, args in func_args.items():
            # Creation of EvalDerivative here enable optimization down the line
            # Base at centred timestep doesn't seem to cause an issue?
            evalderivs.append(dv.EvalDerivative(*args, base=f))
        # Splitting by function seems to enable better compatibility with
        # Devito's stencil optimizations
        expr = sp.Add(*evalderivs)
        # Creation of EvalDerivative here enable optimization down the line
        # The derivative is of self.deriv.expr
        # Set self.expr
        self._expr = expr

    @property
    def deriv(self):
        """The derivative to be substituted"""
        return self._deriv

    @property
    def group(self):
        """Boundary conditions corresponding to the derivative"""
        return self._group

    @property
    def basis_map(self):
        """Mapping between functions and corresponding basis functions"""
        return self._basis_map

    @property
    def strategy(self):
        """Strategy used when matrix system is of insufficient rank"""
        return self._strategy

    @property
    def skin(self):
        """The boundary-adjacent skin where modified stencils are required"""
        return self._skin

    @property
    def geometry(self):
        """The boundary geometry"""
        return self._geometry

    @property
    def interpolants(self):
        """The collection of Interpolant objects"""
        return self._interpolants

    @property
    def projections(self):
        """The collection of Projection objects"""
        return self._projections

    @property
    def wfuncs(self):
        """The weight functions"""
        return self._wfuncs

    @property
    def weight_map(self):
        """Mapping between stencil positions and weights"""
        return self._weight_map

    @property
    def data_map(self):
        """Mapping between stencil positions and weight data"""
        return self._data_map

    @property
    def expr(self):
        """The symbolic derivative expression"""
        return self._expr
