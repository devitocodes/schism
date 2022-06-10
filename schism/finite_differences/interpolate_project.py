"""
Objects for fitting of polynomial basis funtions and projection onto the
interior stencil.
"""

import sympy as sp


class MultiInterpolant:
    """
    A group of interpolants fitting function values and boundary conditions
    for various stencil spans or polynomial orders.

    Attributes
    ----------
    interpolants : tuple
        Interpolant objects attached to the MultiInterpolant

    Methods
    -------
    add(interpolant)
        Add an interpolant
    project(projections)
        Project the interpolants onto the interior stencil using their
        respective projections.
    """
    def __init__(self):
        self._interpolants = []

    def add(self, interpolant):
        """Add an Interpolant"""
        # FIXME: Will need to do a bunch of other stuff in due course
        self._interpolants.append(interpolant)

    @property
    def interpolants(self):
        """Interpolant objects attached to the MultiInterpolant"""
        return tuple(self._interpolants)


class MultiProjection:
    """
    A group of projections projecting fitted polynomials onto the interior
    derivative stencil.

    Attributes
    ----------
    projections : tuple
        The Projection objects attached to the MultiProjection

    Methods
    -------
    add(projection)
        Add a Projection
    """
    def __init__(self):
        self._projections = []

    def add(self, projection):
        """Add a Projection"""
        # FIXME: will need to do a bunch of other stuff in due course
        self._projections.append(projection)

    @property
    def projections(self):
        """Projection objects attached to the MultiProjection"""
        return tuple(self._projections)


class Interpolant:
    """
    Encapsulates the fitting of a set of polynomial basis functions to interior
    values and boundary conditions.

    Parameters
    ----------
    support : SupportRegion
        The support region used to fit this basis
    """
    def __init__(self, support, group):
        self._support = support
        self._group = group
        self._get_interior_vector()

    def _get_interior_vector(self):
        """
        Generate the vector of interior points corresponding with the support
        region.
        """
        # FIXME: Function order needs to be made consistent
        # FIXME: Apply some kind of sort to the functions before looping
        footprint_map = self.support.footprint_map
        # Needs to be an index notation like f[t, x-1, y+1]
        vec = []
        # Loop over functions
        for func in self.group.funcs:
            # Get the space and time dimensions of that function
            t = func.time_dim
            dims = func.space_dimensions
            footprint = footprint_map[func]
            # Create entry for each point in the support region
            for point in range(len(footprint[0])):
                space_ind = [dims[dim]+footprint[dim][point]
                             for dim in range(len(dims))]
                ind = (t,) + tuple(space_ind)
                vec.append(func[ind])

        # Make this a sympy Matrix
        self._interior_vector = sp.Matrix(vec)

    @property
    def support(self):
        """The support region used to fit the basis"""
        return self._support

    @property
    def group(self):
        """The boundary condition group"""
        return self._group

    @property
    def interior_vector(self):
        """The vector of interior points corresponding to the support region"""
        return self._interior_vector
