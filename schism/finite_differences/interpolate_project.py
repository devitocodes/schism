"""
Objects for fitting of polynomial basis funtions and projection onto the
interior stencil.
"""


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
