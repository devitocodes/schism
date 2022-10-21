"""
Model object for infrasound source location. Encapsulates the finite-difference
grid and subdomains, damping mask, immersed boundary for topography, and
model parameters.

Note that this is intended as a demonstration of a potential use-case of
Schism and Devito, rather than a code to be used in anger. However, it should
hopefully illustrate one approach by which such a code may be implemented.
"""

import devito as dv
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from examples.seismic import TimeAxis, RickerSource, Receiver


class BoundaryDomain(dv.SubDomain):
    """SubDomain for damping mask"""
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.name = kwargs.get('name')
        if self.name is None:
            raise ValueError("No subdomain name provided")
        self.spec = kwargs.get('spec')
        if self.spec is None:
            raise ValueError("No specification supplied")

    def define(self, dimensions):
        return {dim: sp for dim, sp in zip(dimensions, self.spec)}


class InfrasoundModel:
    """
    Model object for infrasound source location. Encapsulates the
    finite-difference grid and subdomains, damping mask, immersed boundary for
    topography, and model parameters.

    This model features a damping mask 10 points thick along each edge.

    Parameters
    ----------
    dims : int
        Number of dimensions which the model should have. Default is 2.
    shape : tuple
        Shape of the grid (including damping region)
    extent : tuple
        Extent of the grid (including damping region)
    space_order : int
        Order of the spatial discretization
    c : float
        Acoustic wavespeed in air
    src_coords : ndarray
        Coordinates for source location
    src_f : float
        Source peak frequency. Default is 1Hz.
    rec_coords : ndarray
        Coordinates for receiver location
    t0 : float
        Starting time of the model
    tn : float
        End time of the model
    dt : float
        Timestep
    """
    def __init__(self, *args, **kwargs):
        self._ndims = kwargs.get('dims', 2)
        if self._ndims not in (2, 3):
            raise ValueError("Model can be 2D or 3D only")
        self._generate_dims()
        self._setup_subdomains()

        self._shape = kwargs.get('shape', (201, 201))
        self._extent = kwargs.get('extent', (1000., 1000.))
        self._build_grid()

        self._c = kwargs.get('c', 350.)  # Celerity
        self._space_order = kwargs.get('space_order', 4)
        self._setup_fields()

        self._setup_damping()

        self._src_coords = kwargs.get('src_coords')
        self._rec_coords = kwargs.get('rec_coords')
        self._t0 = kwargs.get('t0')
        self._tn = kwargs.get('tn')
        self._dt = kwargs.get('dt')
        self._src_f = kwargs.get('src_f', 1.)
        self._setup_sparse()

    def _generate_dims(self):
        """Set up dimensions according to number specified"""
        xyz = ('x', 'y', 'z')
        dims = tuple([dv.SpaceDimension(name) for name in xyz[:self._ndims]])
        self._dims = dims

    def _setup_subdomains(self):
        """Set up subdomains for boundary condition equations"""
        # Note that subdomain thickness hardcoded here
        lmr = (('left', 10), ('middle', 10, 10), ('right', 10))
        names = ('l', 'm', 'r')

        sdspecs = product(lmr, repeat=self._ndims)
        sdnames = product(names, repeat=self._ndims)
        sds = []  # Subdomains
        for spec, name in zip(sdspecs, sdnames):
            sds.append(BoundaryDomain(name="".join(name), spec=spec))
        self._subdomains = tuple(sds)

    def _build_grid(self):
        """Build the numerical grid and attach the subdomains"""
        self._grid = dv.Grid(shape=self.shape, extent=self.extent,
                             dimensions=self.dims, subdomains=self.subdomains)

    def _setup_fields(self):
        """Set up the various model fields"""
        # Pressure
        self._p = dv.TimeFunction(name='p', grid=self.grid, time_order=2,
                                  space_order=self.space_order)
        # Auxilliary field used for damping mask
        self._A = dv.VectorTimeFunction(name='p', grid=self.grid, time_order=2,
                                        space_order=self.space_order)

        # Damping mask
        self._damp = dv.Function(name='damp', grid=self.grid)

        # Signed distance function for boundary
        self._sdf = dv.Function(name='sdf', grid=self.grid)

    def _setup_damping(self):
        """Initialise the damping mask"""
        c = self.c  # Shorthand
        d = self.damp
        grid = self.grid
        # Hardcoded for 10 layers of PMLs
        eqs = []
        for i in range(self._ndims):
            dist = dv.Max(dv.Max(10-grid.dimensions[i],
                                 grid.dimensions[i]-grid.shape[i]+10), 0)
            eq = dv.Eq(d, d + (3*c/20)*(dist/10)**2*(1/np.log(1e-5)))
            eqs.append(eq)

        op_damp = dv.Operator(eqs, name='initdamp')
        op_damp()
        # REMOVE LATER: temporary boundary mask plot
        # plt.imshow(self.damp.data)
        # plt.colorbar()
        # plt.show()

    def _setup_sparse(self):
        """Initialise the sources and receivers"""
        time_range = TimeAxis(start=self.t0, stop=self.tn, step=self.dt)
        if self._src_coords is not None:
            self._src = RickerSource(name='src', grid=self.grid,
                                     f0=self._src_f,
                                     npoint=self._src_coords.shape[0],
                                     time_range=time_range)
            self._src.coordinates.data[:] = self._src_coords
        else:
            self._src = None

        if self._rec_coords is not None:
            self._rec = Receiver(name='rec', grid=self.grid,
                                 npoint=self._rec_coords.shape[0],
                                 time_range=time_range)
            self._rec.coordinates.data[:] = self._rec_coords
        else:
            self._rec = None

    @property
    def dims(self):
        """Model dimensions"""
        return self._dims

    @property
    def subdomains(self):
        """Grid subdomains"""
        return self._subdomains

    @property
    def shape(self):
        """Shape of the numerical grid"""
        return self._shape

    @property
    def extent(self):
        """Extent of the numerical grid"""
        return self._extent

    @property
    def grid(self):
        """The computational grid"""
        return self._grid

    @property
    def space_order(self):
        """The order of the spatial discretization"""
        return self._space_order

    @property
    def c(self):
        """Acoustic wavespeed (celerity)"""
        return self._c

    @property
    def p(self):
        """Pressure"""
        return self._p

    @property
    def A(self):
        """Auxilliary damping field"""
        return self._A

    @property
    def damp(self):
        """Damping mask"""
        return self._damp

    @property
    def sdf(self):
        """Signed distance function for boundary"""
        return self._sdf

    @property
    def t0(self):
        """Start time"""
        return self._t0

    @property
    def tn(self):
        """End time"""
        return self._tn

    @property
    def dt(self):
        """Timestep"""
        return self._dt

    @property
    def src(self):
        """Ricker source"""
        return self._src

    @property
    def rec(self):
        """The receiver array"""
        return self._rec


if __name__ == '__main__':
    ifm = InfrasoundModel(dims=2)
