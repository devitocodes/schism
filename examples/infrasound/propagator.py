"""
Propagator for infrasound models. Encapsulates the equations and the finite
difference operators including PML absorbing boundary conditions.

Note that this is intended as a demonstration of a potential use-case of
Schism and Devito, rather than a code to be used in anger. However, it should
hopefully illustrate one approach by which such a code may be implemented.
"""

import devito as dv
import sympy as sp
import numpy as np

from cached_property import cached_property


class InfrasoundPropagator:
    """
    Propagator for infrasound models. Encapsulates the equations and the finite
    difference operators.

    Parameters
    ----------
    model : InfrasoundModel
        The model on which the propagator should be run
    mode : str
        Mode of propagation. Can be 'forward' or 'adjoint'. Default is
        'forward'.
    track_zsc : bool
        Track the z-score. This is a measure of energy concentration devised
        by Kim and Lees 2014
    """
    def __init__(self, *args, **kwargs):
        self._model = kwargs.get('model')
        if self.model is None:
            raise ValueError("No model provided")
        self._mode = kwargs.get('mode', 'forward')
        self._setup_dense()
        self._setup_sparse()
        self._track_zsc = kwargs.get('track_zsc', False)
        self._setup_heuristics()

    def _setup_dense(self):
        """Set up the update equations (for dense kernels)"""
        self._p_eqs = []  # List of pressure update equations
        self._p_aux_eqs = []  # List of auxilliary pressure update equations
        self._A_eqs = []  # List of auxilliary field equations

        # Shorthands
        grid = self.model.grid
        p = self.model.p
        p_aux = self.model.p_aux
        A = self.model.A
        c = self.model.c
        dt = self.model.dt
        d = self.model.damp

        main_name = 'm'*self.model._ndims

        if self.mode == 'forward':
            def fwd(f):
                return f.forward

            def bwd(f):
                return f.backward

        else:
            def fwd(f):
                return f.backward

            def bwd(f):
                return f.forward

        # Immersed boundary things
        subs = self.model._subs

        for name, subdomain in grid.subdomains.items():
            if name == main_name:
                self._p_eqs.append(dv.Eq(fwd(p),
                                         2*p - bwd(p) + dt**2*c**2*(subs[p.dx2]+subs[p.dy2]),
                                         subdomain=subdomain))

                self._A_eqs.append(dv.Eq(fwd(A),
                                         A + dt*dv.grad(p),
                                         subdomain=subdomain))
            else:
                # PML regions
                # Update auxilliary field
                # Update auxilliary pressure
                # Update pressure
                for i in range(len(grid.dimensions)):
                    dim = grid.dimensions[i]
                    eq = dv.Eq(fwd(p_aux[i]),
                               p_aux[i]+dt*(c**2*dv.Derivative(fwd(A[i]), dim)
                                            - d[i]*p_aux[i]),
                               subdomain=subdomain)
                    self._p_aux_eqs.append(eq)

                self._p_eqs.append(dv.Eq(fwd(p),
                                   sum([fwd(p_aux[i])
                                        for i in range(len(grid.dimensions))]),
                                   subdomain=subdomain))

                self._A_eqs.append(dv.Eq(fwd(A),
                                         A + dt*(dv.grad(p)
                                         - sp.hadamard_product(d, A)),
                                         subdomain=subdomain))

    def _setup_sparse(self):
        """Set up sparse source and receiver terms"""
        p = self.model.p  # Shorthand
        self._sparse = []

        if self.mode == 'forward':
            # Timestep to update
            pf = p.forward
            if self.model.src is not None:
                self._sparse.append(self.model.src.inject(field=pf,
                                                          expr=self.model.src))

            if self.model.rec is not None:
                self._sparse.append(self.model.rec.interpolate(expr=pf))
        else:
            pf = p.backward
            if self.model.rec is not None:
                self._sparse.append(self.model.rec.inject(field=pf,
                                                          expr=self.model.rec))

    def _setup_heuristics(self):
        """Set up tracking terms"""
        if self.mode == 'forward':
            def fwd(f):
                return f.forward
        else:
            def fwd(f):
                return f.backward

        p = self.model.p
        zsc = self.model.zsc
        c = self.model.c
        dt = self.model.dt
        self._heuristics = []

        main_name = 'm'*self.model._ndims

        if self._track_zsc:
            grid_size = np.prod(self.model.grid.shape)
            # Mean of field
            mu = dv.TimeFunction(name='mu', dimensions=(p.time_dim,),
                                 time_order=2, shape=(self.model.src.nt,))
            self._heuristics.append(dv.Inc(mu, fwd(p)/grid_size))
            # Squared standard deviation
            sigma2 = dv.TimeFunction(name='sigma2', dimensions=(p.time_dim,),
                                     time_order=2, shape=(self.model.src.nt,))
            self._heuristics.append(dv.Inc(sigma2,
                                           (fwd(p)-mu)**2/grid_size))
            sigma = dv.sqrt(sigma2)
            # Scaling for geometric attenuation (approximate)
            scale = dv.TimeFunction(name='scale', dimensions=(p.time_dim,),
                                    time_order=2, shape=(self.model.src.nt,))
            self._heuristics.append(dv.Eq(fwd(scale), scale + c*dt))
            zsc_eq = dv.Eq(fwd(zsc),
                           dv.Max(zsc, scale**2*(fwd(p)-mu)/sigma),
                           subdomain=self.model.grid.subdomains[main_name])
            self._heuristics.append(zsc_eq)

    def run(self):
        """Run the propagator"""
        # FIXME: Add t_m and t_M back (will need to construct a dict)
        self.operator(dt=self.model.dt)

    @property
    def model(self):
        """Model on which the propagator is to be run"""
        return self._model

    @property
    def mode(self):
        """Mode of propagation"""
        return self._mode

    @cached_property
    def operator(self):
        """The finite-difference operator"""
        return dv.Operator(self._A_eqs + self._p_aux_eqs
                           + self._p_eqs + self._sparse
                           + self._heuristics,
                           name=self.mode)
