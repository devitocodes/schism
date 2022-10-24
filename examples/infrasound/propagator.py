"""
Propagator for infrasound models. Encapsulates the equations and the finite
difference operators.

Note that this is intended as a demonstration of a potential use-case of
Schism and Devito, rather than a code to be used in anger. However, it should
hopefully illustrate one approach by which such a code may be implemented.
"""

import devito as dv

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
    """
    def __init__(self, *args, **kwargs):
        self._model = kwargs.get('model')
        if self.model is None:
            raise ValueError("No model provided")
        self._mode = kwargs.get('mode', 'forward')
        self._setup_dense()
        self._setup_sparse()

    def _setup_dense(self):
        """Set up the update equations (for dense kernels)"""
        self._p_eqs = []  # List of pressure update equations
        self._A_eqs = []  # List of auxilliary field equations

        # Shorthands
        grid = self.model.grid
        p = self.model.p
        A = self.model.A
        c = self.model.c
        dt = self.model.dt
        d = self.model.damp

        main_name = 'm'*self.model._ndims
        main_domain = grid.subdomains[main_name]

        if self.mode == 'forward':
            pf = p.forward
            pb = p.backward
            Af = A.forward
        else:
            pf = p.backward
            pb = p.forward
            Af = A.backward

        # FIXME: Manually check the direction of propagation
        for name, subdomain in grid.subdomains.items():
            if name == main_name:
                self._p_eqs.append(dv.Eq(pf,
                                         2*p - pb + dt**2*c**2*p.laplace,
                                         subdomain=subdomain))

                self._A_eqs.append(dv.Eq(Af,
                                         A + dt*dv.grad(p),
                                         subdomain=subdomain))
            else:
                self._p_eqs.append(dv.Eq(pf,
                                         p + dt*(c**2*dv.div(Af) + d*p),
                                         subdomain=subdomain))

                self._A_eqs.append(dv.Eq(Af,
                                         A + dt*(dv.grad(p) + d*A),
                                         subdomain=subdomain))

    def _setup_sparse(self):
        """Set up sparse source and receiver terms"""
        p = self.model.p  # Shorthand
        self._sparse = []

        if self.mode == 'forward':
            # Timestep to update
            pf = p.forward
        else:
            pf = p.backward

        if self.model.src is not None:
            self._sparse.append(self.model.src.inject(field=pf,
                                                      expr=self.model.src))

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
        return dv.Operator(self._A_eqs + self._p_eqs + self._sparse,
                           name=self.mode)
