"""
Propagator for infrasound models. Encapsulates the equations and the finite
difference operators.

Note that this is intended as a demonstration of a potential use-case of
Schism and Devito, rather than a code to be used in anger. However, it should
hopefully illustrate one approach by which such a code may be implemented.
"""

import devito as dv


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
        self._setup_equations()

    def _setup_equations(self):
        """Set up the update equations"""
        self._p_eqs = []  # List of pressure update equations
        self._A_eqs = []  # List of auxilliary field equations

        # Shorthands
        grid = self.model.grid
        p = self.model.p
        c = self.model.c

        if self.mode == 'forward':
            # Timestep to update
            pu = p.forward
        else:
            pu = p.backward

        main_name = 'm'*self.model._ndims
        main_domain = grid.subdomains[main_name]
        self._eqs.append(dv.Eq(pu,
                               dv.solve(p.dt2 - c**2*p.laplace, pu),
                               subdomain=main_domain))

    def run(self, t_m=None, t_M=None):
        """Run the propagator"""
        self.operator(t_m=t_m, t_M=t_M, dt=self.model.dt)

    @property
    def model(self):
        """Model on which the propagator is to be run"""
        return self._model

    @property
    def mode(self):
        """Mode of propagation"""
        return self._mode

    @dv.cached_property
    def operator(self):
        """The finite-difference operator"""
        return dv.Operator(self._A_eqs + self._p_eqs + self.recording,
                           name=self.mode)
