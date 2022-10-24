"""
A simple example demonstrating an infrasound propagation simulation with
topography.
"""

import numpy as np
import matplotlib.pyplot as plt

from model import InfrasoundModel
from propagator import InfrasoundPropagator


src_coords = np.array([500., 500.])[np.newaxis, :]
rec_coords = np.array([[250., 250.], [750., 750.]])
t0, tn, dt = 0., 2.5, 0.007  # Courant number ~0.5
src_f = 2.
model = InfrasoundModel(dims=2, shape=(201, 201), extent=(1000., 1000.),
                        src_coords=src_coords, rec_coords=rec_coords,
                        t0=t0, tn=tn, dt=dt, src_f=src_f)

propagator = InfrasoundPropagator(model=model, mode='forward')
propagator.run()

print(np.linalg.norm(model.p.data[-1]))

plt.imshow(model.p.data[-1, 10:-10, 10:-10])
plt.colorbar()
plt.show()
