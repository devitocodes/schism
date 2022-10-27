"""
A simple example demonstrating an infrasound propagation simulation with
topography.
"""

import numpy as np
import matplotlib.pyplot as plt

from model import InfrasoundModel
from propagator import InfrasoundPropagator


src_coords = np.array([4800., 2550.])[np.newaxis, :]
rec_coords = np.array([[2000., 1900.], [7505., 3025.],
                       [7200., 1275.], [2150., 3300.],
                       [3000., 1000.], [6505, 4500.]])
# rec_coords = np.array([[2400., 1275.], [7200., 3825.],
#                        [7200., 1275.], [2400., 3825.]])
t0, tn, dt = 0., 13., 0.021  # Courant number ~0.25
src_f = 1.
model = InfrasoundModel(dims=2, shape=(321, 171), extent=(9600., 5100.),
                        src_coords=src_coords, rec_coords=rec_coords,
                        t0=t0, tn=tn, dt=dt, src_f=src_f)

propagator = InfrasoundPropagator(model=model, mode='forward')
propagator.run()

print(np.linalg.norm(model.p.data[-1]))

plt.imshow(model.p.data[-1].T)
plt.colorbar()
plt.show()

print(np.linalg.norm(model.rec.data[:, 0]))
plt.plot(model.rec.data[:, 0])
plt.show()

# Reset the fields
model.p.data[:] = 0
model.p_aux[0].data[:] = 0
model.p_aux[1].data[:] = 0
model.A[0].data[:] = 0
model.A[1].data[:] = 0

bpropagator = InfrasoundPropagator(model=model, mode='adjoint',
                                   track_zsc=True)
bpropagator.run()

print(np.linalg.norm(model.p.data[-1]))
plt.imshow(model.p.data[-1].T)
plt.colorbar()
plt.show()

plt.imshow(model.zsc.data.T)
plt.colorbar()
plt.show()
