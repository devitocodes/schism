"""
A simple example demonstrating an infrasound propagation simulation with
topography.
"""

import numpy as np
import matplotlib.pyplot as plt

from model import InfrasoundModel
from propagator import InfrasoundPropagator


src_coords = np.array([4800., 2250.])[np.newaxis, :]
# rec_coords = np.array([[2000., 4400.], [7505., 3025.],
#                        [7200., 1275.], [2150., 3300.],
#                        [3000., 1000.], [6505, 4500.]])
rec_coords = np.array([[3000., 2250.], [2000., 1800.],
                       [1000., 1600.], [6500., 2300.],
                       [7500., 1800.], [8500, 1500.],
                       [2500., 4000.], [7000., 4000.]])
t0, tn, dt = 0., 13., 0.021  # Courant number ~0.25
src_f = 1.
sdf_data = -np.load('surface_files/mt_st_helens_2d.npy')
# Plot extent
plt_ext = (0., 9600., 0., 5100.)
plt.imshow(sdf_data.T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()
model = InfrasoundModel(dims=2, shape=(321, 171), extent=(9600., 5100.),
                        src_coords=src_coords, rec_coords=rec_coords,
                        t0=t0, tn=tn, dt=dt, src_f=src_f, sdf_data=sdf_data,
                        boundary=True)


propagator = InfrasoundPropagator(model=model, mode='forward')
propagator.run()

print(np.linalg.norm(model.p.data[-1]))

plt.imshow(model.p.data[-1].T, origin='lower', extent=plt_ext)
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

# Normalise the recordings
max_amplitude = np.amax(np.abs(model.rec.data), axis=0)
model.rec.data[:] /= max_amplitude[np.newaxis, :]

bpropagator = InfrasoundPropagator(model=model, mode='adjoint',
                                   track_zsc=True)
bpropagator.run()

print(np.linalg.norm(model.p.data[-1]))
plt.imshow(model.p.data[-1].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plt.imshow(model.zsc.data.T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()
