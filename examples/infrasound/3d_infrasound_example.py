"""
A simple example demonstrating an infrasound propagation simulation with
topography in 3D. Runs a source localisation method based off of Kim and Lees
2014 with topography implemented as an immersed boundary. Note that the metric
used is altered for simplicity however.
"""

import numpy as np
import matplotlib.pyplot as plt

from model import InfrasoundModel
from propagator import InfrasoundPropagator
from plotting import plot_st_helens, plot_top_down

src_coords = np.array([4800., 4800., 2250.])[np.newaxis, :]
rec_coords = np.array([[4800., 1400., 1400.],
                       [8200., 4800., 1500.],
                       [1400., 4800., 1500.],
                       [4800., 8200., 1650.],
                       [2400., 2400., 1400.],
                       [7200., 2400., 1500.],
                       [2400., 7200., 1550.],
                       [7200., 7200., 1500.]])

# 16s is plenty
t0, tn, dt = 0., 26., 0.021  # Courant number ~0.25 (could be increased)
src_f = 1.  # Source frequency is 1Hz

sdf_data = -np.load('surface_files/mt_st_helens_3d.npy')
# Plot extent
plt_ext = (0., 9600., 0., 5100.)
xmid = 321//2
plt.imshow(sdf_data[xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()
model = InfrasoundModel(dims=3, shape=(321, 321, 171),
                        extent=(9600., 9600., 5100.),
                        space_order=4,
                        src_coords=src_coords, rec_coords=rec_coords,
                        t0=t0, tn=tn, dt=dt, src_f=src_f, sdf_data=sdf_data,
                        boundary=True)

propagator = InfrasoundPropagator(model=model, mode='forward')
propagator.run()

print(np.linalg.norm(model.p.data[-1]))


plt.imshow(model.p.data[-1, xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plt.imshow(model.p.data[-1, :, xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plot_st_helens(model.p.data[-1], src_coords, rec_coords,
               np.array([-4800., -4800., 0.]), (30, 30, 30))

for i in range(rec_coords.shape[0]):
    print(np.linalg.norm(model.rec.data[:, i]))
    plt.plot(model.rec.data[:, i])
    plt.show()

# Next step is backpropagation
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
plt.imshow(model.p.data[-1, xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plt.imshow(model.p.data[-1, :, xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plot_st_helens(model.p.data[-1], src_coords, rec_coords,
               np.array([-4800., -4800., 0.]), (30, 30, 30))

plt.imshow(model.zsc.data[-1, xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plt.imshow(model.zsc.data[-1, :, xmid].T, origin='lower', extent=plt_ext)
plt.colorbar()
plt.show()

plot_st_helens(model.zsc.data[-1], src_coords, rec_coords,
               np.array([-4800., -4800., 0.]), (30, 30, 30))

plot_top_down(model.zsc.data[-1], -4800., 4800., -4800., 4800.)
