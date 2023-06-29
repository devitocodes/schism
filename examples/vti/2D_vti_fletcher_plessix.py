"""
Simple example implementing a free-surface with ether Duveneck/Plessix VTI
or Fletcher et al's VTI formulation. This code will save four snapshots at
intervals of 0.2*tn, where tn is the end time (zero timestep will not be
outputted). This code will use real-world topography taken from a Digital
Elevation Map (DEM) of Mt St Helens, USA. The constant material properties
ensure all reflections are products of the immersed boundary treatment,
rather than resulting from any discontinuity in impedence. Alternatively,
the formulation of Plessix and Cao 2011 can be used (assuming
locally-constant) parameters. This is equivelant to the formulation of
Duveneck et al 2008.
"""

import matplotlib.pyplot as plt
import numpy as np
import devito as dv
import os

from schism import BoundaryGeometry, BoundaryConditions, Boundary
from examples.seismic import TimeAxis, RickerSource


def run(sdf, s_o, nsnaps, mode):
    """Run a forward model if no file found to read"""
    grid = sdf.grid
    bg = BoundaryGeometry(sdf)

    # Create fields
    p = dv.TimeFunction(name='p', grid=grid, space_order=s_o, time_order=2)
    q = dv.TimeFunction(name='q', grid=grid, space_order=s_o, time_order=2)

    bc_list = [dv.Eq(p, 0), dv.Eq(p.dx2+p.dy2, 0),
               dv.Eq(q, 0), dv.Eq(q.dx2+q.dy2, 0)]

    if s_o >= 4:
        bc_list.append(dv.Eq(p.dx4 + 2*p.dx2dy2 + p.dy4, 0))
        bc_list.append(dv.Eq(q.dx4 + 2*q.dx2dy2 + q.dy4, 0))

    # TODO: add higher-order bcs
    bcs = BoundaryConditions(bc_list)
    boundary = Boundary(bcs, bg)

    derivs = (p.dx2, p.dy2, q.dx2, q.dy2)
    subs = boundary.substitutions(derivs)

    v_py = 2.5  # km/s

    t0 = 0.  # Simulation starts a t=0
    tn = 1500.  # Simulation last 0.8 seconds (800 ms)
    # Note: grid increment hardcoded, courant number 0.5
    dt = 0.5*30/v_py  # Time step from grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.008  # Source peak frequency is 8Hz (0.008 kHz)
    src = RickerSource(name='src', grid=grid, f0=f0,
                       npoint=1, time_range=time_range)

    src.coordinates.data[0, 0] = 4800.
    src.coordinates.data[0, 1] = 2750.

    # Other material parameters concerning anisotropy
    ep = dv.Function(name='epsilon', grid=grid)
    de = dv.Function(name='delta', grid=grid)
    de.data[:] = 0.1
    ep.data[:] = 0.25
    sig = 0.5  # Sigma parameter to control pseudoshear

    v_pn = v_py*dv.sqrt(1+2*de)
    v_px = v_py*dv.sqrt(1+2*ep)

    if mode == 'fletcher':
        v_sy = dv.sqrt(v_py**2*(ep-de)/sig)
    elif mode == 'plessix':
        v_sy = 0
    else:
        raise ValueError("Unrecognised mode: " + mode)

    # Set up snapshotting
    steps = src.nt
    factor = int(steps/nsnaps)
    t_sub = dv.ConditionalDimension('t_sub', parent=grid.time_dim,
                                    factor=factor)

    # Buffer size needs to be more robust
    psave = dv.TimeFunction(name='psave', grid=grid, time_order=0,
                            save=nsnaps+1, time_dim=t_sub)

    rhs_p = v_px**2*p.dx2 + v_py**2*q.dy2 \
        + v_sy**2*p.dy2 - v_sy**2*q.dy2
    rhs_q = v_pn**2*p.dx2 + v_py**2*q.dy2 \
        - v_sy**2*p.dx2 + v_sy**2*q.dx2

    eq_p = dv.Eq(p.forward,
                 2*p - p.backward
                 + dt**2*rhs_p.subs(subs))

    eq_q = dv.Eq(q.forward,
                 2*q - q.backward
                 + dt**2*rhs_q.subs(subs))

    eq_save = dv.Eq(psave, p)

    src_p = src.inject(field=p.forward, expr=src*dt**2)
    src_q = src.inject(field=q.forward, expr=src*dt**2)

    op = dv.Operator([eq_p, eq_q, eq_save] + src_p + src_q)
    op(time=time_range.num-1, dt=dt)

    return psave.data


def plot_snaps(psave_data, shift, sdf):
    # FIXME: Hardcoded for four snapshots
    # Plot extent
    plt_ext = (0., 9600., 0.-shift*30., 5100.-shift*30.)

    # Plot surface with SDF contours
    xvals = np.linspace(0., 9600., psave_data.shape[1])
    yvals = np.linspace(0.-shift*30., 5100.-shift*30., psave_data.shape[2])
    xmsh, ymsh = np.meshgrid(xvals, yvals, indexing='ij')

    fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(9.6, 5.1),
                            sharex=True, sharey=True)

    for i in range(1, 5):
        vmax = np.amax(np.abs(psave_data[i]))
        vmin = -vmax
        if i > 2:
            j = 1
        else:
            j = 0
        axs[j, (i+1) % 2].imshow(psave_data[i].T, origin='lower',
                                 extent=plt_ext, vmax=vmax, vmin=vmin,
                                 cmap='seismic')
        axs[j, (i+1) % 2].contour(xmsh, ymsh, sdf.data, [0])
        if j == 1:
            axs[j, (i+1) % 2].set_xlabel("Distance (m)")
        if (i+1) % 2 == 0:
            axs[j, (i+1) % 2].set_ylabel("Elevation (m)")
        axs[j, (i+1) % 2].set_yticks([-1000., 0., 1000., 2000., 3000.])
    plt.show()


def load_sdf(file, s_o, shift):
    """Load a signed-distance function from file"""
    sdf_data = np.load(append_path(file))

    # Move the surface upwards by grid increments
    nx, ny = sdf_data.shape
    fill_vals = np.full((nx, shift), np.amax(sdf_data))
    sdf_shift = np.concatenate((fill_vals, sdf_data[:, :-shift]), axis=1)

    # Set up the grid
    # Note: size currently hardcoded
    extent = (9600., 5100.)
    grid = dv.Grid(shape=sdf_shift.shape, extent=extent)
    sdf = dv.Function(name='sdf', grid=grid, space_order=s_o)
    sdf.data[:] = sdf_shift
    return sdf


def append_path(file):
    """Turn a relative path into an absolute one"""
    path = os.path.dirname(os.path.abspath(__file__))
    return path + file


def main():
    shift = 50  # Number of grid increments to shift surface
    s_o = 4  # Space order
    mode = 'fletcher'  # VTI version (fletcher or plessix)
    # Load the signed distance function data
    sdf_file = "/../infrasound/surface_files/mt_st_helens_2d.npy"
    sdf = load_sdf(sdf_file, s_o, shift)

    outfile = append_path("/2D_vti_" + mode + "_snaps.npy")

    nsnaps = 4  # Number of snaps

    # If not run then run and save output
    # If output found then plot it
    try:
        psave_data = np.load(outfile)
        plot_snaps(psave_data, shift, sdf)
    except FileNotFoundError:
        psave_data = run(sdf, s_o, nsnaps, mode)
        np.save(outfile, psave_data)


if __name__ == "__main__":
    main()
