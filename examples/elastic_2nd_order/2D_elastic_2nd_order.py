"""
Simple example implementing a free-surface with the 2nd-order formulation
of the elastic wave equation, as given by Kelly et al. 1976. This code
will save four snapshots at intervals of 0.2*tn, where tn is the end time
(zero timestep will not be outputted). This code will use real-world
topography taken from a Digital Elevation Map (DEM) of Mt St Helens, USA.
The constant material properties ensure all reflections are products of
the immersed boundary treatment, rather then resulting from any
discontinuity.
"""

import matplotlib.pyplot as plt
import numpy as np
import devito as dv
import os

from schism import BoundaryGeometry, BoundaryConditions, Boundary
from examples.seismic import TimeAxis, RickerSource


def run(sdf, s_o, nsnaps):
    """Run a forward model if no file found to read"""
    grid = sdf.grid
    bg = BoundaryGeometry(sdf)

    # Create fields
    ux = dv.TimeFunction(name='ux', grid=grid,
                         space_order=s_o, time_order=2)
    uy = dv.TimeFunction(name='uy', grid=grid,
                         space_order=s_o, time_order=2)

    nx = bg.n[0]
    ny = bg.n[1]

    v_p = 2.5  # km/s
    v_s = 1.5  # km/s
    rho = 3.  # tons/m^3

    # Lame parameters
    mu = rho*v_s**2
    lam = rho*v_p**2 - 2*mu

    # For convenience
    b = 1/rho

    bc_list = [dv.Eq(nx*(lam+2*mu)*ux.dx + nx*lam*uy.dy
                     + ny*mu*ux.dy + ny*mu*uy.dx, 0),
               dv.Eq(nx*mu*ux.dy + nx*mu*uy.dx
                     + ny*(lam+2*mu)*uy.dy + ny*lam*ux.dx, 0)]

    if s_o >= 4:
        bc_list.append(dv.Eq(lam*mu*nx*uy.dy3 + mu**2*ny*ux.dy3
                             + ux.dx2dy*ny*(lam**2 + 2*lam*mu + 2*mu**2)
                             + ux.dx3*nx*(lam**2 + 4*lam*mu + 4*mu**2)
                             + ux.dxdy2*nx*(2*lam*mu + 3*mu**2)
                             + uy.dx2dy*nx*(lam**2 + 3*lam*mu + mu**2)
                             + uy.dx3*ny*(lam*mu + 2*mu**2)
                             + uy.dxdy2*ny*(lam**2 + 3*lam*mu + 3*mu**2), 0))
        bc_list.append(dv.Eq(lam*mu*ny*ux.dx3 + mu**2*nx*uy.dx3
                             + ux.dx2dy*nx*(lam**2 + 3*lam*mu + 3*mu**2)
                             + ux.dxdy2*ny*(lam**2 + 3*lam*mu + mu**2)
                             + ux.dy3*nx*(lam*mu + 2*mu**2)
                             + uy.dx2dy*ny*(2*lam*mu + 3*mu**2)
                             + uy.dxdy2*nx*(lam**2 + 2*lam*mu + 2*mu**2)
                             + uy.dy3*ny*(lam**2 + 4*lam*mu + 4*mu**2), 0))

    # TODO: add higher-order bcs
    bcs = BoundaryConditions(bc_list)
    boundary = Boundary(bcs, bg)

    derivs = (ux.dx2, ux.dy2, ux.dxdy, uy.dx2, uy.dy2, uy.dxdy)
    subs = boundary.substitutions(derivs)

    t0 = 0.  # Simulation starts a t=0
    tn = 1500.  # Simulation last 0.8 seconds (800 ms)
    # Note: grid increment hardcoded, courant number 0.5
    dt = 0.5*30/v_p  # Time step from grid spacing

    # Stability check
    alpha = np.sqrt((lam+2*mu)/rho)
    beta = np.sqrt(mu/rho)
    dx = np.amin(grid.spacing)
    print(dt)
    print("Must be smaller than")
    print(dx/np.sqrt(alpha**2 + beta**2))

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.0065  # Source peak frequency is 6.5Hz (0.0065 kHz)
    src = RickerSource(name='src', grid=grid, f0=f0,
                       npoint=1, time_range=time_range)

    src.coordinates.data[0, 0] = 4800.
    src.coordinates.data[0, 1] = 2750.

    # Set up snapshotting
    steps = src.nt
    factor = int(steps/nsnaps)
    t_sub = dv.ConditionalDimension('t_sub', parent=grid.time_dim,
                                    factor=factor)

    # Buffer size needs to be more robust
    uxsave = dv.TimeFunction(name='uxsave', grid=grid, time_order=0,
                             save=nsnaps+1, time_dim=t_sub)
    uysave = dv.TimeFunction(name='uysave', grid=grid, time_order=0,
                             save=nsnaps+1, time_dim=t_sub)

    rhs_ux = b*(lam+2*mu)*ux.dx2 + b*mu*ux.dy2 + b*(lam+mu)*uy.dxdy
    rhs_uy = b*(lam+2*mu)*uy.dy2 + b*mu*uy.dx2 + b*(lam+mu)*ux.dxdy

    eq_ux = dv.Eq(ux.forward,
                  2*ux - ux.backward
                  + dt**2*rhs_ux.subs(subs))

    eq_uy = dv.Eq(uy.forward,
                  2*uy - uy.backward
                  + dt**2*rhs_uy.subs(subs))

    eq_save_ux = dv.Eq(uxsave, ux)
    eq_save_uy = dv.Eq(uysave, uy)

    src_ux = src.inject(field=ux.forward, expr=src)

    op = dv.Operator([eq_ux, eq_uy, eq_save_ux, eq_save_uy]
                     + src_ux)
    op(time=time_range.num-1, dt=dt)

    return uxsave.data, uysave.data


def plot_snaps(vxsave_data, vysave_data, shift, sdf):
    # FIXME: Hardcoded for four snapshots
    # Plot extent
    plt_ext = (0., 9600., 0.-shift*30., 5100.-shift*30.)

    # Plot surface with SDF contours
    xvals = np.linspace(0., 9600., vxsave_data.shape[1])
    yvals = np.linspace(0.-shift*30., 5100.-shift*30., vxsave_data.shape[2])
    xmsh, ymsh = np.meshgrid(xvals, yvals, indexing='ij')

    fig, axs = plt.subplots(4, 2, constrained_layout=True,
                            figsize=(9.6, 5.1*2),
                            sharex=True, sharey=True)

    for i in range(1, 5):
        vmax = np.amax(np.abs(vxsave_data[i]))
        vmin = -vmax
        axs[i-1, 0].imshow(vxsave_data[i].T, origin='lower',
                           extent=plt_ext, vmax=vmax, vmin=vmin,
                           cmap='seismic')
        axs[i-1, 0].contour(xmsh, ymsh, sdf.data, [0])
        if i == 4:
            axs[i-1, 0].set_xlabel("Distance (m)")
        axs[i-1, 0].set_ylabel("Elevation (m)")
        axs[i-1, 0].set_yticks([-1000., 0., 1000., 2000., 3000.])

    for i in range(1, 5):
        vmax = np.amax(np.abs(vysave_data[i]))
        vmin = -vmax
        axs[i-1, 1].imshow(vysave_data[i].T, origin='lower',
                           extent=plt_ext, vmax=vmax, vmin=vmin,
                           cmap='seismic')
        axs[i-1, 1].contour(xmsh, ymsh, sdf.data, [0])
        if i == 4:
            axs[i-1, 1].set_xlabel("Distance (m)")
        axs[i-1, 1].set_yticks([-1000., 0., 1000., 2000., 3000.])
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
    # Load the signed distance function data
    sdf_file = "/../infrasound/surface_files/mt_st_helens_2d.npy"
    sdf = load_sdf(sdf_file, s_o, shift)

    outfile_vx = append_path("/2D_elastic_2nd_order_vx_snaps.npy")
    outfile_vy = append_path("/2D_elastic_2nd_order_vy_snaps.npy")

    nsnaps = 4  # Number of snaps

    # If not run then run and save output
    # If output found then plot it
    try:
        vxsave_data = np.load(outfile_vx)
        vysave_data = np.load(outfile_vy)
        plot_snaps(vxsave_data, vysave_data, shift, sdf)
    except FileNotFoundError:
        vxsave_data, vysave_data = run(sdf, s_o, nsnaps)
        np.save(outfile_vx, vxsave_data)
        np.save(outfile_vy, vysave_data)


if __name__ == "__main__":
    main()