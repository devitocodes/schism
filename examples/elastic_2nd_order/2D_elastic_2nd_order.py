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


def get_iso_bcs(nx, ny, ux, uy, lam, mu, s_o):
    """Returns boundary conditions for the isotropic case"""
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

    return bc_list


def get_vti_bcs(nx, ny, ux, uy, v_p2, v_s2, ep, de, s_o):
    """Returns boundary conditions for the vti case"""
    # More shorthands
    v_p4 = v_p2**2
    v_s4 = v_s2**2

    # Note that a factor of rho has been removed here
    txx = (1+2*ep)*v_p2*ux.dx + (de*v_p2-v_p2+2*v_s2)*uy.dy
    tyy = (1+2*ep)*v_p2*uy.dy + (de*v_p2-v_p2+2*v_s2)*ux.dx
    txy = v_s2*ux.dy + v_s2*uy.dx

    # With fourth-order boundary conditions
    bc_list = [dv.Eq(nx*txx + ny*txy, 0),
               dv.Eq(nx*txy + ny*tyy, 0)]

    if s_o >= 4:
        bc4 = [dv.Eq(ny*ux.dy3*v_s4
                     + ux.dx2dy*ny*(de*v_p2*np.sqrt(2*de*v_p4
                                                    - 2*de*v_p2*v_s2
                                                    + v_p4
                                                    - 2*v_p2*v_s2
                                                    + v_s4)
                                    + 2*ep*v_p2*v_s2 + ny*v_p2*v_s2
                                    - v_p2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2
                                                   + v_p4 - 2*v_p2*v_s2
                                                   + v_s4)
                                    + 2*v_s2*np.sqrt(2*de*v_p4
                                                     - 2*de*v_p2*v_s2
                                                     + v_p4
                                                     - 2*v_p2*v_s2
                                                     + v_s4))
                     + ux.dx3*nx*(4*ep**2*v_p4 + 4*ep*v_p4 + v_p4)
                     + ux.dxdy2*nx*(2*ep*v_p2*v_s2 + v_p2*v_s2
                                    + v_s2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2
                                                   + v_p4 - 2*v_p2*v_s2
                                                   + v_s4))
                     + uy.dx2dy*nx*(2*de*ep*v_p4 + de*v_p4 - 2*ep*v_p4
                                    + 4*ep*v_p2*v_s2 - v_p4
                                    + 2*v_p2*v_s2
                                    + v_s2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2
                                                   + v_p4
                                                   - 2*v_p2*v_s2
                                                   + v_s4))
                     + uy.dx3*ny*(2*ep*v_p2*v_s2 + v_p2*v_s2)
                     + uy.dxdy2*ny*(2*ep*v_p2*np.sqrt(2*de*v_p4
                                                      - 2*de*v_p2*v_s2
                                                      + v_p4
                                                      - 2*v_p2*v_s2
                                                      + v_s4)
                                    + v_p2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2
                                                   + v_p4
                                                   - 2*v_p2*v_s2
                                                   + v_s4)
                                    + v_s4)
                     + uy.dy3*nx*(de*v_p2*v_s2 - v_p2*v_s2 + 2*v_s4), 0),
               dv.Eq(nx*ux.dy3*v_p2*v_s2 + nx*uy.dx3*v_s4
                     + ux.dx2dy*nx*(2*ep*v_p2*np.sqrt(2*de*v_p4
                                                      - 2*de*v_p2*v_s2
                                                      + v_p4
                                                      - 2*v_p2*v_s2
                                                      + v_s4)
                                    + v_p2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2
                                                   + v_p4
                                                   - 2*v_p2*v_s2 + v_s4)
                                    + v_s4)
                     + ux.dx3*ny*(de*v_p2*v_s2 - v_p2*v_s2 + 2*v_s4)
                     + ux.dxdy2*ny*(de*v_p4 - v_p4 + 2*v_p2*v_s2
                                    + v_s2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2
                                                   + v_p4 - 2*v_p2*v_s2
                                                   + v_s4))
                     + uy.dx2dy*ny*(2*ep*v_p2*v_s2 + v_p2*v_s2
                                    + v_s2*np.sqrt(2*de*v_p4
                                                   - 2*de*v_p2*v_s2 + v_p4
                                                   - 2*v_p2*v_s2 + v_s4))
                     + uy.dxdy2*nx*(de*v_p2*np.sqrt(2*de*v_p4
                                                    - 2*de*v_p2*v_s2 + v_p4
                                                    - 2*v_p2*v_s2 + v_s4)
                                    + v_p2*v_s2
                                    - nx*v_p2*np.sqrt(2*de*v_p4
                                                      - 2*de*v_p2*v_s2
                                                      + v_p4 - 2*v_p2*v_s2
                                                      + v_s4)
                                    + 2*v_s2*np.sqrt(2*de*v_p4
                                                     - 2*de*v_p2*v_s2 + v_p4
                                                     - 2*v_p2*v_s2 + v_s4))
                     + uy.dy3*ny*(2*ep*v_p4 + v_p4))]

        bc_list += bc4

    return bc_list


def run(sdf, s_o, nsnaps, mode):
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

    # For convenience
    b = 1/rho

    if mode == 'iso':
        # Lame parameters
        mu = rho*v_s**2
        lam = rho*v_p**2 - 2*mu
        bc_list = get_iso_bcs(nx, ny, ux, uy, lam, mu, s_o)
    elif mode == 'vti':
        # Anisotropy parameters
        de = 0.1
        ep = 0.25

        v_p2 = v_p**2
        v_s2 = v_s**2
        v_pn2 = (1+2*de)*v_p**2
        bc_list = get_vti_bcs(nx, ny, ux, uy, v_p2, v_s2, ep, de, s_o)

    # TODO: add higher-order bcs
    bcs = BoundaryConditions(bc_list)
    boundary = Boundary(bcs, bg)

    derivs = (ux.dx2, ux.dy2, ux.dxdy, uy.dx2, uy.dy2, uy.dxdy)
    subs = boundary.substitutions(derivs)

    t0 = 0.  # Simulation starts a t=0
    tn = 1500.  # Simulation last 0.8 seconds (800 ms)
    # Note: grid increment hardcoded, courant number 0.5
    dt = 0.5*30/v_p  # Time step from grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.006  # Source peak frequency is 6Hz (0.006 kHz)
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

    if mode == 'iso':
        rhs_ux = b*(lam+2*mu)*ux.dx2 + b*mu*ux.dy2 + b*(lam+mu)*uy.dxdy
        rhs_uy = b*(lam+2*mu)*uy.dy2 + b*mu*uy.dx2 + b*(lam+mu)*ux.dxdy
    elif mode == 'vti':
        rhs_ux = v_p2*(1+2*ep)*ux.dx2 + v_s2*ux.dy2 \
            + np.sqrt((v_p2-v_s2)*(v_pn2-v_s2))*uy.dxdy
        rhs_uy = v_s2*uy.dx2 + v_p2*uy.dy2 \
            + np.sqrt((v_p2-v_s2)*(v_pn2-v_s2))*ux.dxdy

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


def plot_snaps(uxsave_data, uysave_data, shift, sdf):
    # FIXME: Hardcoded for four snapshots
    # Plot extent
    plt_ext = (0., 9600., 0.-shift*30., 5100.-shift*30.)

    # Plot surface with SDF contours
    xvals = np.linspace(0., 9600., uxsave_data.shape[1])
    yvals = np.linspace(0.-shift*30., 5100.-shift*30., uxsave_data.shape[2])
    xmsh, ymsh = np.meshgrid(xvals, yvals, indexing='ij')

    fig, axs = plt.subplots(4, 2, constrained_layout=True,
                            figsize=(9.6, 5.1*2),
                            sharex=True, sharey=True)

    for i in range(1, 5):
        vmax = np.amax(np.abs(uxsave_data[i]))
        vmin = -vmax
        axs[i-1, 0].imshow(uxsave_data[i].T, origin='lower',
                           extent=plt_ext, vmax=vmax, vmin=vmin,
                           cmap='seismic')
        axs[i-1, 0].contour(xmsh, ymsh, sdf.data, [0])
        if i == 4:
            axs[i-1, 0].set_xlabel("Distance (m)")
        axs[i-1, 0].set_ylabel("Elevation (m)")
        axs[i-1, 0].set_yticks([-1000., 0., 1000., 2000., 3000.])

    for i in range(1, 5):
        vmax = np.amax(np.abs(uysave_data[i]))
        vmin = -vmax
        axs[i-1, 1].imshow(uysave_data[i].T, origin='lower',
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

    # Mode, can be 'iso' or 'vti
    mode = 'vti'

    outfile_ux = append_path("/2D_elastic_2nd_order_ux_snaps_"
                             + mode + ".npy")
    outfile_uy = append_path("/2D_elastic_2nd_order_uy_snaps_"
                             + mode + ".npy")

    nsnaps = 4  # Number of snaps

    # If not run then run and save output
    # If output found then plot it
    try:
        uxsave_data = np.load(outfile_ux)
        uysave_data = np.load(outfile_uy)
        plot_snaps(uxsave_data, uysave_data, shift, sdf)
    except FileNotFoundError:
        uxsave_data, uysave_data = run(sdf, s_o, nsnaps, mode)
        np.save(outfile_ux, uxsave_data)
        np.save(outfile_uy, uysave_data)


if __name__ == "__main__":
    main()
