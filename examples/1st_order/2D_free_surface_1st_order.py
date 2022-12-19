"""
Simple example implementing free-surface with the 1st-order acoustic wave
equation. This code will save four snapshots at intervals of 0.2*tn, where tn
is the end time (zero timestep will not be outputted). This code will use
real-world topography taken from a Digital Elevation Map (DEM) of Mt St
Helens, USA. The constant material properties ensure all reflections are
products of the immersed boundary treatment, rather then resulting from any
discontinuity in impedence.
"""

import matplotlib.pyplot as plt
import numpy as np
import devito as dv
import sympy as sp
import os

from schism import BoundaryGeometry, BoundaryConditions, Boundary
from examples.seismic import TimeAxis, RickerSource


def run(sdf, sdf_x, sdf_y, s_o, nsnaps):
    """Run a forward model if no file found to read"""
    grid = sdf.grid
    x, y = grid.dimensions
    h_x = x.spacing
    h_y = y.spacing
    zero = sp.core.numbers.Zero()
    # Set cutoff for the velocity subgrids to zero
    # This means that any velocity point on the interior can be updated
    # This prevents non-physical ripple effects caused by over-extended extrapolations
    # By default, this cutoff is 0.5 (half a grid spacing)
    cutoff = {(h_x/2, zero): 0., (zero, h_y/2): 0.}

    bg = BoundaryGeometry((sdf, sdf_x, sdf_y), cutoff=cutoff)

    # Create pressure function
    p = dv.TimeFunction(name='p', grid=grid, space_order=s_o, time_order=1,
                        staggered=dv.NODE)
    # Create velocity vector function
    v = dv.VectorTimeFunction(name='v', grid=grid, space_order=s_o,
                              time_order=1)

    bc_list = [dv.Eq(p, 0),  # Zero pressure on free surface
               dv.Eq(p.dx2 + p.dy2, 0),  # Zero laplacian
               dv.Eq(v[0].dx + v[1].dy, 0)]  # Divergence of velocity equals zero
               

    if s_o >= 4:
        bc_list += [dv.Eq(p.dx4 + 2*p.dx2dy2 + p.dy4, 0),  # Zero biharmonic
                    dv.Eq(v[0].dx3 + v[1].dx2dy
                          + v[0].dxdy2 + v[1].dy3, 0)] # Laplacian of divergence is zero

    # TODO: add higher-order bcs
    bcs = BoundaryConditions(bc_list)
    boundary = Boundary(bcs, bg)

    pdx = p.dx(x0=x+x.spacing/2)
    pdy = p.dy(x0=y+y.spacing/2)
    vxdx = v[0].forward.dx(x0=x)
    vydy = v[1].forward.dy(x0=y)

    derivs = (pdx, pdy, vxdx, vydy)
    subs = boundary.substitutions(derivs)

    c = 2.5  # km/s
    rho = 1000  # kgm-3

    t0 = 0.  # Simulation starts a t=0
    tn = 1500.  # Simulation last 0.8 seconds (800 ms)
    # Note: grid increment hardcoded, courant number 0.5
    dt = 0.5*30/c  # Time step from grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.008  # Source peak frequency is 8Hz (0.008 kHz)
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
    psave = dv.TimeFunction(name='psave', grid=grid, time_order=0,
                            save=nsnaps+1, time_dim=t_sub)
    vxsave = dv.TimeFunction(name='vxsave', grid=grid, time_order=0,
                             save=nsnaps+1, time_dim=t_sub)
    vysave = dv.TimeFunction(name='vysave', grid=grid, time_order=0,
                             save=nsnaps+1, time_dim=t_sub)

    
    # Pressure update
    eq_p = dv.Eq(p.forward, p + dt*rho*c**2*(subs[vxdx] + subs[vydy]))
    # Velocity updates
    eq_vx = dv.Eq(v[0].forward, v[0] + dt*subs[pdx]/rho)
    eq_vy = dv.Eq(v[1].forward, v[1] + dt*subs[pdy]/rho)

    eq_psave = dv.Eq(psave, p)
    eq_vxsave = dv.Eq(vxsave, v[0])
    eq_vysave = dv.Eq(vysave, v[1])

    src_term = src.inject(field=p.forward, expr=c*src*dt**2)

    op = dv.Operator([eq_vx, eq_vy, eq_p, eq_psave, eq_vxsave, eq_vysave] + src_term)
    op(time=time_range.num-1, dt=dt)

    return psave.data, vxsave.data, vysave.data


def plot_snaps(psave_data, vxsave_data, vysave_data, shift, sdf):
    # FIXME: Hardcoded for four snapshots
    # Plot extent
    plt_ext = (0., 9600., 0.-shift*30., 5100.-shift*30.)

    # Plot surface with SDF contours
    xvals = np.linspace(0., 9600., psave_data.shape[1])
    yvals = np.linspace(0.-shift*30., 5100.-shift*30., psave_data.shape[2])
    xmsh, ymsh = np.meshgrid(xvals, yvals, indexing='ij')

    fig, axs = plt.subplots(4, 3, constrained_layout=True, figsize=(9.6*1.5, 5.1*2),
                            sharex=True, sharey=True)

    for i in range(1, 5):
        vmax = np.amax(np.abs(psave_data[i]))
        vmin = -vmax
        axs[i-1, 0].imshow(psave_data[i].T, origin='lower',
                           extent=plt_ext, vmax=vmax, vmin=vmin,
                           cmap='seismic')
        axs[i-1, 0].contour(xmsh, ymsh, sdf.data, [0])
        if i == 4:
            axs[i-1, 0].set_xlabel("Distance (m)")
        axs[i-1, 0].set_ylabel("Elevation (m)")
        axs[i-1, 0].set_yticks([-1000., 0., 1000., 2000., 3000.])

    for i in range(1, 5):
        vmax = np.amax(np.abs(vxsave_data[i]))
        vmin = -vmax
        axs[i-1, 1].imshow(vxsave_data[i].T, origin='lower',
                           extent=plt_ext, vmax=vmax, vmin=vmin,
                           cmap='seismic')
        axs[i-1, 1].contour(xmsh, ymsh, sdf.data, [0])
        if i == 4:
            axs[i-1, 1].set_xlabel("Distance (m)")
        axs[i-1, 1].set_yticks([-1000., 0., 1000., 2000., 3000.])

    for i in range(1, 5):
        vmax = np.amax(np.abs(vysave_data[i]))
        vmin = -vmax
        axs[i-1, 2].imshow(vysave_data[i].T, origin='lower',
                           extent=plt_ext, vmax=vmax, vmin=vmin,
                           cmap='seismic')
        axs[i-1, 2].contour(xmsh, ymsh, sdf.data, [0])
        if i == 4:
            axs[i-1, 2].set_xlabel("Distance (m)")
        axs[i-1, 2].set_yticks([-1000., 0., 1000., 2000., 3000.])
    plt.show()


def load_sdf(files, s_o, shift):
    """Load a signed-distance function from file"""
    sdfs = []
    for stagger, file in files.items():
        sdf_data = np.load(append_path(file))

        # Move the surface upwards by grid increments
        nx, ny = sdf_data.shape
        fill_vals = np.full((nx, shift), np.amax(sdf_data))
        sdf_shift = np.concatenate((fill_vals, sdf_data[:, :-shift]), axis=1)

        if stagger == 'node':
            # Set up the grid
            # Note: size currently hardcoded
            extent = (9600., 5100.)
            grid = dv.Grid(shape=sdf_shift.shape, extent=extent)
        x, y = grid.dimensions

        if stagger == 'node':
            staggered = dv.NODE
        elif stagger == 'x':
            staggered = x
        elif stagger == 'y':
            staggered = y

        sdf = dv.Function(name='sdf', grid=grid, space_order=s_o,
                          staggered=staggered)
        sdf.data[:] = sdf_shift
        sdfs.append(sdf)
    return tuple(sdfs)


def append_path(file):
    """Turn a relative path into an absolute one"""
    path = os.path.dirname(os.path.abspath(__file__))
    return path + file


def main():
    shift = 50  # Number of grid increments to shift surface
    s_o = 4  # Space order
    # Load the signed distance function data
    sdf_file = "/../infrasound/surface_files/mt_st_helens_2d.npy"
    sdf_file_x = "/../infrasound/surface_files/mt_st_helens_2d_x.npy"
    sdf_file_y = "/../infrasound/surface_files/mt_st_helens_2d_y.npy"
    sdf, sdf_x, sdf_y = load_sdf({'node': sdf_file,
                                  'x': sdf_file_x,
                                  'y': sdf_file_y},
                                 s_o, shift)
    outfile_p = append_path("/2D_free_surface_snaps_p.npy")
    outfile_vx = append_path("/2D_free_surface_snaps_vx.npy")
    outfile_vy = append_path("/2D_free_surface_snaps_vy.npy")

    nsnaps = 4  # Number of snaps

    # If not run then run and save output
    # If output found then plot it
    try:
        p_snaps = np.load(outfile_p)
        vx_snaps = np.load(outfile_vx)
        vy_snaps = np.load(outfile_vy)
        plot_snaps(p_snaps, vx_snaps, vy_snaps, shift, sdf)
    except FileNotFoundError:
        p_snaps, vx_snaps, vy_snaps = run(sdf, sdf_x, sdf_y, s_o, nsnaps)
        np.save(outfile_p, p_snaps)
        np.save(outfile_vx, vx_snaps)
        np.save(outfile_vy, vy_snaps)


if __name__ == "__main__":
    main()
