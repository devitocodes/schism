"""
Simple 3D example implementing rigid-surface with the 2nd-order acoustic wave
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
import pyvista as pv
import os

from schism import BoundaryGeometry, BoundaryConditions, Boundary
from examples.seismic import TimeAxis, RickerSource


def run(sdf, s_o, nsnaps):
    """Run a forward model if no file found to read"""
    grid = sdf.grid
    bg = BoundaryGeometry(sdf)

    # Create pressure function
    p = dv.TimeFunction(name='p', grid=grid, space_order=s_o, time_order=2)

    bc_list = [dv.Eq(p.dx, 0),
               dv.Eq(p.dy, 0),
               dv.Eq(p.dz, 0)]

    if s_o >= 4:
        bc_list += [dv.Eq(p.dx3 + p.dxdy2 + p.dxdz2, 0),
                    dv.Eq(p.dx2dy + p.dy3 + p.dydz2, 0),
                    dv.Eq(p.dx2dz + p.dy2dz + p.dz3, 0)]

    # TODO: add higher-order bcs
    bcs = BoundaryConditions(bc_list)
    boundary = Boundary(bcs, bg)

    derivs = (p.dx2, p.dy2, p.dz2)
    subs = boundary.substitutions(derivs)

    c = 0.35  # km/s

    t0 = 0.  # Simulation starts a t=0
    tn = 10000.  # Simulation last 10 seconds (10000 ms)
    # Note: grid increment hardcoded, courant number 0.5
    dt = 0.5*30/c  # Time step from grid spacing

    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    f0 = 0.001  # Source peak frequency is 1Hz (0.001 kHz)
    src = RickerSource(name='src', grid=grid, f0=f0,
                       npoint=1, time_range=time_range)

    src.coordinates.data[0, 0] = 4800.
    src.coordinates.data[0, 1] = 4800.
    src.coordinates.data[0, 2] = 2000.  # From model base

    # Set up snapshotting
    steps = src.nt
    factor = int(steps/nsnaps)
    t_sub = dv.ConditionalDimension('t_sub', parent=grid.time_dim,
                                    factor=factor)

    # Buffer size needs to be more robust
    psave = dv.TimeFunction(name='psave', grid=grid, time_order=0,
                            save=nsnaps+1, time_dim=t_sub)

    eq = dv.Eq(p.forward,
               2*p - p.backward
               + dt**2*c**2*(subs[p.dx2] + subs[p.dy2] + subs[p.dz2]))

    eq_save = dv.Eq(psave, p)

    src_term = src.inject(field=p.forward, expr=c*src*dt**2)

    op = dv.Operator([eq, eq_save] + src_term)
    op(time=time_range.num-1, dt=dt)

    return psave.data


def plot_snaps(psave_data, shift, sdf):
    """Plot the snapshot"""
    y_slice = sdf.shape[1]//2
    # Reduce the snapshots down to fewer dimensions
    psave_data = psave_data[:, :, y_slice]
    # FIXME: Hardcoded for four snapshots
    # Plot extent
    plt_ext = (0., 9600., 0.+shift*30., 5100.+shift*30.)

    # Plot surface with SDF contours
    xvals = np.linspace(0., 9600., psave_data.shape[1])
    yvals = np.linspace(0.+shift*30., 5100.+shift*30., psave_data.shape[2])
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
        axs[j, (i+1) % 2].contour(xmsh, ymsh, sdf.data[:, y_slice], [0])
        if j == 1:
            axs[j, (i+1) % 2].set_xlabel("Distance (m)")
        if (i+1) % 2 == 0:
            axs[j, (i+1) % 2].set_ylabel("Elevation (m)")
        axs[j, (i+1) % 2].set_yticks([0., 1000., 2000., 3000., 4000.])
    plt.show()


def render_snaps(psave_data, shift):
    """Make a render of the second-to-last snapshot"""
    data = psave_data[-2]
    opacity = (np.abs(data)/np.amax(np.abs(data)))**0.4

    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(data.shape) + 1
    # Needed to line up the surface mesh correctly for plotting
    # This mesh uses the centre of the crater and sea level as reference
    mesh.origin = (-4800., -4800., shift*30)
    mesh.spacing = (30., 30., 30.)

    # Add the data values to the cell data
    mesh.cell_data["opacity"] = opacity.flatten(order="F")
    mesh.cell_data["values"] = data.flatten(order="F")

    slicex = mesh.slice(normal=[1, 0, 0])
    slicey = mesh.slice(normal=[0, 1, 0])
    slicexy = mesh.slice(normal=[1, 1, 0])
    sliceyx = mesh.slice(normal=[-1, 1, 0])

    surface_file = append_path("/../infrasound/surface_files/mt_st_helens.ply")
    surface = pv.read(surface_file)

    plotter = pv.Plotter()
    vmax = np.amax(np.abs(data))
    vmin = -vmax
    plotter.add_mesh(slicex, opacity='opacity', cmap='seismic',
                     clim=[vmin, vmax])
    plotter.add_mesh(slicey, opacity='opacity', cmap='seismic',
                     clim=[vmin, vmax])
    plotter.add_mesh(slicexy, opacity='opacity', cmap='seismic',
                     clim=[vmin, vmax])
    plotter.add_mesh(sliceyx, opacity='opacity', cmap='seismic',
                     clim=[vmin, vmax])
    plotter.add_mesh(surface, opacity=1., specular=0.2, specular_power=0.2)
    plotter.remove_scalar_bar()
    camera_pos = list(plotter.camera.position)
    camera_pos[1] = -camera_pos[1]
    camera_pos[2] = 0.5*camera_pos[2]
    plotter.camera.position = tuple(camera_pos)
    plotter.camera.zoom(1.1)

    plotter.show(screenshot=append_path("/3D_rigid_surface_render"))


def load_sdf(file, s_o, shift):
    """Load a signed-distance function from file"""
    sdf_data = -np.load(append_path(file))

    # Move the surface upwards by grid increments
    nx, ny, nz = sdf_data.shape
    fill_vals = np.full((nx, ny, shift), np.amax(sdf_data))
    sdf_shift = np.concatenate((sdf_data[..., shift:], fill_vals), axis=2)

    # Set up the grid
    # Note: size currently hardcoded
    extent = (9600., 9600., 5100.)
    grid = dv.Grid(shape=sdf_shift.shape, extent=extent)
    sdf = dv.Function(name='sdf', grid=grid, space_order=s_o)
    sdf.data[:] = sdf_shift
    return sdf


def append_path(file):
    """Turn a relative path into an absolute one"""
    path = os.path.dirname(os.path.abspath(__file__))
    return path + file


def main():
    shift = 20  # Number of grid increments to shift surface
    s_o = 4  # Space order
    # Load the signed distance function data
    sdf_file = "/../infrasound/surface_files/mt_st_helens_3d.npy"
    sdf = load_sdf(sdf_file, s_o, shift)

    outfile = append_path("/3D_rigid_surface_snaps.npy")

    nsnaps = 4  # Number of snaps

    # If not run then run and save output
    # If output found then plot it
    try:
        psave_data = np.load(outfile)
        plot_snaps(psave_data, shift, sdf)
        render_snaps(psave_data, shift)
    except FileNotFoundError:
        psave_data = run(sdf, s_o, nsnaps)
        np.save(outfile, psave_data)


if __name__ == "__main__":
    main()
