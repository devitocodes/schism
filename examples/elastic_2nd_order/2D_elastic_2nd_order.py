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
import sympy as sp
import os

from schism import BoundaryGeometry, BoundaryConditions, Boundary
from examples.seismic import TimeAxis, RickerSource


def as_tensor(matrix, voigt):
    """Convert Voigt notation to stiffness tensor in 2D"""
    tensor = sp.tensor.array.MutableDenseNDimArray.zeros(2, 2, 2, 2)

    for k1, v1 in voigt.items():
        i, j = k1
        alpha = v1
        for k2, v2 in voigt.items():
            # Linting complains about l as a variable so use m
            k, m = k2
            beta = v2
            tensor[i, j, k, m] = matrix[alpha, beta]
    return tensor


def tensor_to_matrix(tensor, voigt):
    """Convert stiffness tensor to Voigt notation in 2D"""
    matrix = sp.zeros(3, 3)

    for k1, v1 in voigt.items():
        i, j = k1
        alpha = v1
        for k2, v2 in voigt.items():
            k, m = k2
            beta = v2
            matrix[alpha, beta] = tensor[i, j, k, m]
    return matrix


def rotate(matrix, R, voigt):
    """Rotate a stiffness tensor in Voigt notation using matrix R"""
    C = as_tensor(matrix, voigt)
    rotated = sp.tensor.array.MutableDenseNDimArray.zeros(2, 2, 2, 2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for m in range(2):
                    for ii in range(2):
                        for jj in range(2):
                            for kk in range(2):
                                for mm in range(2):
                                    gg = R[ii, i]*R[jj, j]*R[kk, k]*R[mm, m]
                                    rotated[i, j, k, m] += gg*C[ii, jj, kk, mm]
    return tensor_to_matrix(rotated, voigt)


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


def get_vti_bcs(nx, ny, ux, uy, v_p2, v_s2, v_pn2, v_px2, s_o):
    """Returns boundary conditions for the VTI case"""
    # More shorthands
    v_p4 = v_p2**2
    v_s4 = v_s2**2

    # Note that a factor of rho has been removed here
    txx = v_px2*ux.dx \
        + np.sqrt((v_p2-v_s2)*(v_pn2-v_s2))*uy.dy - v_s2*uy.dy
    tyy = np.sqrt((v_p2-v_s2)*(v_pn2-v_s2))*uy.dy - v_s2*ux.dx \
        + v_p2*uy.dy
    txy = v_s2*ux.dy + v_s2*uy.dx

    # With fourth-order boundary conditions
    bc_list = [dv.Eq(nx*txx + ny*txy, 0),
               dv.Eq(nx*txy + ny*tyy, 0)]

    if s_o >= 4:
        bc4 = [dv.Eq(nx*ux.dx3*v_pn2*v_px2
                     + ny*ux.dy3*v_s4
                     + ny*uy.dx3*v_pn2*v_s2
                     + ny*ux.dx2dy*(v_p2*v_pn2 - v_p2*v_s2 + v_s4
                                    - v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4))
                     + nx*ux.dxdy2*(v_px2*v_s2
                                    + v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4))
                     + nx*uy.dx2dy*(-v_pn2*v_s2
                                    + v_pn2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                    - v_pn2*v_s2 + v_s4)
                                    + v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4))
                     + ny*uy.dxdy2*(v_p2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                 - v_pn2*v_s2 + v_s4)
                                    + v_s4)
                     + nx*uy.dy3*(-v_s4
                                  + v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                 - v_pn2*v_s2 + v_s4)), 0),
               dv.Eq(nx*ux.dy3*v_p2*v_s2
                     + nx*uy.dx3*v_s4
                     + ny*uy.dy3*v_p4
                     + nx*ux.dx2dy*(v_px2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                  - v_pn2*v_s2 + v_s4)
                                    + v_s4)
                     + ny*ux.dx3*(-v_s4
                                  + v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                 - v_pn2*v_s2 + v_s4))
                     + ny*ux.dxdy2*(-v_p2*v_s2
                                    + v_p2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4)
                                    + v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4))
                     + ny*uy.dx2dy*(v_p2*v_s2
                                    + v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4))
                     + nx*uy.dxdy2*(v_p2*v_pn2 - v_pn2*v_s2 + v_s4
                                    - v_s2*np.sqrt(v_p2*v_pn2 - v_p2*v_s2
                                                   - v_pn2*v_s2 + v_s4)), 0)]

        bc_list += bc4

    return bc_list


def get_tti_bcs(nx, ny, ux, uy,
                d11, d12, d13, d22, d23, d33,
                s_o):
    """Returns boundary conditions for the TTI case"""
    # Note that a factor of rho has been removed here
    txx = d11*ux.dx + d12*uy.dy + d13*ux.dy + d13*uy.dx
    tyy = d12*ux.dx + d22*uy.dy + d23*ux.dy + d23*uy.dx
    txy = d13*ux.dx + d23*uy.dy + d33*ux.dy + d33*uy.dx

    bc_list = [dv.Eq(nx*txx + ny*txy, 0),
               dv.Eq(nx*txy + ny*tyy, 0)]

    if s_o >= 4:
        bc4 = [dv.Eq(ux.dx2dy*(3*d11*d13*nx + d11*d33*ny + d12**2*ny
                               + d12*d13*nx + d12*d33*ny + 2*d13**2*ny
                               + d13*d23*ny + 2*d13*d33*nx)
                     + ux.dx3*(d11**2*nx + d11*d13*ny + d12*d13*ny
                               + d13**2*nx)
                     + ux.dxdy2*(d11*d33*nx + 2*d12*d23*ny + d12*d33*nx
                                 + 2*d13**2*nx + d13*d23*nx + 3*d13*d33*ny
                                 + d23*d33*ny + d33**2*nx)
                     + ux.dy3*(d13*d33*nx + d23**2*ny + d23*d33*nx
                               + d33**2*ny)
                     + uy.dx2dy*(d11*d12*nx + d11*d23*ny + d12*d23*ny
                                 + d12*d33*nx + 2*d13**2*nx + d13*d22*ny
                                 + d13*d23*nx + 2*d13*d33*ny + d23*d33*ny
                                 + d33**2*nx)
                     + uy.dx3*(d11*d13*nx + d11*d33*ny + d13*d23*ny
                               + d13*d33*nx)
                     + uy.dxdy2*(2*d12*d13*nx + d12*d22*ny + d12*d23*nx
                                 + 2*d13*d23*ny + d13*d33*nx + d22*d33*ny
                                 + d23**2*ny + 2*d23*d33*nx + d33**2*ny)
                     + uy.dy3*(d12*d33*nx + d22*d23*ny + d23**2*nx
                               + d23*d33*ny), 0),
               dv.Eq(ux.dx2dy*(d11*d12*nx + d11*d33*nx + d12*d13*ny
                               + 2*d12*d23*ny
                               + d13**2*nx + 2*d13*d23*nx + 2*d13*d33*ny
                               + d23*d33*ny + d33**2*nx)
                     + ux.dx3*(d11*d13*nx + d12*d33*ny + d13**2*ny
                               + d13*d33*nx)
                     + ux.dxdy2*(d11*d23*nx + d12*d13*nx + d12*d22*ny
                                 + d12*d33*ny
                                 + d13*d22*nx + d13*d23*ny + d13*d33*nx
                                 + 2*d23**2*ny + 2*d23*d33*nx + d33**2*ny)
                     + ux.dy3*(d13*d23*nx + d22*d23*ny + d22*d33*nx
                               + d23*d33*ny)
                     + uy.dx2dy*(2*d12*d13*nx + d12*d33*ny + d13*d23*ny
                                 + d13*d33*nx + d22*d33*ny + 2*d23**2*ny
                                 + 3*d23*d33*nx + d33**2*ny)
                     + uy.dx3*(d13**2*nx + d13*d33*ny + d23*d33*ny
                               + d33**2*nx)
                     + uy.dxdy2*(d12**2*nx + d12*d23*ny + d12*d33*nx
                                 + d13*d23*nx
                                 + 3*d22*d23*ny + d22*d33*nx + 2*d23**2*nx
                                 + 2*d23*d33*ny)
                     + uy.dy3*(d12*d23*nx + d22**2*ny + d22*d23*nx
                               + d23**2*ny), 0)]
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
        v_px2 = (1+2*ep)*v_p**2
        bc_list = get_vti_bcs(nx, ny, ux, uy, v_p2, v_s2,
                              v_pn2, v_px2, s_o)
    elif mode == 'tti':
        # Anisotropy parameters
        de = 0.1
        ep = 0.25
        # Tilt
        th = np.radians(45)

        # Seismic wavespeeds
        v_p02 = v_p**2
        v_s02 = v_s**2
        v_pn2 = (1+2*de)*v_p**2
        v_px2 = (1+2*ep)*v_p**2

        # VTI stiffness tensor entries (uses conventions from 3D case)
        c11 = rho*v_px2
        c13 = rho*np.sqrt((v_p02-v_s02)*(v_pn2-v_s02))-rho*v_s02
        c33 = rho*v_p02
        c44 = rho*v_s02

        # Assemble the VTI stiffness tensor (Voigt notation)
        C = sp.Matrix([[c11, c13, 0],
                       [c13, c33, 0],
                       [0, 0, c44]])

        # Rotation matrix
        R = sp.Matrix([[np.cos(th), -np.sin(th)],
                       [np.sin(th), np.cos(th)]])

        # Apparatus to perform rotation
        voigt = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 2}

        # Get the rotated stiffness tensor (D)
        D = rotate(C, R, voigt)

        # Shorthands for components of D
        d11 = D[0, 0]
        d12 = D[0, 1]
        d13 = D[0, 2]
        d22 = D[1, 1]
        d23 = D[1, 2]
        d33 = D[2, 2]

        bc_list = get_tti_bcs(nx, ny, ux, uy,
                              d11, d12, d13, d22, d23, d33,
                              s_o)

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
        rhs_ux = v_px2*ux.dx2 + v_s2*ux.dy2 \
            + np.sqrt((v_p2-v_s2)*(v_pn2-v_s2))*uy.dxdy
        rhs_uy = v_s2*uy.dx2 + v_p2*uy.dy2 \
            + np.sqrt((v_p2-v_s2)*(v_pn2-v_s2))*ux.dxdy
    elif mode == 'tti':
        rhs_ux = b*(d11*ux.dx2 + 2*d13*ux.dxdy + d33*ux.dy2
                    + d13*uy.dx2 + (d12+d33)*uy.dxdy + d23*uy.dy2)
        rhs_uy = b*(d13*ux.dx2 + (d12+d33)*ux.dxdy + d23*ux.dy2
                    + d33*uy.dx2 + 2*d23*uy.dxdy + d22*uy.dy2)

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
    for i in range(1, 5):
        print(np.linalg.norm(uxsave_data[i])
              + np.linalg.norm(uysave_data[i]))
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

    # Mode, can be 'iso', 'vti', or 'tti'
    mode = 'tti'

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
