"""Utilites for convergence testing the exact solution"""

import pickle
import numpy as np
import devito as dv
import sympy as sp
import matplotlib.pyplot as plt

from schism import BoundaryGeometry, BoundaryConditions, Boundary


def read_sdf(surface, dims):
    """Unpickle an sdf"""
    fname = 'sdfs/' + surface + '_' + str(dims) + 'd.dat'
    with open(fname, 'rb') as f:
        sdf = pickle.load(f)
    return sdf


def x1(x, y, A):
    return x - A*np.sin(x)*np.cosh(y)


def y1(x, y, A):
    return y - A*np.cos(x)*np.sinh(y)


def c(x, y, A, c0):
    term1 = (1-A*np.cos(x)*np.cosh(y))**2
    term2 = (A*np.sin(x)*np.sinh(y))**2
    return c0*(term1 + term2)**-0.5


def alpha(m, c0):
    return c0*np.sqrt(m**2+(np.pi/2)**2)


def u(t, x, y, c0, m, A):
    x1_vals = x1(x, y, A)
    y1_vals = y1(x, y, A)
    alp = alpha(m, c0)
    return np.cos(m*x1_vals-alp*t)*np.cos(np.pi*y1_vals/2)


def update_sdf_with_subdomains(sdf, subdomains):
    """
    Create a new sdf with the same size as the old one, albeit attached to a
    grid with the specified subdomains.

    Parameters
    ----------
    sdf : Function
        The SDF to be modified
    subdomains : list
        Subdomains to be attached to the grid
    """
    grid = sdf.grid
    # Define a new grid with these subdomains
    new_grid = dv.Grid(shape=grid.shape, extent=grid.extent,
                       origin=grid.origin, dimensions=grid.dimensions,
                       subdomains=subdomains)

    # Rebuild the SDF on the new grid so we can use these subdomains

    new_sdf = dv.Function(name='sdf', grid=new_grid,
                          space_order=sdf.space_order)
    new_sdf.data[:] = sdf.data[:]

    return new_grid, new_sdf


def setup_celerity(grid, xmsh, ymsh, A, c0):
    """Set up the celerity (wavespeed) field"""
    # Set up the celerity field
    cel = dv.Function(name='cel', grid=grid)
    # Cap the celerity at 3, as these points are outside the domain anyway
    # Prevents excessive reduction of the timestep
    cel_data = np.minimum(c(xmsh, ymsh, A, c0), 3)
    # Add the edges
    cel_data_full = np.concatenate((cel_data[-5:-1], cel_data, cel_data[1:5]))
    cel.data[:] = cel_data_full[:]

    return cel


def initialise_wavefield(p, xmsh, ymsh, c0, m, A, dt, interior_mask):
    """Initialise the current and backward timesteps of the pressure field"""
    u0 = u(0, xmsh, ymsh, c0, m, A)
    u0[np.logical_not(interior_mask)] = 0
    un1 = u(-dt, xmsh, ymsh, c0, m, A)
    un1[np.logical_not(interior_mask)] = 0

    p_data_n1 = np.concatenate((un1[-5:-1], un1, un1[1:5]))

    p_data_0 = np.concatenate((u0[-5:-1], u0, u0[1:5]))

    p.data[0] = p_data_n1
    p.data[1] = p_data_0


class MainDomain(dv.SubDomain):  # Main section of the grid
    name = 'main'

    def __init__(self):
        super().__init__()

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 4, 4), y: ('middle', 0, 2)}


class Left(dv.SubDomain):  # Left wraparound region for periodic bcs
    name = 'left'

    def __init__(self):
        super().__init__()

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('left', 4), y: y}


class Right(dv.SubDomain):  # Right wraparound region for periodic bcs
    name = 'right'

    def __init__(self):
        super().__init__()

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('right', 4), y: y}


class Base(dv.SubDomain):  # Base with zero flux (centre)
    name = 'base'

    def __init__(self):
        super().__init__()

    def define(self, dimensions):
        x, y = dimensions
        return {x: ('middle', 4, 4), y: ('right', 2)}


def zero_flux_base(eq, subdomain):
    """
    Transform an equation such that the stencils are folded back on
    themselves at the edge of the domain.
    """
    lhs, rhs = eq.evaluate.args

    yzf = subdomain.dimensions[-1]
    y = yzf.parent

    # Functions present in stencil
    funcs = dv.symbolics.retrieve_functions(rhs)
    mapper = {}
    for f in funcs:
        # Get the y index
        yind = f.indices[-1]
        if (yind-y).as_coeff_Mul()[0] > 0:
            pos = abs((f.shape[-1] - 1)*y.spacing - yind)
            ymod = dv.symbolics.INT((f.shape[-1]-1)*y.spacing-pos)
            mapper.update({f: f.subs({yind: ymod})})

    return dv.Eq(lhs, rhs.subs(mapper), subdomain=subdomain)


def calculate_error(refinement, has_1D_basis=False):
    """
    Calculate the error for a given refinement, benchmarking the numerical
    solution against an exact one.
    """

    sdf = read_sdf('exact_solution_surface_periodic_' + str(refinement), 2)

    # Use SDF grid
    grid = sdf.grid

    # Misc parameters
    c0 = 1
    m = 8
    A = 0.25

    # Trim off the edges where we will have bcs
    xvals = np.linspace(0, 2*np.pi, grid.shape[0]-8)
    yvals = np.linspace(-3*np.pi/4, 0, grid.shape[1])

    xmsh, ymsh = np.meshgrid(xvals, yvals, indexing='ij')

    main_domain = MainDomain()
    left = Left()
    right = Right()
    base = Base()
    subdomains = [main_domain, left, right, base]

    new_grid, new_sdf = update_sdf_with_subdomains(sdf, subdomains)

    bg = BoundaryGeometry(new_sdf)

    interior_mask = bg.interior_mask[(sp.core.numbers.Zero(),
                                      sp.core.numbers.Zero())][4:-4]

    # Set up the celerity field
    cel = setup_celerity(new_grid, xmsh, ymsh, A, c0)

    # Fix the CFL number at 0.1
    dt = 0.1*new_grid.spacing[0]/np.amax(cel.data)

    # Set up the TimeFunction
    p = dv.TimeFunction(name='p', grid=new_grid, space_order=4, time_order=2)
    initialise_wavefield(p, xmsh, ymsh, c0, m, A, dt, interior_mask)

    if has_1D_basis:
        bc_list = [dv.Eq(p, 0),
                   dv.Eq(p.dx2, 0),
                   dv.Eq(p.dy2, 0),
                   dv.Eq(p.dx4, 0),
                   dv.Eq(p.dy4, 0)]
        strategy = 'reduce'
    else:
        bc_list = [dv.Eq(p, 0),  # Zero pressure on free surface
                   dv.Eq(p.dx2+p.dy2, 0),  # Zero laplacian
                   dv.Eq(p.dx4 + 2*p.dx2dy2 + p.dy4, 0)]  # Zero biharmonic
        strategy = 'expand'

    bcs = BoundaryConditions(bc_list)
    # Construct the immersed boundary
    boundary = Boundary(bcs, bg, has_1D_basis=has_1D_basis, strategy=strategy)

    # Get the modified stencils
    derivs = (p.dx2, p.dy2)
    subs = boundary.substitutions(derivs)

    # Standard equation with no substitutions
    eq_normal = dv.Eq(p.forward,
                      2*p-p.backward+dt**2*cel**2*(p.dx2+p.dy2),
                      subdomain=main_domain)
    eq_main = dv.Eq(p.forward,
                    2*p-p.backward+dt**2*cel**2*(subs[p.dx2]+subs[p.dy2]),
                    subdomain=main_domain)

    t = new_grid.stepping_dim
    x, y = new_grid.dimensions
    left_bcs = [dv.Eq(p[t+1, i, y],
                      p[t+1, new_grid.shape[0]-9+i, y],
                      subdomain=left) for i in range(4)]
    right_bcs = [dv.Eq(p[t+1, new_grid.shape[0]-4+i, y],
                       p[t+1, i+5, y],
                       subdomain=right) for i in range(4)]

    base_bc = zero_flux_base(eq_normal, base)

    # Set up number of timesteps
    t_max = 2*np.pi/c0
    nsteps = t_max/dt

    op = dv.Operator([eq_main, base_bc] + left_bcs + right_bcs)
    op.apply(dt=dt, t_M=int(nsteps))

    # Remember to trim edges where periodic bcs are applied
    # Index in the buffer where the final timestep should be stored:
    ind = (1 + int(nsteps)) % 3

    uend = u(dt*(int(nsteps)), xmsh, ymsh, c0, m, A)
    uend[np.logical_not(interior_mask)] = 0

    plt.imshow(uend.T-p.data[ind, 4:-4].T,
               extent=(0, 2*np.pi, 0, -3*np.pi/4),
               aspect='auto', cmap='seismic')
    plt.title("Error")
    plt.colorbar()
    plt.show()

    max_err = np.amax(np.abs(uend-p.data[ind, 4:-4]))
    return max_err
