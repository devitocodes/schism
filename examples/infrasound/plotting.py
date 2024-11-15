"""Useful pyvista plotters for visualising the 3D data"""

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt


def plot_st_helens(data, src_loc, rec_loc, origin, spacing):
    """Plot a 3D render of slices through a dataset with topography"""
    mesh = pv.UniformGrid()
    mesh.dimensions = np.array(data.shape) + 1
    mesh.origin = origin
    mesh.spacing = spacing

    # Add the data values to the cell data
    mesh.cell_data["values"] = data.flatten(order="F")

    slicex = mesh.slice(normal=[1, 0, 0])
    slicey = mesh.slice(normal=[0, 1, 0])

    surface = pv.read("surface_files/mt_st_helens.ply")

    plotter = pv.Plotter()
    plotter.add_mesh(slicex, opacity=1.)
    plotter.add_mesh(slicey, opacity=1.)
    plotter.add_mesh(surface)

    for src in src_loc:
        center = src + origin
        sphere = pv.Sphere(radius=90, center=center)
        plotter.add_mesh(sphere, color='red')

    for rec in rec_loc:
        center = rec + origin
        sphere = pv.Sphere(radius=90, center=center)
        plotter.add_mesh(sphere, color='blue')

    plotter.show()


def plot_top_down(data, xmin, xmax, ymin, ymax):
    """Plot a top-down view of the dataset"""
    extent = (xmin, xmax, ymin, ymax)
    dmax = np.amax(data, axis=-1)

    plt.imshow(dmax.T, extent=extent, origin='lower')
    plt.xlabel("E-W (km)")
    plt.ylabel("N-S (km)")
    plt.show()
