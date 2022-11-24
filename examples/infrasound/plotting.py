"""Useful pyvista plotters for visualising the 3D data"""

import pyvista as pv
import numpy as np


def plot_st_helens(data, src_loc, rec_loc, origin, spacing):
    """Plot a 3D render of slices through a dataset with topography"""
    # Create the spatial reference
    mesh = pv.UniformGrid()

    # Set the grid dimensions: shape + 1 because we want to inject our values on
    #   the CELL data
    mesh.dimensions = np.array(data.shape) + 1

    # Edit the spatial reference
    mesh.origin = origin  # The bottom left corner of the data set
    mesh.spacing = spacing  # These are the cell sizes along each axis

    # Add the data values to the cell data
    mesh.cell_data["values"] = data.flatten(order="F")  # Flatten the array!

    # Now plot the grid!
    # slices = mesh.slice_orthogonal(z=1800)
    slicex = mesh.slice(normal=[1, 0, 0])
    slicey = mesh.slice(normal=[0, 1, 0])

    surface = pv.read("surface_files/mt_st_helens.ply")

    plotter = pv.Plotter()
    plotter.add_mesh(slicex, opacity=0.95)
    plotter.add_mesh(slicey, opacity=0.95)
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


def main():
    data = np.random.rand(321, 321, 171)

    origin = np.array([-4800., -4800., 0.])
    src_loc = np.array([4800., 4800., 2250.])[np.newaxis, :]
    rec_loc = np.array([[4800., 1400., 1400.],
                       [8200., 4800., 1500.],
                       [1400., 4800., 1500.],
                       [4800., 8200., 1650.],
                       [2400., 2400., 1400.],
                       [7200., 2400., 1500.],
                       [2400., 7200., 1550.],
                       [7200., 7200., 1500.]])

    spacing = (30, 30, 30)
    plot_st_helens(data, src_loc, rec_loc, origin, spacing)


if __name__ == "__main__":
    main()
