## Examples and Tutorials

This folder contains tutorial notebooks explaining the use of the Schism API and the capabilities of the underlying mathematical and software framework. They are broadly ordered according to increasing complexity. There is also a set of example scripts demonstrating the implementation of various propagators featuring real-world topography.

## Tutorial Notebooks
The tutorial notebooks cover implementation of immersed boundaries with the following wave equations:
* 2nd-order isotropic acoustic (free and rigid surfaces)
* 1st-order isotropic acoustic
* Fowler et al. 2010 VTI
* Fletcher et al 2009 TTI
* 2nd-order isotropic elastic
* 1st-order isotropic elastic
Note that Schism is also capable of imposing boundary conditions with other PDEs, for example the diffusion equation.

There is also a comparison of the use of 1D and N-dimensional extrapolations for imposing the immersed boundary, given in notebook 7.

Notebooks 7, 9, and 15 are used to generate figures used in the acoustic wave equations paper.

## Examples Scripts
The example scripts are separated into 6 folders:
* `2nd_order` (contains 2nd-order acoustic examples)
* `1st_order` (contains 1st-order acoustic examples)
* `vti` (contains Duveneck et al. 2008 and Fowler et al. 2010 examples)
* `tti` (contains Fletcher et al. 2009 example)
* `elastic_2nd_order` (contains 2nd-order iso-elastic example)
* `infrasound` (contains infrasound source location examples)

The examples in `2nd_order` and `1st_order` are featured in the acoustic wave equations paper. This includes 2D and 3D runs with free and rigid surfaces featuring the Mt St Helens topographic profile.

Note that the first run of each of these scripts saves snapshots whilst subsequent runs plot them.