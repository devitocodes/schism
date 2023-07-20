# Dockerfile for running Schism

To aid the reproducibility of work carried out with Schism, alongside its wider usage, this directory contains a Dockerfile which can be used to run the code. With this, both tests and examples can be run.

# Building and running the image

To build the image, one must navigate to the main directory (that above this one) and run the following command:

`docker build --network=host --file docker/dockerfile --tag schism .`.

To run a bash shell inside the image, use:

`docker run -i -t schism /bin/bash`.

Note that these commands may require `sudo`. Once inside the image, one will want to activate the `venv` created, using:

`source venv/bin/activate`.

From here, one can run the tests and examples, found in `app/schism`.