# awp-odc-os
The Anelastic Wave Propagation software (awp-odc-os) simulates wave propagation
in a 3D viscoelastic or elastic solid. Wave propagation use a variant of the staggered
grid finite difference scheme to approximately solve the 3D elastodynamic
equations for velocity and stress. The GPU version offers the absorbing boundary conditions
(ABC) of Cerjan for dealing with artificial wave reflection at external boundaries.

awp-odc-os is implemented in C and CUDA.  The Message Passing Interface
supports parallel computation (MPI-2) and parallel I/O (MPI-IO).

## Distribution Contents
* [src/](src) source code and platform dependent makefiles
* [User documentation](http://hpgeoc.github.io/awp-odc-os/doc/)
* [Development wiki](https://github.com/HPGeoC/awp-odc-os/wiki)

## System Requirements
* C compiler
* CUDA compiler
* MPI library

## License
awp-odc-os is licensed under [BSD-2](LICENSE)
