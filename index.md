---
layout: default
---

The Anelastic Wave Propagation software (awp-odc-os) simulates wave propagation in a 3D viscoelastic or elastic solid. Wave propagation use a variant of the staggered grid finite difference scheme to approximately solve the 3D elastodynamic equations for velocity and stress. The GPU version offers the absorbing boundary conditions (ABC) of Cerjan for dealing with artificial wave reflection at external boundaries.

awp-odc-os is implemented in C and CUDA. The Message Passing Interface supports parallel computation (MPI-2) and parallel I/O (MPI-IO).
