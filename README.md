# **awp-odc-os (nonlinear version)**

The Anelastic Wave Propagation software (awp-odc-os nonlinear version) simulates wave propagation in a 3D viscoelastic or elastic solid. Wave propagation use a variant of the staggered grid finite difference scheme to approximately solve the 3D elastodynamic equations for velocity and stress. The GPU version offers the absorbing boundary conditions (ABC) of Cerjan for dealing with artificial wave reflection at external boundaries.

awp-odc-os is implemented in C and CUDA.  The Message Passing Interface supports parallel computation (MPI-2) and parallel I/O (MPI-IO).

## User Guide

#### Table of Contents
* [Table of Contents](Table of Contents)
* [Distribution Contents](Distribution Contents)
* [System Requirements](System Requirements)
* [License](License)
* [Installation](Installation)
* [Running awp-odc-os (nonlinear version)](Running awp-odc-os (nonlinear version)
* [Important Notes](Important Notes)
* [Source File Processing](Source File Processing)
* [Mesh File Processing](Mesh File Processing)
* [I/O Behavior](I/O Behavior)
* [Output](Output)
* [Lustre File System Specifics](Lustre File System Specifics)

#### Distribution Contents


#### System Requirements
* C compiler
* CUDA compiler
* MPI library

#### License
awp-odc-os (nonlinear version) is licensed under [BSD-2](LICENSE)

#### Installation
To install awp-odc-os, perform the following steps:

1. Code access: [https://github.com/HPGeoC/awp-odc-os](https://github.com/HPGeoC/awp-odc-os). The "nonlinear" branch contains the latest published and tested version of awp-odc-os (nonlinear version).

2. Prepare a directory for the setup and unpack awp-odc-os-nonlinear.zip

  > unzip awp-odc-os-nonlinear.zip

  (`debug/`, `output_ckp/` and `output_sfc/`, see Running awp-odc-os section)

3. Compile code

  > cd awp-odc-os
  >
  > cd src

  (depends on the system, example below is based on Cray XK7 on Blue Waters at NCSA)

  > module swap PrgEnv-cray PrgEnv-gnu
  >
  > module load cudatoolkit
  >
  > module unload darshan
  >
  > make clean -f Makefile.[MACHINE].[COMPILER]
  >
  > make -f makefile.[MACHINE].[COMPILER]

  ([MACHINE] represents machine name, e.g. titan, bluewaters) ([COMPILER] represents compiler name, e.g. gnu, pgi, cray)

4. Executable pmcl3d located in `src/`

#### Running awp-odc-os (nonlinear version)

1. Sample file access: [http://hpgeoc.sdsc.edu/downloads/awp-odc-os-nonlinear-example.tar.gz](http://hpgeoc.sdsc.edu/downloads/awp-odc-os-nonlinear-example.tar.gz)

2. Copy the zip file into awp-odc-os home directory and unpack. One folders *run/* will be extracted into the home directory.

  > cd awp-odc-os-nonlinear
  >
  > tar -zxvf ./awp-odc-os-v1.0-examples.tar.gz

3. Run the environment setting script. This script will prepare required folders and link executable and input files into `run/` folder.

  > cd run

4. Submit pbs job from `run/` directory. Like the run script, the job submission process is platform dependent. On Blue Waters, for instance, the run.bluewaters.pbs script can be found in `run/` and submitted via (modify your pbs script - account, email address):

  > qsub run.bluewaters.pbs

#### Important Notes

1. Parameter settings reference info in `src/` command.c

2. Key model parameters of the executable (pmcl3d):

  <table>
    <tr><th>parameter(s)</th><th>result</th></tr>
    <tr><td>`-X` `-Y` `-Z`</td><td>_grid points in each direction (or NX, NY, NZ)_                            </td></tr>
    <tr><td>`-x` `-y`     </td><td>_GPUs used in x/y direction each, total x*y GPUs used or NPX, NPY, NPZ(=1)_</td></tr>
    <tr><td>`--TMAX`      </td><td>_time step in seconds (total time steps are TMAX/DT)_                      </td></tr>
    <tr><td>`--DT`        </td><td>_total propagation time to run in seconds_                                 </td></tr>
    <tr><td>`--DH`        </td><td>_discretization in space (spatial step for x, y, z (meters))_              </td></tr>
    <tr><td>`--NVAR`      </td><td>_number of variables in a grid point_                                      </td></tr>
    <tr><td>`--NVE`       </td><td>_visco or elastic scheme (1=visco, 0=elastic)_                             </td></tr>
    <tr><td>`--NSRC`      </td><td>_number of source nodes on fault_                                          </td></tr>
    <tr><td>`--NST`       </td><td>_number of time steps in rupture functions_                                </td></tr>
    <tr><td>`--IFAULT`    </td><td>_mode selection and fault or initial stress setting (0-2)_                 </td></tr>
    <tr><td>`--MEDIASTART`</td><td>_initial media restart option(0=homogeneous)_                              </td></tr>
  </table>

3. Key I/O parameters of the executable (pmcl3d):

4. Other model parameters:



#### Source File Processing

#### Mesh File Processing

#### I/O Behavior

#### Output

#### Lustre File System Specifics
