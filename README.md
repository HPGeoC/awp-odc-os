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
* [Running awp-odc-os (nonlinear version)](Running awp-odc-os (nonlinear version))
* [Important Notes](Important Notes)
* [Source File Processing](Source File Processing)
* [Mesh File Processing](Mesh File Processing)
* [I/O Behavior](I/O Behavior)
* [Output](Output)
* [Lustre File System Specifics](Lustre File System Specifics)

#### Distribution Contents
* [src/](src) source code and platform dependent makefiles

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

  ([MACHINE] represents machine name, e.g. `titan`, `bluewaters`) ([COMPILER] represents compiler name, e.g. `gnu`, `pgi`, `cray`)

4. Executable pmcl3d located in `src/`

#### Running awp-odc-os (nonlinear version)

1. Sample file access: [http://hpgeoc.sdsc.edu/downloads/awp-odc-os-nonlinear-example.tar.gz](http://hpgeoc.sdsc.edu/downloads/awp-odc-os-nonlinear-example.tar.gz)

2. Copy the zip file into awp-odc-os home directory and unpack. One folders `run/` will be extracted into the home directory.

  > cd awp-odc-os-nonlinear
  >
  > tar -zxvf ./awp-odc-os-nonlinear-examples.tar.gz

3. Run the environment setting script. This script will prepare required folders and link executable and input files into `run/` folder.

  > cd run

  (depends on the system, e.g. on Blue Waters)

  this script creates following folders:

  - `input/`      - single small source and mesh input files (for small scale tests)
  - `input_rst/`  - pre-partitioned source and mesh files (for large scale tests, see Source file processing section and Mesh file processing section)
  - `output_ckp/` - run statistics and checkpoints if enabled
  - `output_sfc/` - output folder striping might be needed for lustre system
  - `debug`       - output folder for debug information

  > ./env.sh

4. Submit pbs job from `run/` directory. Like the run script, the job submission process is platform dependent. On Blue Waters, for instance, the run.bluewaters.pbs script can be found in `run/` and submitted via (modify your pbs script - account, email address):

  > qsub run.bluewaters.pbs

#### Important Notes

1. Parameter settings reference info in `src/` command.c

2. Key model parameters of the executable (pmcl3d):

  <table>
    <tr><th> parameter(s) </th><th> result                                                                    </th></tr>
    <tr><td> -X -Y -Z     </td><td> grid points in each direction (or NX, NY, NZ)                             </td></tr>
    <tr><td> -x -y        </td><td> GPUs used in x/y direction each, total x*y GPUs used or NPX, NPY, NPZ(=1) </td></tr>
    <tr><td> --TMAX       </td><td> time step in seconds (total time steps are TMAX/DT)                       </td></tr>
    <tr><td> --DT         </td><td> total propagation time to run in seconds                                  </td></tr>
    <tr><td> --DH         </td><td> discretization in space (spatial step for x, y, z (meters))               </td></tr>
    <tr><td> --NVAR       </td><td> number of variables in a grid point                                       </td></tr>
    <tr><td> --NVE        </td><td> visco or elastic scheme (0=elastic, 1=visco, 3=plasticity)                </td></tr>
    <tr><td> --NSRC       </td><td> number of source nodes on fault                                           </td></tr>
    <tr><td> --NST        </td><td> number of time steps in rupture functions                                 </td></tr>
    <tr><td> --IFAULT     </td><td> mode selection and fault or initial stress setting (0-2)                  </td></tr>
    <tr><td> --MEDIASTART </td><td> initial media restart option(0=homogeneous)                               </td></tr>
  </table>

3. Key I/O parameters of the executable (pmcl3d):

  <table>
    <tr><th> parameter(s)    </th><th> result                                                          </th></tr>
    <tr><td> --READ_STEP     </td><td> CPU reads # step sources from file system                       </td></tr>
    <tr><td> --READ_STEP_GPU </td><td> CPU reads larger chunks and sends to GPU at every READ_STEP_GPU
                                  <br> (when IFAULT=2, READ_STEP must be divisible by READ_STEP_GPU)   </td></tr>
    <tr><td> --WRITE_STEP    </td><td> # timesteps to write the buffer to the files                    </td></tr>
    <tr><td> --NTISKP        </td><td> # timesteps to skip to copy velocities from GPU to CPU          </td></tr>
    <tr><td> --NSKPX         </td><td> # points to skip in recording points in X                       </td></tr>
    <tr><td> --NSKPY         </td><td> # points to skip in recording points in Y                       </td></tr>
  </table>

4. Other model parameters:

  <table>
    <tr><th> parameter(s) </th><th> result                                                            </th></tr>
    <tr><td> --ND         </td><td> ABC thickness (grid-points), Cerjan >= 20                         </td></tr>
    <tr><td> --NPC        </td><td> Cerjan(0), or PML(1), current version only implemented for Cerjan
                               <br>  (NPC=0) (npx*npy*npz)/IOST < the total number of OSTs in a file
                               <br> system (IOST definition see section Lustre file system specifics) </td></tr>
    <tr><td> --SoCalQ     </td><td> parameter set for California Vp-Vs Q relationship (SoCalQ=1),
                               <br> default SoCalQ=0                                                  </td></tr>
  </table>

#### Source File Processing

#### Mesh File Processing

#### I/O Behavior

#### Output

#### Lustre File System Specifics
