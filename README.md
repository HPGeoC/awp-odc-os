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

5. Modify `BLOCK_Z_SIZE` in [src/pmcl3d_cons.h]() manually parameter `BLOCK_Z_SIZE` must be powers of 2 (32, 64 …) and can be divided by `NZ BLCOK_SIZE_Z` is preferred to be as big as possible for better performance. Default `BLOCK_Z_SIZE` in this version of awp-odc-os (nonlinear version) is 256.

6. Check `--READ_STEP` and `--READ_STEP_GPU` in input parameters. Current code require `--READ_STEP` must equals to `--READ_STEP_GPU`

7. Example Inputs [http://hpgeoc.sdsc.edu/downloads/awp-odc-os-nonlinear-example.tar.gz](http://hpgeoc.sdsc.edu/downloads/awp-odc-os-nonlinear-example.tar.gz)

  - Source: moment source inputs (6 variables each time step) 101 steps source input file (binary)
      <table>
        <tr><th> 1st source             </th><th> 1st timestep           </th><th> 2nd timestep           </th><th>...</th></tr>
        <tr><td> location (int)         </td><td> value (float)          </td><td> value (float)          </td><td>...</td></tr>
        <tr><td> x, y, z                </td><td> xx, yy, zz, xy, xz, yz </td><td> xx, yy, zz, xy, xz, yz </td><td>...</td></tr>
        <tr><th> 101th timestep         </th><th> 2nd source             </th><th> 1st timestep           </th><th>...</th></tr>
        <tr><td> value (float)          </td><td> location (int)         </td><td> value (float)          </td><td>...</td></tr>
        <tr><td> xx, yy, zz, xy, xz, yz </td><td> x, y, z                </td><td> xx, yy, zz, xy, xz, yz </td><td>...</td></tr>
      </table>

  - Mesh: mesh file for 560x560x512 size (binary)
      <table>
        <tr><th> (x, y, z)   </th><th> (x, y, z)   </th><th>...</th><th> (x, y, z)   </th><th> (x, y, z)   </th><th>...</th></tr>
        <tr><td> (1, 1, 1)   </td><td> (2, 1, 1)   </td><td>...</td><td> (1, 2, 1)   </td><td> (2, 2, 1)   </td><td>...</td></tr>
        <tr><td> vp, vs, den </td><td> vp, vs, den </td><td>...</td><td> vp, vs, den </td><td> vp, vs, den </td><td>...</td></tr>
      </table>

8. Lustre Striping

  For large scale run using more than tens of GPUs on lustre file system, striping is needed. Visit system user guide for more info about striping general information for 320x320x2048, 2x2, NTISKP=20, write_step=100, nskpx=2, nskpy=2 each core holds data size (160/2)x(160/2)x1x4_bytesx100_write_steps=2.44MB

  > lfs setstripe -s 3m -c -1 -i -1 output_sfc

  mesh read uses mpi-io, each core data size 160*160*2048*3_variables*4=600MB first setup striping for a new file named mesh

  > lfs setstrspe -s 600m -c 4 -i -1 mesh

  stripe_count best equal nr of GPUs to be used, max. 160 (or -c -1) then copy over mesh file to check striping setting,

  > lfs getstripe mesh

9. Result checking

  Results are generated in output_sfc/. Check output_ckp/ckp if the results have nan, make sure to meet stability criteria. in output_ckp/ckp first line:

  > STABILITY CRITERIA .5 > CMAX*DT/DX: (YOUR MODEL VALUE CAN NOT >= .5)

  in ckp file, 4th line value is DH (DISCRETIZATION IN SPACE) 5th line value is DT (DISCRETIZATION IN TIME) 7th line value is CMAX (HIGHEST P-VELOCITY ENCOUNTERED)

  example outputs in ckp are:

  ```
  STABILITY CRITERIA .5 > CMAX*DT/DX:     0.489063
  # OF X,Y,Z NODES PER PROC:      64, 64, 64
  # OF TIME STEPS:        2001
  DISCRETIZATION IN SPACE:        10.000000
  DISCRETIZATION IN TIME: 0.002500
  PML REFLECTION COEFFICIENT:     0.920000
  HIGHEST P-VELOCITY ENCOUNTERED: 1956.252441
  LOWEST P-VELOCITY ENCOUNTERED:  1213.844971
  HIGHEST S-VELOCITY ENCOUNTERED: 1206.831787
  LOWEST S-VELOCITY ENCOUNTERED:  307.113190
  HIGHEST DENSITY ENCOUNTERED:    1800.000000
  LOWEST  DENSITY ENCOUNTERED:    1700.000000
  SKIP OF SEISMOGRAMS IN TIME (LOOP COUNTER):     20
  ABC CONDITION, PML=1 OR CERJAN=0:       0
  FD SCHEME, VISCO=1 OR ELASTIC=0:        1
  Q, FAC,Q0,EX,FP:  1.000000, 150.000000, 0.600000, 0.500000
  20 :   -4.765808e-11   4.182098e-11   2.644118e-11
  40 :    9.461845e-07  -3.580994e-06   1.263729e-05
  60 :    1.063490e-05  -4.698996e-04   1.164289e-03
  80 :   -2.310131e-03  -2.625566e-03   5.300559e-03
  100 :   -9.565352e-03  -1.003452e-02   1.873909e-02
  120 :   -3.383595e-02  -3.024311e-02   5.216454e-02
  140 :   -9.331797e-02  -7.074796e-02   1.110601e-01
  160 :   -1.989845e-01  -4.708065e-02   7.033654e-03
  180 :    3.427376e-02  -6.913421e-03   8.890347e-03
  ...
  ```

#### Source File Processing

#### Mesh File Processing

#### I/O Behavior

#### Output

#### Lustre File System Specifics
