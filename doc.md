---
layout: page
title: Documentation
permalink: doc/
---

# Table of Contents
* Do not remove this line (it will not be displayed)
{:toc}

# Distribution Contents
* [src/](https://github.com/HPGeoC/awp-odc-os/tree/master/src) source code and platform dependent makefiles
* [doc/](https://github.com/HPGeoC/awp-odc-os/tree/gh-pages/doc.md) offline access to this documentation

# System Requirements
* C compiler
* CUDA compiler
* MPI library

# License
awp-odc-os is licensed under [BSD-2](https://github.com/HPGeoC/awp-odc-os/tree/master/LICENSE)


# Installation
To install awp-odc-os, perform the following steps:

1. Code access: [https://github.com/HPGeoC/awp-odc-os](https://github.com/HPGeoC/awp-odc-os). The master-branch (default) contains the latest published and tested version of awp-odc-os.
2. Prepare a directory for the setup and unpack awp-odc-os-master.zip

   ```
   unzip awp-odc-os-master.zip
   ```

   (```run/``` and ```input/``` see [Running awp-odc-os section](#running-awp-odc-os))

3. Compile code

   ```
   cd awp-odc-os-v1.0
   cd src
   ```

   (depends on the system, example below is based on Cray XK7 on [Blue Waters](https://bluewaters.ncsa.illinois.edu/user-guide) at NCSA)

   ```
   module swap PrgEnv-cray PrgEnv-gnu
   module load cudatoolkit
   module unload darshan
   make clean -f Makefile.[MACHINE].[COMPILER]
   make -f makefile.[MACHINE].[COMPILER]
   ```

   (```[MACHINE]``` represents machine name, e.g. ```titan```, ```bluewaters```)
   (```[COMPILER]``` represents compiler name, e.g. ```gnu```, ```pgi```, ```cray```)

4. Executable ```pmcl3d``` located in ```src/```

# Running awp-odc-os

1. Sample file access: http://hpgeoc.sdsc.edu/downloads/awp-odc-os-v1.0-example.tar.gz

2. Copy the zip file into awp-odc-os home directory and unpack. Two folders ```run/``` and ```input/``` will be extracted into the home directory.
   ```
   cd awp-odc-os-v1.0
   tar -zxvf ./awp-odc-os-v1.0-examples.tar.gz
   ```

3. Run the environment setting script. This script will prepare required folders and link executable and input files into ```run/``` folder.

   ```
   cd run
   ```

   (depends on the system, e.g. on Blue Waters)

   ```
   ./env.sh
   ```

   this script creates following folders:

   * ```input/``` - single small source and mesh input files (for small scale tests)
   * ```input_rst/``` - pre-partitioned source and mesh files (for large scale tests, see [Source file processing section](#source-file-processing) and [Mesh file processing section](#mesh-file-processing))
   * ```output_ckp/``` - run statistics and checkpoints if enabled
   * ```output_sfc/```  - output folder striping might be needed for lustre system


4. Submit pbs job from ```run/``` directory. Like the run script, the job submission process is platform dependent.
   On Blue Waters, for instance, the ```run.bluewaters.pbs``` script can be found in ```run/``` and submitted via (modify your pbs script - account, email address):

   ```
   qsub run.bluewaters.pbs
   ```

# Important Notes

1. Parameter settings reference info in src/command.c

2. Key model parameters of the executable (```pmcl3d```):

   | parameter(s)     | result                                                                        |
   |------------------|:------------------------------------------------------------------------------|
   |```-X -Y -Z```    | grid points in each direction (or NX, NY, NZ)                                 |
   |```-x -y```       | GPUs used in x/y direction each, total x*y GPUs used or NPX, NPY, NPZ(=1)     |
   |```--TMAX```      | total propagation time to run in seconds                                      |
   |```--DT```        | time step in seconds (total time steps are TMAX/DT)                           |
   |```--DH```        | discretization in space (spatial step for x, y, z (meters))                   |
   |```--NVAR```      | number of variables in a grid point                                           |
   |```--NVE```       | visco or elastic scheme (1=visco, 0=elastic)                                  |
   |```--NSRC```      | number of source nodes on fault                                               |
   |```--NST```       | number of time steps in rupture functions                                     |
   |```--IFAULT```    | mode selection and fault or initial stress setting (0-2)                      |
   |```--MEDIASTART```| initial media restart option(0=homogeneous)                                   |

3. Key I/O parameters of the executable (```pmcl3d```):

   | parameter(s)     | result                                                                           |
   |---------------------|:------------------------------------------------------------------------------|
   |```--READ_STEP```    | CPU reads # step sources from file system                                     |
   |```--READ_STEP_GPU```| CPU reads larger chunks and sends to GPU at every READ_STEP_GPU (when IFAULT=2, READ_STEP must be divisible by READ_STEP_GPU) |
   |```--WRITE_STEP```   | # timesteps to write the buffer to the files                                  |
   |```--NTISKP```       | # timesteps to skip to copy velocities from GPU to CPU                        |
   |```--NSKPX```        | # points to skip in recording points in X                                     |
   |```--NSKPY```        | # points to skip in recording points in Y                                     |

4. Other model parameters

   | parameter(s)     | result                                                                        |
   |------------------|:------------------------------------------------------------------------------|
   |```--ND```        | ABC thickness (grid-points), Cerjan >= 20                                     |
   |```--NPC```       | Cerjan(```0```), or PML(```1```), current version only implemented for Cerjan (NPC=```0```) ```(npx*npy*npz)/IOST < the total number of OSTs in a file system``` (IOST definition see section Lustre file system specifics) |
   |```--SoCalQ```    | parameter set for California Vp-Vs Q relationship (SoCalQ=```1```), default SoCalQ=```0``` |

5. Modify ```BLOCK_Z_SIZE``` in [src/pmcl3d_cons.h](https://github.com/HPGeoC/awp-odc-os/tree/master/src/pmcl3d_cons.h) manually parameter ```BLOCK_Z_SIZE``` must be powers of 2 (32, 64 ...) and can be divided by ```NZ BLCOK_SIZE_Z``` is preferred to be as big as possible for better performance.
   Default ```BLOCK_Z_SIZE``` in this version of awp-odc-os is 256.

6. Check ```--READ_STEP``` and ```--READ_STEP_GPU``` in input parameters.
   Current code require ```--READ_STEP``` must equals to ```--READ_STEP_GPU```

7. Example Inputs (http://hpgeoc.sdsc.edu/downloads/awp-odc-os-v1.0-example.tar.gz)
   * ```source```: moment source inputs (6 variables each time step)
                   101 steps source input file (binary)

     ```
     |   1st source   |      1st timestep      |      2nd timestep      | ...
     | location (int) |      value (float)     |      value (float)     | ...
     |     x, y, z    | xx, yy, zz, xy, xz, yz | xx, yy, zz, xy, xz, yz | ...
     |     101th timestep     |   2nd source   |      1st timestep      | ...
     |      value (float)     | location (int) |      value (float)     | ...
     | xx, yy, zz, xy, xz, yz |     x, y, z    | xx, yy, zz, xy, xz, yz | ...
     ```

   * ```mesh```: mesh file for 320x320x2048 size (binary)

     ```
     |  (x, y, z)  |  (x, y, z)  | ... |  (x, y, z)  |  (x, y, z)  | ...
     |  (1, 1, 1)  |  (2, 1, 1)  | ... |  (1, 2, 1)  |  (2, 2, 1)  | ...
     | vp, vs, den | vp, vs, den | ... | vp, vs, den | vp, vs, den | ...
     ```

8. Lustre Striping

   For large scale run using more than tens of GPUs on lustre file system, striping is needed.
   Visit system user guide for more info about striping general information for 320x320x2048, 2x2, ```NTISKP```=20, ```write_step```=100, ```nskpx```=2, ```nskpy```=2 each core holds data size ```(160/2)*(160/2)*1*4_bytes*100_write_steps=2.44MB```

   ```
   lfs setstripe -s 3m -c -1 -i -1 output_sfc
   ```

   mesh read uses mpi-io, each core data size ```160*160*2048*3_variables*4=600MB``` first setup striping for a new file named ```mesh```

   ```
   lfs setstrspe -s 600m -c 4 -i -1 mesh
   ```

   stripe_count best equal nr of GPUs to be used, max. 160 (or ```-c -1```) then copy over mesh file to check striping setting,

   ```
   lfs getstripe mesh
   ```

9. Result checking

   Results are generated in ```output_sfc/```. Check ```output_ckp/ckp``` if the results have nan, make sure to meet stability criteria. in ```output_ckp/ckp``` first line:

   ```
   STABILITY CRITERIA .5 > CMAX*DT/DX: (YOUR MODEL VALUE CAN NOT >= .5)
   ```
 
   in ckp file, 4th line value is ```DH``` (```DISCRETIZATION IN SPACE```)
   5th line value is ```DT``` (```DISCRETIZATION IN TIME```)
   7th line value is ```CMAX``` (```HIGHEST P-VELOCITY ENCOUNTERED```)

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
   ```

# Source file processing
The parameter ```IFAULT``` controls how source files are read into awp-odc-os, the type of simulation to run, and what data to output. Here is a table describing ```IFAULT``` options:

| ```IFAULT``` | source read                    | simulation type                     | output                        |
|--------------|:-------------------------------|:------------------------------------|-------------------------------|
| 0            | serial read of 1 file (ascii)  | wave propagation, small scale tests | wave velocity (surface/volume) |
| 1            | serial read of 1 file (binary) | wave propagation, small scale tests | wave velocity (surface/volume)
| 2            | MPI-IO to read 1 file (binary) | wave propagation, large scale       | wave velocity (surface/volume)

For ```IFAULT``` = 0,1,2 the user can select to write only surface velocity or both surface and volume velocity. If the parameter ```ISFCVLM``` = 0, only surface velocity is written.

When ```ISFCVLM``` = 1, both volume and surface velocity are written. Surfaces and volumes of interest can also be specified by the user. Each direction (X,Y,Z) has 6 parameters that determine the observation resolution and observation size for surface and volume. Letting ```[W]``` represent the X, Y, or Z direction, the following table shows the decimation parameters associated with each value of ```IFAULT```.

Source file locations in the cases ```IFAULT```=0,1 are specified in the run scripts.
The line

```
--INSRC input/source
```

specifies that the text based source file should be read from the ```input/``` directory.

```
------------------------------------------------------------------------------
IFAULT   NBG[W]  NED[W]   NSKP[W]  NBG[W]2   NED[W]2   NSKP[W]2
------------------------------------------------------------------------------
0        x       x        x        x         x         x
1        x       x        x        x         x         x
2        x       x        x        x         x         x
         SURFACE DECIMATION          VOLUME DECIMATION
------------------------------------------------------------------------------
```

# Mesh file processing
  * The parameter ```MEDIARESTART``` controls how the mesh file is read in.  Currently,
    the user has 3 options to choose from:

    | ```MEDIARESTART``` | description             | uses                                   |
    |:-------------------|:------------------------|:---------------------------------------|
    | 0                  | create homogeneous mesh | fast initialization, small scale tests |
    | 1                  | serial read of 1 file   | small scale tests                      |
    | 2                  | MPI-IO to read 1 file   | large scale run, **recommended**       |

  * ```NVAR``` specifies the number of variables for each grid point in a mesh file
    There are three different cases:

    | ```NVAR_VALUE``` | ```ACT_NVAR``` | variables                  |
    |:-----------------|:---------------|----------------------------|
    | 3                | 3              | [vp,vs,dd] **recommended** |
    | 5                | 5              | [vp,vs,dd,pq,sq]           |
    | 8                | 5              | [x,y,z,vp,vs,dd,pq,sq]     |

Memory limitations for large-scale simulations necessitate partitioning in the x direction when ```MEDIARESTART``` = 2. The following constraints must be placed on the partioning:
  1. ```real(nx*ny*(nvar+act_nvar)*4)/real(PARTDEG) < MEMORY SIZE```
  2. ```npy``` should be divisible by ```partdeg``` and ```npx*npy*npz >= nz*PARTDEG```
  3. ```npy and ny >= PARTDEG```
  4. ```npx*npy*npz > nz*PARTDEG```


# I/O behavior
```IO_OPT``` enables or disables data output. The user has complete control of how much simulation data should be stored, how much of the computational domain should be sampled, and how often this stored data is written to file.

```NTISKP``` (```NTISKP2```) specify how many timesteps to skip when recording simulation data.  The default value for both parameter is 1

| parameter   | use                                           |
|:------------|:----------------------------------------------|
|```NTISKP``` | wave propagation mode surface velocity output |
|```NTISKP2```| wave propagation mode volume velocity output  |

For example, if ```NTISKP``` = 5, relevant data is recorded every 5 timesteps
and stored in temporary buffers.

```READ_STEP```, ```READ_STEP_GPU``` determines how often buffered data is written to CPU (```READ_STEP```)
and GPU (```READ_STEP_GPU```). The default ```READ_STEP```=```READ_STEP_GPU```.

| parameter         | use                                                   |
|:------------------|:------------------------------------------------------|
| ```READ_STEP```   | source input read from file system to cpu, # of steps |
|```READ_STEP_GPU```|  source input read from CPU to GPU, # of steps<br/> ```READ_STEP_GPU``` <= ```READ_STEP```<br/> CPU reads larger chunks and sends to GPU at every ```READ_STEP_GPU``` (when ```IFAULT```=2, ```READ_STEP``` must be divisible by ```READ_STEP_GPU```) |


```WRITE_STEP``` (```WRITE_STEP2```) determines how often buffered data is written to output files. The default value for both parameter is 1.


| parameter         | use                                           |
|:------------------|:----------------------------------------------|
|```WRITE_STEP```   | wave propagation mode surface velocity output |
|```WRITE_STEP2```  | wave propagation mode volume velocity output  |

```NTISKP``` (```NTISKP2```) and ```WRITE_STEP``` (```WRITE_STEP2```) together determine how often
file writing is performed. Output file(s) is(are) accessed every ```NTISKP*WRITE_STEP``` timesteps or ```NTISKP2*WRITE_STEP2``` timesteps, depending on ```IFAULT```.

# Output
* ```output_sfc/```: wave propagation mode surface velocity output
* ```output_vlm/```: wave propagation mode volume velocity output
* ```output_ckp/```: check points output

For example, ```TMAX```=300, ```DT```=0.005, ```NTISKP```=5, ```NBGX```=1, ```NEGX```=2800, ```NSKPX```=1,
```NBGY```=1, ```NEGY```=2800, ```NSKPY```=1, ```NBGZ```=1, ```NEGZ```=1, ```NSKPZ```=1, ```WRITE_STEP```=1000.

| setting                         | result                                            |
|:--------------------------------|:--------------------------------------------------|
|```TMAX/DT=60000```              | the output contains 60000 time steps              |
|```NTISKP=5```                   | Only output time_step=5,10,15,...,60000           |
|```NBG[X-Z],NEG[X-Z],NSKP[X-Z]```| Output volume: 2800x2800x1. Only output surface.  |
|```WRITE_STEP```                 | Write a file every 1000 time steps                |

So for ```output_sfc```, the output files will be:

```
S[X-Z]96PS0005000 S[X-Z]96PS0010000 S[X-Z]96PS0015000 ... S[X-Z]96PS0060000
```

Each file has ```floatsize*nx*ny*write_step=4*2800*2800*1000=31360000000``` bytes.

The current version of awp-odc-os supports fast-X as output format (fast-X format : efficient for visualization operation):

```
|          Time Step 1          |Time Step 2| ... |Time Step n|
|(1,1,1)|(2,1,1)|...|(nx,ny,nz) |    ...    | ... |    ...    |
```

# Lustre file system specifics
The Lustre file system provides a means of parallelizing sequential I/O operations,
called file striping. File striping distributes reading and writing operations
across multiple Object Storage Targets (OSTs). In awp-odc-os, multiple OSTs can be
utilized for file checkpointing and mesh reading.
