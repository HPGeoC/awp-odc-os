/**
 @brief Reads command line arguments.
 
 @bug If any file name is longer than 50 characters then the @c command function will copy past the end of
 the corresponding @c char array.
 
 @section LICENSE
 Copyright (c) 2013-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
**************************************************************************************************************** 
*  command.c						                                                                           *
*  Process Command Line	                                                                                       *
*                                                                                                              *
*  Name         Type        Command             Description 	                                               *
*  TMAX         <FLOAT>       -T              propagation time	                	                           *
*  DH           <FLOAT>       -H              spatial step for x, y, z (meters)                                *
*  DT           <FLOAT>       -t              time step (seconds)                                              *
*  ARBC         <FLOAT>       -A              coefficient for PML (3-4), or Cerjan (0.90-0.96)                 *
*  PHT          <FLOAT>       -P                                                                               *
*  NPC          <INTEGER>     -M              PML or Cerjan ABC (1=PML, 0=Cerjan)                              *
*  ND           <INTEGER>     -D              ABC thickness (grid-points) PML <= 20, Cerjan >= 20              *
*  NSRC         <INTEGER>     -S              number of source nodes on fault                                  *
*  NST          <INTEGER>     -N              number of time steps in rupture functions                        *
*  NVAR         <INTEGER>     -n              number of variables in a grid point                              *
*  NVE          <INTEGER>     -V              visco or elastic scheme (1=visco, 0=elastic)                     *
*  MEDIASTART   <INTEGER>     -B              initial media restart option(0=homogenous)                       *
*  IFAULT       <INTEGER>     -I              mode selection and fault or initial stress setting (1 or 2)      *
*  READ_STEP    <INTEGER>     -R                                                                               *
*  READ_STEP_GPU<INTEGER>     -Q              CPU reads larger chunks and sends to GPU at every READ_STEP_GPU  *
*                                               (IFAULT=2) READ_STEP must be divisible by READ_STEP_GPU        *
*  NX           <INTEGER>     -X              x model dimension in nodes                                       *      
*  NY           <INTEGER>     -Y              y model dimension in nodes                                       *
*  NZ           <INTEGER>     -Z              z model dimension in nodes                                       *
*  PX           <INTEGER>     -x              number of processors in the x direction                          *
*  PY           <INTEGER>     -y              number of processors in the y direction                          *
*  NBGX         <INTEGER>                     index (starts with 1) to start recording points in X             *
*  NEDX         <INTEGER>                     index to end recording points in X (-1 for all)                  *
*  NSKPX        <INTEGER>                     #points to skip in recording points in X                         *
*  NBGY         <INTEGER>                     index to start recording points in Y                             *
*  NEDY         <INTEGER>                     index to end recording points in Y (-1 for all)                  *
*  NSKPY        <INTEGER>                     #points to skip in recording points in Y                         *
*  NBGZ         <INTEGER>                     index to start recording points in Z                             *
*  NEDZ         <INTEGER>                     index to end recording points in Z (-1 for all)                  *
*  NSKPZ        <INTEGER>                     #points to skip in recording points in Z                         *
*  IDYNA        <INTEGER>     -i              mode selection of dynamic rupture model                          *
*  SoCalQ       <INTEGER>     -s              Southern California Vp-Vs Q relationship enabling flag           *
*  FAC          <FLOAT>       -l              Q                                                                *
*  Q0           <FLOAT>       -h              Q                                                                *
*  EX           <FLOAT>       -x              Q                                                                *
*  FP           <FLOAT>       -p              Q bandwidth central frequency                                    *
*  NTISKP       <INTEGER>     -r              # timesteps to skip to copy velocities from GPU to CPU           *
*  WRITE_STEP   <INTEGER>     -W              # timesteps to write the buffer to the files                     *
*                                               (written timesteps are n*NTISKP*WRITE_STEP for n=1,2,...)      *
*  INSRC        <STRING>                      source input file (if IFAULT=2, then this is prefix of tpsrc)    *
*  INVEL        <STRING>                      mesh input file                                                  *
*  INSRC_I2     <STRING>                      split source input file prefix for IFAULT=2 option               *
*  CHKFILE      <STRING>      -c              Checkpoint statistics file to write to                           *
****************************************************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <math.h>
#include <string.h>

// Default IN3D Values
const float def_TMAX       = 20.00;
const float def_DH         = 200.0;
const float def_DT         = 0.01; 
const float def_ARBC       = 0.92;
const float def_PHT        = 0.1;

const int   def_NPC        = 0;
const int   def_ND         = 20;
const int   def_NSRC       = 1;
const int   def_NST        = 91;
const int   def_NVAR       = 3;

const int   def_NVE        = 1;
const int   def_MEDIASTART = 0;
const int   def_IFAULT     = 1; 
const int   def_READ_STEP  = 91;
const int   def_READ_STEP_GPU = 91;

const int   def_NTISKP     = 10;
const int   def_WRITE_STEP = 10;

const int   def_NX         = 224;
const int   def_NY         = 224; 
const int   def_NZ         = 1024;

const int   def_PX         = 1;
const int   def_PY         = 1;

const int   def_NBGX       = 1;
const int   def_NEDX       = -1;   // use -1 for all
const int   def_NSKPX      = 1;
const int   def_NBGY       = 1;
const int   def_NEDY       = -1;   // use -1 for all
const int   def_NSKPY      = 1;
const int   def_NBGZ       = 1;
const int   def_NEDZ       = 1;    // only surface
const int   def_NSKPZ      = 1;

const int   def_IDYNA      = 0;
const int   def_SoCalQ     = 1;

const float def_FAC        = 0.005;
const float def_Q0         = 5.0;
const float def_EX         = 0.0;
const float def_FP         = 2.5;

const char  def_INSRC[50]  = "input/FAULTPOW";
const char  def_INVEL[50]  = "input/media";

const char  def_OUT[50] = "output_sfc";

const char  def_INSRC_TPSRC[50] = "input_rst/srcpart/tpsrc/tpsrc";
const char  def_INSRC_I2[50]  = "input_rst/srcpart/split_faults/fault";

const char  def_CHKFILE[50]   = "output_sfc/CHKP";



/**
 Reads command line arguments and assigns them to corresponding variables.
 
 @param argc                    Number of command line arguments (passed from @c main function )
 @param argv                    Command line arguments (passed from @c main function)
 @param[out] TMAX               Total simulation time in seconds. Set with command line option @c -T or @c --TMAX. Defaults to 20.0
 @param[out] DH                 Spatial step size for x, y, and z dimensions in meters. Set with command line option @c -H or @c --DH. Defaults to 200.0
 @param[out] DT                 Time step size in seconds. Set with command line option @c -t or @c --DT. Defaults to 0.01
 @param[out] ARBC               Coefficient for PML (3-4), or Cerjan (0.90-0.96). Set with command line option @c -A or @c --ARBC. Defaults to 0.92
 @param[out] PHT                Set with command line option @c -P or @c --PHT. Defaults to 0.1
 @param[out] NPC                PML or Cerjan ABC (1=PML, 0=Cerjan). Set with command line option @c -M or @c --NPC. Defaults to 0
 @param[out] ND                 ABC thickness (grid-points) PML <= 20, Cerjan >= 20. Set with command line option @c -D or @c --ND. Defaults to 20
 @param[out] NSRC               Number of source nodes on fault. Set with command line option @c -S or @c --NSRC. Defaults to 1
 @param[out] NST                Number of time steps in rupture functions. Set with command line option @c -N or @c --NST. Defaults to 91
 @param[out] NVAR               Number of variables in a grid point. Set with command line option @c -n or @c --NVAR. Defaults to 3
 @param[out] NVE                Visco or elastic scheme (1=visco, 0=elastic). Set with command line option @c -V or @c --NVE. Defaults to 1
 @param[out] MEDIASTART         Initial media restart option(0=homogenous). Set with command line option @c -B or @c --MEDIASTART. Defaults to 0
 @param[out] IFAULT             Mode selection and fault or initial stress setting (1 or 2). Set with command line option @c -I or @c --IFAULT. Defaults to 1
 @param[out] READ_STEP          Number of rupture timesteps to read at a time from the source file. Set with command line option @c -R or @c --READ_STEP. Defaults to 91
 @param[out] READ_STEP_GPU      CPU reads larger chunks and sends to GPU at every @c READ_STEP_GPU. 
                                    If @c IFAULT==2 then @c READ_STEP must be divisible by @c READ_STEP_GPU. Set with command line option @c -Q or 
                                    @c --READ_STEP_GPU. Defaults to 91
 @param[out] NTISKP             Number of timesteps to skip when copying velocities from GPU to CPU. Set with command line option @c -r or @c --NTISKP. Defaults to 10
 @param[out] WRITE_STEP         Number of timesteps to skip when writing velocities from CPU to files. So the timesteps that get written to the files are
                                    ( @c n*NTISKP*WRITE_STEP for @c n=1,2,...). Set with command line option @c -W or @c --WRITE_STEP. Defaults to 10
 @param[out] NX                 Number of nodes in the x dimension. Set with command line option @c -X or @c --NX. Defaults to 224
 @param[out] NY                 Number of nodes in the y dimension. Set with command line option @c -Y or @c --NY. Defaults to 224
 @param[out] NZ                 Number of nodes in the z dimension. Set with command line option @c Z or @c --NZ. Defaults to 1024
 @param[out] PX                 Number of processes in the x dimension (using 2 dimensional MPI topology). Set with command line option @c -x
                                    or @c --PX. Defaults to 1
 @param[out] PY                 Number of processes in the y dimension (using 2 dimensional MPI topology). Set with command line option @c -y
                                    or @c --PY. Defaults to 1
 @param[out] NBGX               Index (starting from 1) of the first x node to record values at (e.g. if @c NBGX==10, then the output file
                                    will not have data for the first 9 nodes in the x dimension). Set with command line option @c -1 or @c --NBGX.
                                    Defaults to 1
 @param[out] NEDX               Index (starting from 1) of the last x node to record values at. Set to -1 to record all the way to the end. Set with
                                    command line option @c -2 or @c --NEDX. Defaults to -1
 @param[out] NSKPX              Number of nodes to skip in the x dimension when recording values. (e.g. if @c NBGX==10, @c NEDX==40, @c NSKPX==10, then
                                    x nodes 10, 20, 30, and 40 will have their values recorded in the output file. Set with command line option
                                    @c -3 or @c --NSKPX. Defaults to 1
 @param[out] NBGY               Index (starting from 1) of the first y node to record values at. Set with command line option @c -11 or @c --NEDX.
                                    Defaults to 1
 @param[out] NEDY               Index (starting from 1) of the last y node to record values at. Set to -1 to record all the way to the end. Set with
                                    command line option @c -12 or @c --NEDY. Defaults to -1
 @param[out] NSKPY              Number of nodes to skip in the y dimension when recording values. Set with command line option
                                    @c -13 or @c --NSKPY. Defaults to 1
 @param[out] NBGZ               Index (starting from 1) of the first x node to record values at. Note that z==1 is the surface node. Set with
                                    command line option @c -21 or @c --NBGZ. Defaults to 1
 @param[out] NEDZ               Index (starting from 1) of the last z node to record values at. Note that z==1. Set to -1 to record all the way to the end. 
                                    Set with command line option @c -22 or @c --NEDZ. Defaults to 1 (only records surface nodes)
 @param[out] NSKPZ              Number of nodes to skip in the z dimension when recording values. Set with command line option
                                    @c -23 or @c --NSKPZ. Defaults to 1
 @param[out] FAC                Set with command line option @c -l or @c --FAC. Defaults to 0.005
 @param[out] Q0                 Set with command line option @c -h or @c --Q0. Defaults to 5.0
 @param[out] EX                 Set with command line option @c -x or @c --EX. Defaults 0.0
 @param[out] FP                 Q bandwidth central frequency. Set with command line option @c -q or @c --FP. Defaults to 2.5
 @param[out] IDYNA              Mode selection of dynamic rupture model. Set with command line option @c -i or @c --IDYNA. Defaults to 0
 @param[out] SoCalQ             Southern California Vp-Vs Q relationship enabling flag. Set with command line option @c -s or @c --SoCalQ. Defaults to 1
 @param[out] INSRC              Source input file (if @c IFAULT==2, then this is prefix of @c tpsrc). Set with command line option @c -100 @c --INSRC.
                                    Defaults to @c "input/FAULTPOW"
 @param[out] INVEL              Mesh input file. Set with command line option @c - 101 or @c --INVEL. Defaults to @c "input/media"
 @param[out] OUT                Output folder. Set with command line option @c -o or @c --OUT. Defaults to @c "output_sfc"
 @param[out] INSRC_I2           Split source input file prefix for @c IFAULT==2 option. Set with command line option @c -102 or @c --INSRC_I2.
                                    Defaults to @c "input_rst/srcpart/split_faults/fault"
 @param[out] CHKFILE            Checkpoint statistics file to write to. Set with command line option @c -o or @c --CHKFILE. Defaults to
                                    @c "output_sfc/CHKP"
 
 @warning The number of MPI processes must be greater than or equal to @c PX*PY
 @warning If @c IFAULT==2 then @c READ_STEP must be divisible by @c READ_STEP_GPU
 
 @warning All file names must be under 50 characters in length!!!
 */
void command(int argc,    char **argv,
	     float *TMAX, float *DH,       float *DT,   float *ARBC,    float *PHT,
             int *NPC,    int *ND,         int *NSRC,   int *NST,       int *NVAR,
             int *NVE,    int *MEDIASTART, int *IFAULT, int *READ_STEP, int *READ_STEP_GPU,
             int *NTISKP, int *WRITE_STEP,
    	       int *NX,     int *NY,         int *NZ,     int *PX,        int *PY,
             int *NBGX,   int *NEDX,       int *NSKPX, 
             int *NBGY,   int *NEDY,       int *NSKPY, 
             int *NBGZ,   int *NEDZ,       int *NSKPZ, 
             float *FAC,   float *Q0,      float *EX,   float *FP,   int *IDYNA,     int *SoCalQ,
             char *INSRC, char *INVEL,     char *OUT,   char *INSRC_I2, char *CHKFILE)
{

   // Fill in default values
   *TMAX       = def_TMAX;
   *DH         = def_DH;
   *DT         = def_DT;
   *ARBC       = def_ARBC;
   *PHT        = def_PHT;

   *NPC        = def_NPC;
   *ND         = def_ND;
   *NSRC       = def_NSRC;
   *NST        = def_NST;
   
   *NVE        = def_NVE;
   *MEDIASTART = def_MEDIASTART;
   *NVAR       = def_NVAR;
   *IFAULT     = def_IFAULT; 
   *READ_STEP  = def_READ_STEP;
   *READ_STEP_GPU = def_READ_STEP_GPU;

   *NTISKP     = def_NTISKP;
   *WRITE_STEP = def_WRITE_STEP;

   *NX         = def_NX;
   *NY         = def_NY;
   *NZ         = def_NZ;
   *PX         = def_PX;
   *PY         = def_PY;

   *NBGX       = def_NBGX;
   *NEDX       = def_NEDX;
   *NSKPX      = def_NSKPX;
   *NBGY       = def_NBGY;
   *NEDY       = def_NEDY;
   *NSKPY      = def_NSKPY;
   *NBGZ       = def_NBGZ;
   *NEDZ       = def_NEDZ;
   *NSKPZ      = def_NSKPZ;

   *IDYNA      = def_IDYNA;
   *SoCalQ     = def_SoCalQ;
   *FAC        = def_FAC;
   *Q0         = def_Q0;
   *EX         = def_EX;
   *FP         = def_FP;

    strcpy(INSRC, def_INSRC);
    strcpy(INVEL, def_INVEL);
    strcpy(OUT, def_OUT);
    strcpy(INSRC_I2, def_INSRC_I2);
    strcpy(CHKFILE, def_CHKFILE);

    extern char *optarg;
    static const char *optstring = "-T:H:t:A:P:M:D:S:N:V:B:n:I:R:Q:X:Y:Z:x:y:z:i:l:h:30:p:s:r:W:1:2:3:11:12:13:21:22:23:100:101:102:o:c:";
    static struct option long_options[] = {
        {"TMAX", required_argument, NULL, 'T'},
        {"DH", required_argument, NULL, 'H'},
        {"DT", required_argument, NULL, 't'},
        {"ARBC", required_argument, NULL, 'A'},
        {"PHT", required_argument, NULL, 'P'},
        {"NPC", required_argument, NULL, 'M'},
        {"ND", required_argument, NULL, 'D'},
        {"NSRC", required_argument, NULL, 'S'},
        {"NST", required_argument, NULL, 'N'},
        {"NVE", required_argument, NULL, 'V'},
        {"MEDIASTART", required_argument, NULL, 'B'},
        {"NVAR", required_argument, NULL, 'n'},
        {"IFAULT", required_argument, NULL, 'I'},
        {"READ_STEP", required_argument, NULL, 'R'},
        {"READ_STEP_GPU", required_argument, NULL, 'Q'},
        {"NX", required_argument, NULL, 'X'},
        {"NY", required_argument, NULL, 'Y'},
        {"NZ", required_argument, NULL, 'Z'},
        {"PX", required_argument, NULL, 'x'},
        {"PY", required_argument, NULL, 'y'},
        {"NBGX", required_argument, NULL, 1},
        {"NEDX", required_argument, NULL, 2},
        {"NSKPX", required_argument, NULL, 3},
        {"NBGY", required_argument, NULL, 11},
        {"NEDY", required_argument, NULL, 12},
        {"NSKPY", required_argument, NULL, 13},
        {"NBGZ", required_argument, NULL, 21},
        {"NEDZ", required_argument, NULL, 22},
        {"NSKPZ", required_argument, NULL, 23},
        {"IDYNA", required_argument, NULL, 'i'},
        {"SoCalQ", required_argument, NULL, 's'},
        {"FAC", required_argument, NULL, 'l'},
        {"Q0", required_argument, NULL, 'h'},
        {"EX", required_argument, NULL, 30},
        {"FP", required_argument, NULL, 'p'},
        {"NTISKP", required_argument, NULL, 'r'},
        {"WRITE_STEP", required_argument, NULL, 'W'},
        {"INSRC", required_argument, NULL, 100},
        {"INVEL", required_argument, NULL, 101},
        {"OUT", required_argument, NULL, 'o'},
        {"INSRC_I2", required_argument, NULL, 102},
        {"CHKFILE", required_argument, NULL, 'c'},
    };

    // If IFAULT=2 and INSRC is not set, then *INSRC = def_INSRC_TPSRC, not def_INSRC
    int insrcIsSet = 0;
    // If IFAULT=1 and READ_STEP_GPU is not set, it should be = READ_STEP
    int readstepGpuIsSet = 0;
    int c;
    while ((c=getopt_long(argc, argv, optstring, long_options, NULL)) != -1)
    {
        switch (c) {
            case 'T':
                *TMAX       = atof(optarg); break;
            case 'H':
                *DH         = atof(optarg); break;
            case 't':
                *DT         = atof(optarg); break;
            case 'A':
                *ARBC       = atof(optarg); break;
            case 'P':
                *PHT        = atof(optarg); break;
            case 'M':
                *NPC        = atoi(optarg); break;
            case 'D':
                *ND         = atoi(optarg); break;
            case 'S':
                *NSRC       = atoi(optarg); break;
            case 'N':
                *NST        = atoi(optarg); break;
            case 'V':
                *NVE        = atoi(optarg); break;
            case 'B':
                *MEDIASTART = atoi(optarg); break;
            case 'n':
                *NVAR       = atoi(optarg); break;
            case 'I':
	              *IFAULT     = atoi(optarg); break;
            case 'R':
                *READ_STEP  = atoi(optarg); break;		
            case 'Q':
                readstepGpuIsSet = 1;
                *READ_STEP_GPU  = atoi(optarg); break;		
            case 'X':
                *NX         = atoi(optarg); break;
            case 'Y':
                *NY         = atoi(optarg); break;
            case 'Z':
                *NZ         = atoi(optarg); break;
            case 'x':
                *PX         = atoi(optarg); break;
            case 'y':
                *PY         = atoi(optarg); break;
            case 1:
                *NBGX       = atoi(optarg); break;
            case 2:
                *NEDX       = atoi(optarg); break;
            case 3:
                *NSKPX      = atoi(optarg); break;
            case 11:
                *NBGY       = atoi(optarg); break;
            case 12:
                *NEDY       = atoi(optarg); break;
            case 13:
                *NSKPY      = atoi(optarg); break;
            case 21:
                *NBGZ       = atoi(optarg); break;
            case 22:
                *NEDZ       = atoi(optarg); break;
            case 23:
                *NSKPZ      = atoi(optarg); break;
            case 'i':
                *IDYNA      = atoi(optarg); break;
            case 's':
                *SoCalQ     = atoi(optarg); break;
            case 'l':
                *FAC        = atof(optarg); break;
            case 'h':
                *Q0         = atof(optarg); break;
            case 30:
                *EX         = atof(optarg); break;
            case 'p':
                *FP         = atof(optarg); break;
            case 'r':
                *NTISKP     = atoi(optarg); break;
            case 'W':
                *WRITE_STEP = atoi(optarg); break;
            case 100:
                insrcIsSet = 1;
                strcpy(INSRC, optarg); break;
            case 101:
                strcpy(INVEL, optarg); break;
            case 'o':
                strcpy(OUT, optarg); break;
            case 102:
                strcpy(INSRC_I2, optarg); break;
            case 'c':
                strcpy(CHKFILE, optarg); break;
            default:
                printf("Usage: %s \nOptions:\n\t[(-T | --TMAX) <TMAX>]\n\t[(-H | --DH) <DH>]\n\t[(-t | --DT) <DT>]\n\t[(-A | --ARBC) <ARBC>]\n\t[(-P | --PHT) <PHT>]\n\t[(-M | --NPC) <NPC>]\n\t[(-D | --ND) <ND>]\n\t[(-S | --NSRC) <NSRC>]\n\t[(-N | --NST) <NST>]\n",argv[0]);
                printf("\n\t[(-V | --NVE) <NVE>]\n\t[(-B | --MEDIASTART) <MEDIASTART>]\n\t[(-n | --NVAR) <NVAR>]\n\t[(-I | --IFAULT) <IFAULT>]\n\t[(-R | --READ_STEP) <x READ_STEP for CPU>]\n\t[(-Q | --READ_STEP_GPU) <READ_STEP for GPU>]\n");
                printf("\n\t[(-X | --NX) <x length]\n\t[(-Y | --NY) <y length>]\n\t[(-Z | --NZ) <z length]\n\t[(-x | --NPX) <x processors]\n\t[(-y | --NPY) <y processors>]\n\t[(-z | --NPZ) <z processors>]\n");
                printf("\n\t[(-1 | --NBGX) <starting point to record in X>]\n\t[(-2 | --NEDX) <ending point to record in X>]\n\t[(-3 | --NSKPX) <skipping points to record in X>]\n\t[(-11 | --NBGY) <starting point to record in Y>]\n\t[(-12 | --NEDY) <ending point to record in Y>]\n\t[(-13 | --NSKPY) <skipping points to record in Y>]\n\t[(-21 | --NBGZ) <starting point to record in Z>]\n\t[(-22 | --NEDZ) <ending point to record in Z>]\n\t[(-23 | --NSKPZ) <skipping points to record in Z>]\n");
                printf("\n\t[(-i | --IDYNA) <i IDYNA>]\n\t[(-s | --SoCalQ) <s SoCalQ>]\n\t[(-l | --FAC) <l FAC>]\n\t[(-h | --Q0) <h Q0>]\n\t[(-30 | --EX) <e EX>]\n\t[(-p | --FP) <p FP>]\n\t[(-r | --NTISKP) <time skipping in writing>]\n\t[(-W | --WRITE_STEP) <time aggregation in writing>]\n");
                printf("\n\t[(-100 | --INSRC) <source file>]\n\t[(-101 | --INVEL) <mesh file>]\n\t[(-o | --OUT) <output file>]\n\t[(-102 | --INSRC_I2) <split source file prefix (IFAULT=2)>]\n\t[(-c | --CHKFILE) <checkpoint file to write statistics>]\n\n");
                exit(-1);
        }
    }
    // If IFAULT=2 and INSRC is not set, then *INSRC = def_INSRC_TPSRC, not def_INSRC
    if(*IFAULT == 2 && !insrcIsSet){
      strcpy(INSRC, def_INSRC_TPSRC);
    }
    if(!readstepGpuIsSet){
      *READ_STEP_GPU = *READ_STEP;
    }
    return;
}
