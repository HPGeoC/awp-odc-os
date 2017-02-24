/**
@section LICENSE
Copyright (c) 2013-2017, Regents of the University of California, San Diego State University
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "pmcl3d.h"
#include <assert.h>
const char *getTime() {
    time_t mytime;
    mytime = time(NULL);
    return ctime(&mytime);
}

#define RANK0PRINT(string, ...) \
if (0 == rank) { \
  fprintf(stderr, "%s: " string, getTime(), ##__VA_ARGS__); \
}


const double   micro = 1.0e-6;

void SetDeviceConstValue(float DH, float DT, int nxt, int nyt, int nzt,
   float fmajor, float fminor, float *Rz, float *RzT);
void BindArrayToTexture(float* vx1, float* vx2, int* ww,float* wwo, int memsize);
void UnBindArrayFromTexture();
void dvelcx_H(float* u1,    float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy,       float* xz, float* yz,
              float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, int nyt,   int nzt,   cudaStream_t St, int s_i,   int e_i);

void dvelcx_H_opt(float* u1,    float* v1,    float* w1,
                  float* xx,  float* yy, float* zz, float* xy,
                  float* xz, float* yz,
                  float* dcrjx, float* dcrjy, float* dcrjz,
                  float* d_1, int nyt,   int nzt,
                  cudaStream_t St, int s_i,   int e_i);

void dvelcy_H(float* u1,       float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy,   float* xz,   float* yz,
              float* dcrjx,    float* dcrjy, float* dcrjz, float* d_1, int nxt,   int nzt,   float* s_u1, float* s_v1, float* s_w1,
              cudaStream_t St, int s_j,      int e_j,      int rank);

void dstrqc_H(float* xx,       float* yy,     float* zz,    float* xy,    float* xz, float* yz,
              float* r1,       float* r2,     float* r3,    float* r4,    float* r5, float* r6,
              float* u1,       float* v1,     float* w1,    float* lam,   float* mu, float* qp, float* coeff,
              float* qs,       float* dcrjx,  float* dcrjy, float* dcrjz, int nyt,   int nzt,
              cudaStream_t St, float* lam_mu, int NX,       int rankx,    int ranky, int s_i,
              int e_i,         int s_j,       int e_j);

void dstrqc_H_new(float* xx,       float* yy,     float* zz,    float* xy,    float* xz, float* yz,
                  float* r1,       float* r2,     float* r3,    float* r4,    float* r5, float* r6,
                  float* u1,       float* v1,     float* w1,    float* lam,   float* mu, float* qp,float* coeff,
                  float* qs,       float* dcrjx,  float* dcrjy, float* dcrjz, int nyt,   int nzt,
                  cudaStream_t St, float* lam_mu,
                  //float *vx1, float *vx2, float *ww, float *wwo, // peng's version
                  float *vx1, float *vx2, int *ww, float *wwo,
                  int NX,       int rankx,    int ranky, int  s_i,
                  int e_i,         int s_j,       int e_j);


void fstr_H(float* zz, float* xz, float* yz, cudaStream_t St, int s_i, int e_i, int s_j, int e_j);

/*void drprecpc_calc_H(float *xx, float *yy, float *zz, float *xy, float *xz, float *yz,
        float *mu, float *d1, float *inixx, float *iniyy, float *inizz,
        float *inixy, float *inixz, float *iniyz,
        float *yldfac,float *cohes, float *pfluid, float *phi,
        float *EPxx, float *EPyy, float *EPzz, float *neta,
        float *buf_L, float *buf_R, float *buf_F, float *buf_B,
        float *buf_FL, float *buf_FR, float *buf_BL, float *buf_BR,
        int nzt,
        int xls, int xre, int yls, int yre, cudaStream_t St);*/
void drprecpc_calc_H_opt(float *xx, float *yy, float *zz, float *xy, float *xz, float *yz,
        float *mu, float *d1, float *sigma2,
        float *yldfac,float *cohes, float *phi,
        float *neta,
        float *buf_L, float *buf_R, float *buf_F, float *buf_B,
        float *buf_FL, float *buf_FR, float *buf_BL, float *buf_BR,
        int nzt,
        int xls, int xre, int yls, int yre, cudaStream_t St);

void drprecpc_app_H(float *xx, float *yy, float *zz, float *xy, float *xz, float *yz,
        float *mu,
        float *sigma2,
        float *yldfac,
        int nzt, int xls, int xre, int yls, int yre, cudaStream_t St);

void addsrc_H(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,  cudaStream_t St,
              float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
              float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz);

void calcRecordingPoints(int *rec_nbgx, int *rec_nedx,
  int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz,
  int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
  long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
  int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY,
  int NBGZ, int NEDZ, int NSKPZ, int *coord);

double gethrtime()
{
    struct timeval TV;
    int RC = gettimeofday(&TV,NULL);

    if (RC == -1){
       printf("Bad call to gettimeofday\n");
       return(-1);
    }

    return ( ((double)TV.tv_sec ) + micro * ((double)  TV.tv_usec));
}

int main(int argc,char **argv)
{
    CUCHK(cudaDeviceReset());
    CUCHK(cudaGetLastError());
//  variable definition begins
    float TMAX, DH, DT, ARBC, PHT;
    int   NPC, ND, NSRC, NST;
    int   NVE, NVAR, MEDIASTART, IFAULT, READ_STEP, READ_STEP_GPU;
    int   NX, NY, NZ, PX, PY, IDYNA, SoCalQ;
    int   NBGX, NEDX, NSKPX, NBGY, NEDY, NSKPY, NBGZ, NEDZ, NSKPZ;
    int   nxt, nyt, nzt;
    MPI_Offset displacement;
    float FAC, Q0, EX, FP;
    char  INSRC[250], INVEL[250], OUT[250], INSRC_I2[250], CHKFILE[250];
    double GFLOPS = 1.0;
    double GFLOPS_SUM = 0.0;
    Grid3D u1=NULL, v1=NULL, w1=NULL;
    Grid3D d1=NULL, mu=NULL, lam=NULL;
    Grid3D xx=NULL, yy=NULL, zz=NULL, xy=NULL, yz=NULL, xz=NULL;
    Grid3D r1=NULL, r2=NULL, r3=NULL, r4=NULL, r5=NULL, r6=NULL;
    Grid3D qp=NULL, qs=NULL;
    PosInf tpsrc=NULL;
    Grid1D taxx=NULL, tayy=NULL, tazz=NULL, taxz=NULL, tayz=NULL, taxy=NULL;
    Grid1D Bufx=NULL,coeff=NULL;
    Grid1D Bufy=NULL, Bufz=NULL;
    //Plasticity output buffers
    Grid1D Bufeta=NULL, Bufeta2=NULL;
    Grid3D vx1=NULL,   vx2=NULL,  wwo=NULL,  lam_mu=NULL;
    Grid3Dww ww=NULL;
    Grid1D dcrjx=NULL, dcrjy=NULL, dcrjz=NULL;
    float vse[2], vpe[2], dde[2];
    FILE *fchk;
    // plasticity variables
    Grid3D sigma2=NULL;
    Grid3D cohes=NULL, phi=NULL;
    Grid3D yldfac=NULL, neta=NULL;
//  GPU variables
    long int num_bytes;
    float* d_d1;
    float* d_u1;
    float* d_v1;
    float* d_w1;
    float* d_f_u1;
    float* d_f_v1;
    float* d_f_w1;
    float* d_b_u1;
    float* d_b_v1;
    float* d_b_w1;
    float* d_dcrjx;
    float* d_dcrjy;
    float* d_dcrjz;
    float* d_lam;
    float* d_mu;
    float* d_qp;
    float* d_coeff;
    float* d_qs;
    float* d_vx1;
    float* d_vx2;
    int* d_ww;
    float* d_wwo;
    float* d_xx;
    float* d_yy;
    float* d_zz;
    float* d_xy;
    float* d_xz;
    float* d_yz;
    float* d_r1;
    float* d_r2;
    float* d_r3;
    float* d_r4;
    float* d_r5;
    float* d_r6;
    float* d_lam_mu;
    int*   d_tpsrc;
    float* d_taxx;
    float* d_tayy;
    float* d_tazz;
    float* d_taxz;
    float* d_tayz;
    float* d_taxy;
    // plasticity
    float *d_sigma2;
    float *d_yldfac,*d_cohes, *d_phi, *d_neta;
    float *d_yldfac_L, *d_yldfac_R, *d_yldfac_F, *d_yldfac_B;
    float *d_yldfac_FL, *d_yldfac_FR, *d_yldfac_BL, *d_yldfac_BR;

//  end of GPU variables
    int i,j,k,idx,idy,idz;
    long int idtmp;
    long int tmpInd;
    const int maxdim = 3;
    float taumax, taumin, tauu;
    Grid3D tau=NULL, tau1=NULL, tau2=NULL;
    Grid3D weights=NULL;
    int npsrc;
    long int nt, cur_step, source_step;
    double time_un = 0.0;
    // time_src and time_mesh measures the time spent
    // in source and mesh reading
    double time_src = 0.0, time_src_tmp = 0.0, time_mesh = 0.0;
    // time_fileio and time_gpuio measures the time spent
    // in file system IO and gpu memory copying for IO
    double time_fileio = 0.0, time_gpuio = 0.0;
    double time_fileio_tmp = 0.0, time_gpuio_tmp = 0.0;
//  MPI+CUDA variables
    cudaError_t cerr;
    size_t cmemfree, cmemtotal;
    cudaStream_t stream_1, stream_2, stream_i;
    int   rank, size, err, srcproc, rank_gpu;
    int   dim[2], period[2], coord[2], reorder;
    //int   fmtype[3], fptype[3], foffset[3];
    int   x_rank_L  = -1,  x_rank_R  = -1,  y_rank_F = -1,  y_rank_B = -1;
    MPI_Comm MCW, MC1;
    MPI_Request  request_x[4], request_y[4];
    MPI_Status   status_x[4],  status_y[4], filestatus;
    MPI_Datatype filetype;
    MPI_File fh;
    int   msg_v_size_x, msg_v_size_y, count_x = 0, count_y = 0;
    int   xls, xre, xvs, xve, xss1, xse1, xss2, xse2, xss3, xse3;
    int   yfs, yfe, ybs, ybe, yls,  yre;
    /* Added by Daniel for plasticity computation boundaries */
    int  xlsp, xrep, ylsp, yrep;
    float* SL_vel;     // Velocity to be sent to   Left  in x direction (u1,v1,w1)
    float* SR_vel;     // Velocity to be Sent to   Right in x direction (u1,v1,w1)
    float* RL_vel;     // Velocity to be Recv from Left  in x direction (u1,v1,w1)
    float* RR_vel;     // Velocity to be Recv from Right in x direction (u1,v1,w1)
    float* SF_vel;     // Velocity to be sent to   Front in y direction (u1,v1,w1)
    float* SB_vel;     // Velocity to be Sent to   Back  in y direction (u1,v1,w1)
    float* RF_vel;     // Velocity to be Recv from Front in y direction (u1,v1,w1)
    float* RB_vel;     // Velocity to be Recv from Back  in y direction (u1,v1,w1)
    // plasticity definitions
    int   rank_FL   = -1,  rank_FR   = -1,  rank_BL  = -1,  rank_BR  = -1;
    MPI_Request request_yldfac[8];
    MPI_Status  status_yldfac[8];
    float *SL_yldfac,   *SR_yldfac,   *SF_yldfac,   *SB_yldfac;
    float *RL_yldfac,   *RR_yldfac,   *RF_yldfac,   *RB_yldfac;
    float *SFL_yldfac,  *SFR_yldfac,  *SBL_yldfac,  *SBR_yldfac;
    float *RFL_yldfac,  *RFR_yldfac,  *RBL_yldfac,  *RBR_yldfac;
    int   msg_yldfac_size_x, msg_yldfac_size_y, count_yldfac=0;
//  variable definition ends

    int tmpSize;
    int WRITE_STEP;
    int NTISKP;
    int rec_NX;
    int rec_NY;
    int rec_NZ;
    int rec_nxt;
    int rec_nyt;
    int rec_nzt;
    int rec_nbgx;   // 0-based indexing, however NBG* is 1-based
    int rec_nedx;   // 0-based indexing, however NED* is 1-based
    int rec_nbgy;   // 0-based indexing
    int rec_nedy;   // 0-based indexing
    int rec_nbgz;   // 0-based indexing
    int rec_nedz;   // 0-based indexing
    char filename[50];
    char filenamebasex[50];
    char filenamebasey[50];
    char filenamebasez[50];
    char filenamebaseeta[50];
    char filenamebaseep[50];

    // moving initial stress computation to GPU
    float fmajor=0, fminor=0, strike[3], dip[3], Rz[9], RzT[9];

//  variable initialization begins
    command(argc,argv,&TMAX,&DH,&DT,&ARBC,&PHT,&NPC,&ND,&NSRC,&NST,
      &NVAR,&NVE,&MEDIASTART,&IFAULT,&READ_STEP,&READ_STEP_GPU,
      &NTISKP,&WRITE_STEP,&NX,&NY,&NZ,&PX,&PY,
      &NBGX,&NEDX,&NSKPX,&NBGY,&NEDY,&NSKPY,&NBGZ,&NEDZ,&NSKPZ,
        &FAC,&Q0,&EX,&FP,&IDYNA,&SoCalQ,INSRC,INVEL,OUT,INSRC_I2,CHKFILE);

    sprintf(filenamebasex,"%s/SX",OUT);
    sprintf(filenamebasey,"%s/SY",OUT);
    sprintf(filenamebasez,"%s/SZ",OUT);
    sprintf(filenamebaseeta,"%s/Eta",OUT);
    sprintf(filenamebaseep,"%s/EP",OUT);

    //printf("After command.\n");
    // Below 12 lines are NOT for HPGPU4 machine!
/*    int local_rank;
    char* str;
    if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK")) != NULL) {
      local_rank = atoi(str);
      rank_gpu = local_rank%3;
    }
    else{
      printf("CANNOT READ LOCAL RANK!\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    //printf("%d) After rank_gpu calc.\n",local_rank);
    cudaSetDevice(rank_gpu);
    //if(local_rank==0) printf("after cudaSetDevice\n");
*/
    // WARNING: Above 12 lines are not for HPGPU4 machine!
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_dup(MPI_COMM_WORLD, &MCW );
    MPI_Barrier(MCW);
    nxt       = NX/PX;
    nyt       = NY/PY;
    nzt       = NZ;
    nt        = (int)(TMAX/DT) + 1;
    dim[0]    = PX;
    dim[1]    = PY;
    period[0] = 0;
    period[1] = 0;
    reorder   = 1;
    err       = MPI_Cart_create(MCW, 2, dim, period, reorder, &MC1);
    err       = MPI_Cart_shift(MC1, 0,  1,  &x_rank_L, &x_rank_R );
    err       = MPI_Cart_shift(MC1, 1,  1,  &y_rank_F, &y_rank_B );
    err       = MPI_Cart_coords(MC1, rank, 2, coord);
    err       = MPI_Barrier(MCW);
    // Below line is only for HPGPU4 machine!
//    rank_gpu = rank%4;
    // Below line is for 1 GPU/node systems
    rank_gpu = 0;
    cudaSetDevice(rank_gpu);

printf("\n\nrank=%d) RS=%d, RSG=%d, NST=%d, IF=%d\n\n\n",
rank, READ_STEP, READ_STEP_GPU, NST, IFAULT);

    // ranks of corner processors
    if(y_rank_F>=0){
      if(x_rank_L>=0) rank_FL = y_rank_F - (rank-x_rank_L);
      if(x_rank_R>=0) rank_FR = y_rank_F - (rank-x_rank_R);
    }
    if(y_rank_B>=0){
      if(x_rank_L>=0) rank_BL = y_rank_B - (rank-x_rank_L);
      if(x_rank_R>=0) rank_BR = y_rank_B - (rank-x_rank_R);
    }
    // same for each processor:
    if(NEDX==-1) NEDX = NX;
    if(NEDY==-1) NEDY = NY;
    if(NEDZ==-1) NEDZ = NZ;
    // make NED's a record point
    // for instance if NBGX:NSKPX:NEDX = 1:3:9
    // then we have 1,4,7 but NEDX=7 is better
    NEDX = NEDX-(NEDX-NBGX)%NSKPX;
    NEDY = NEDY-(NEDY-NBGY)%NSKPY;
    NEDZ = NEDZ-(NEDZ-NBGZ)%NSKPZ;
    // number of recording points in total
    rec_NX = (NEDX-NBGX)/NSKPX+1;
    rec_NY = (NEDY-NBGY)/NSKPY+1;
    rec_NZ = (NEDZ-NBGZ)/NSKPZ+1;

    // specific to each processor:
    calcRecordingPoints(&rec_nbgx, &rec_nedx, &rec_nbgy, &rec_nedy,
      &rec_nbgz, &rec_nedz, &rec_nxt, &rec_nyt, &rec_nzt, &displacement,
      (long int)nxt,(long int)nyt,(long int)nzt, rec_NX, rec_NY, rec_NZ,
      NBGX,NEDX,NSKPX, NBGY,NEDY,NSKPY, NBGZ,NEDZ,NSKPZ, coord);
    printf("%d = (%d,%d)) NX,NY,NZ=%d,%d,%d\nnxt,nyt,nzt=%d,%d,%d\nrec_N=(%d,%d,%d)\nrec_nxt,=%d,%d,%d\nNBGX,SKP,END=(%d:%d:%d),(%d:%d:%d),(%d:%d:%d)\nrec_nbg,ed=(%d,%d),(%d,%d),(%d,%d)\ndisp=%ld\n",
        rank,coord[0],coord[1],NX,NY,NZ,nxt,nyt,nzt,
        rec_NX, rec_NY, rec_NZ, rec_nxt, rec_nyt, rec_nzt,
        NBGX,NSKPX,NEDX,NBGY,NSKPY,NEDY,NBGZ,NSKPZ,NEDZ,
        rec_nbgx,rec_nedx,rec_nbgy,rec_nedy,rec_nbgz,rec_nedz,(long int)displacement);

    int maxNX_NY_NZ_WS = (rec_NX>rec_NY?rec_NX:rec_NY);
    maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>rec_NZ?maxNX_NY_NZ_WS:rec_NZ);
    maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>WRITE_STEP?maxNX_NY_NZ_WS:WRITE_STEP);
    int ones[maxNX_NY_NZ_WS];
    MPI_Aint dispArray[maxNX_NY_NZ_WS];
    for(i=0;i<maxNX_NY_NZ_WS;++i){
      ones[i] = 1;
    }

    err = MPI_Type_contiguous(rec_nxt, MPI_FLOAT, &filetype);
    err = MPI_Type_commit(&filetype);
    for(i=0;i<rec_nyt;i++){
      dispArray[i] = sizeof(float);
      dispArray[i] = dispArray[i]*rec_NX*i;
    }
    err = MPI_Type_create_hindexed(rec_nyt, ones, dispArray, filetype, &filetype);
    err = MPI_Type_commit(&filetype);
    for(i=0;i<rec_nzt;i++){
      dispArray[i] = sizeof(float);
      dispArray[i] = dispArray[i]*rec_NY*rec_NX*i;
    }
    err = MPI_Type_create_hindexed(rec_nzt, ones, dispArray, filetype, &filetype);
    err = MPI_Type_commit(&filetype);
    for(i=0;i<WRITE_STEP;i++){
      dispArray[i] = sizeof(float);
      dispArray[i] = dispArray[i]*rec_NZ*rec_NY*rec_NX*i;
    }
    err = MPI_Type_create_hindexed(WRITE_STEP, ones, dispArray, filetype, &filetype);
    //err = MPI_Type_contiguous(WRITE_STEP, filetype, &filetype);
    err = MPI_Type_commit(&filetype);
    MPI_Type_size(filetype, &tmpSize);
    if(rank==0) printf("filetype size (supposedly=rec_nxt*nyt*nzt*WS*4=%d) =%d\n", rec_nxt*rec_nyt*rec_nzt*WRITE_STEP*4,tmpSize);

/*
    fmtype[0]  = WRITE_STEP;
    //fmtype[1]  = NZ;
    fmtype[1]  = NY;
    fmtype[2]  = NX;
    fptype[0]  = WRITE_STEP;
    //fptype[1]  = nzt;
    fptype[1]  = nyt;
    fptype[2]  = nxt;
    foffset[0] = 0;
    //foffset[1] = 0;
    foffset[1] = nyt*coord[1];
    foffset[2] = nxt*coord[0];
    err = MPI_Type_create_subarray(3, fmtype, fptype, foffset, MPI_ORDER_C, MPI_FLOAT, &filetype);
    err = MPI_Type_commit(&filetype);
*/
    printf("rank=%d, x_rank_L=%d, x_rank_R=%d, y_rank_F=%d, y_rank_B=%d, rank_FL=%d, rank_FR=%d, rank_BL=%d, rank_BR=%d\n",
        rank, x_rank_L, x_rank_R, y_rank_F, y_rank_B,
        rank_FL, rank_FR, rank_BL, rank_BR);

    if(x_rank_L<0) {
       xls = 2+ngsl;
       xlsp = xls;
    }
    else {
       xls = 4;
       xlsp = xls -1;
    }

    if(x_rank_R<0) {
       xre = nxt+ngsl+1;
       xrep = xre;
    }
    else {
       xre = nxt + ngsl2 - 1;
       xrep = xre + 1;
    }

    xvs   = 2+ngsl;
    xve   = nxt+ngsl+1;

    xss1  = xls;
    xse1  = ngsl+3;
    xss2  = ngsl+4;
    xse2  = nxt+ngsl-1;
    xss3  = nxt+ngsl;
    xse3  = xre;

    if(y_rank_F<0) {
       yls = 2+ngsl;
       ylsp = yls;
    }
    else {
       yls = 4;
       ylsp = yls -1;
    }

    if(y_rank_B<0) {
       yre = nyt+ngsl+1;
       yrep = yre;
    }
    else {
       yre = nyt + ngsl2 - 1;
       yrep = yre + 1;
    }

    yfs  = 2+ngsl;
    yfe  = 2+ngsl2-1;
    ybs  = nyt+2;
    ybe  = nyt+ngsl+1;

    time_src -= gethrtime();
    if(rank==0) printf("Before inisource\n");
    err = inisource(rank,   IFAULT, NSRC,  READ_STEP, NST,   &srcproc, NZ, MCW, nxt, nyt, nzt, coord, maxdim, &npsrc,
                    &tpsrc, &taxx,  &tayy, &tazz,     &taxz, &tayz,    &taxy, INSRC, INSRC_I2);
    if(err)
    {
       printf("source initialization failed\n");
       return -1;
    }
    time_src += gethrtime();
    if(rank==0) printf("After inisource. Time elapsed (seconds): %lf\n", time_src);
    if(rank==srcproc)
    {
       printf("rank=%d, source rank, npsrc=%d\n", rank, npsrc);
       num_bytes = sizeof(float)*npsrc*READ_STEP_GPU;
       CUCHK(cudaMalloc((void**)&d_taxx, num_bytes));
       CUCHK(cudaMalloc((void**)&d_tayy, num_bytes));
       CUCHK(cudaMalloc((void**)&d_tazz, num_bytes));
       CUCHK(cudaMalloc((void**)&d_taxz, num_bytes));
       CUCHK(cudaMalloc((void**)&d_tayz, num_bytes));
       CUCHK(cudaMalloc((void**)&d_taxy, num_bytes));
       CUCHK(cudaMemcpy(d_taxx,taxx,num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMemcpy(d_tayy,tayy,num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMemcpy(d_tazz,tazz,num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMemcpy(d_taxz,taxz,num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMemcpy(d_tayz,tayz,num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMemcpy(d_taxy,taxy,num_bytes,cudaMemcpyHostToDevice));
       num_bytes = sizeof(int)*npsrc*maxdim;
       CUCHK(cudaMalloc((void**)&d_tpsrc, num_bytes));
       CUCHK(cudaMemcpy(d_tpsrc,tpsrc,num_bytes,cudaMemcpyHostToDevice));
    }

    d1     = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    mu     = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    lam    = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    lam_mu = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, 1);

    if(NVE==1 || NVE==3)
    {
       qp   = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
       qs   = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
       tau  = Alloc3D(2, 2, 2);
       tau1 = Alloc3D(2, 2, 2);
       tau2 = Alloc3D(2, 2, 2);
       weights = Alloc3D(2, 2, 2);
       coeff = Alloc1D(16);
       weights_sub(weights,coeff, EX, FAC);
    }
    time_mesh -= gethrtime();
    if(NVE==3){
       sigma2  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
       cohes  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
       phi    = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);

       yldfac = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
       neta   = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);

      // initialize
      for(i=0;i<nxt+4+ngsl2;i++)
        for(j=0;j<nyt+4+ngsl2;j++)
          for(k=0;k<nzt+2*align;k++){
            neta[i][j][k] = 0.;
            yldfac[i][j][k] = 1.;
          }

       // msg size (#floats) to send in X direction is nyt
       msg_yldfac_size_x = nyt*nzt;
       msg_yldfac_size_y = nxt*nzt;

       cudaMallocHost((void**)&SL_yldfac, msg_yldfac_size_x*sizeof(float));
       cudaMallocHost((void**)&SR_yldfac, msg_yldfac_size_x*sizeof(float));
       cudaMallocHost((void**)&RL_yldfac, msg_yldfac_size_x*sizeof(float));
       cudaMallocHost((void**)&RR_yldfac, msg_yldfac_size_x*sizeof(float));
       cudaMallocHost((void**)&SF_yldfac, msg_yldfac_size_y*sizeof(float));
       cudaMallocHost((void**)&SB_yldfac, msg_yldfac_size_y*sizeof(float));
       cudaMallocHost((void**)&RF_yldfac, msg_yldfac_size_y*sizeof(float));
       cudaMallocHost((void**)&RB_yldfac, msg_yldfac_size_y*sizeof(float));
       cudaMallocHost((void**)&SFL_yldfac, sizeof(float));
       cudaMallocHost((void**)&SFR_yldfac, sizeof(float));
       cudaMallocHost((void**)&SBL_yldfac, sizeof(float));
       cudaMallocHost((void**)&SBR_yldfac, sizeof(float));
       cudaMallocHost((void**)&RFL_yldfac, sizeof(float));
       cudaMallocHost((void**)&RFR_yldfac, sizeof(float));
       cudaMallocHost((void**)&RBL_yldfac, sizeof(float));
       CUCHK(cudaMallocHost((void**)&RBR_yldfac, sizeof(float)));
    }

    if(rank==0) printf("Before inimesh\n");
    inimesh(rank, MEDIASTART, d1, mu, lam, qp, qs, &taumax, &taumin, tau, weights,coeff, NVAR, FP, FAC, Q0, EX,
            nxt, nyt, nzt, PX, PY, NX, NY, NZ, coord, MCW, IDYNA, NVE, SoCalQ, INVEL,
            vse, vpe, dde);
    time_mesh += gethrtime();
    if(rank==0) printf("After inimesh. Time elapsed (seconds): %lf\n", time_mesh);
    if(rank==0) {
      writeCHK(CHKFILE, NTISKP, DT, DH, nxt, nyt, nzt,
           nt, ARBC, NPC, NVE, FAC, Q0, EX, FP, vse, vpe, dde);
    }
    mediaswap(d1, mu, lam, qp, qs, rank, x_rank_L, x_rank_R, y_rank_F, y_rank_B, nxt, nyt, nzt, MCW);

    for(i=xls;i<xre+1;i++)
      for(j=yls;j<yre+1;j++)
      {
         float t_xl, t_xl2m;
         t_xl             = 1.0/lam[i][j][nzt+align-1];
         t_xl2m           = 2.0/mu[i][j][nzt+align-1] + t_xl;
         lam_mu[i][j][0]  = t_xl/t_xl2m;
      }

    if(NVE==3){
      printf("%d) Computing initial stress\n",rank);
      inidrpr_hoekbrown_light(nxt, nyt, nzt, NVE, coord, DH, rank, mu, lam, d1,
          sigma2, cohes, phi, &fmajor, &fminor, strike, dip, MCW);
      rotation_matrix(strike, dip, Rz, RzT);
    }

    num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2);
    cudaMalloc((void**)&d_lam_mu, num_bytes);
    CUCHK(cudaMemcpy(d_lam_mu,&lam_mu[0][0][0],num_bytes,cudaMemcpyHostToDevice));

    vx1  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    vx2  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    ww   = Alloc3Dww(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    wwo   = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    if(NPC==0)
    {
    dcrjx = Alloc1D(nxt+4+ngsl2);
        dcrjy = Alloc1D(nyt+4+ngsl2);
        dcrjz = Alloc1D(nzt+2*align);

        for(i=0;i<nxt+4+ngsl2;i++)
       dcrjx[i]  = 1.0;
        for(j=0;j<nyt+4+ngsl2;j++)
           dcrjy[j]  = 1.0;
        for(k=0;k<nzt+2*align;k++)
           dcrjz[k]  = 1.0;

        inicrj(ARBC, coord, nxt, nyt, nzt, NX, NY, ND, dcrjx, dcrjy, dcrjz);
    }

    if(NVE==1 || NVE==3)
    {
        float dt1 = 1.0/DT;
        for(i=0;i<2;i++)
          for(j=0;j<2;j++)
            for(k=0;k<2;k++)
            {
               tauu          = tau[i][j][k];
           tau2[i][j][k] = exp(-DT/tauu);
           tau1[i][j][k] = 0.5*(1.-tau2[i][j][k]);
            }

    init_texture(nxt, nyt, nzt, tau1, tau2, vx1, vx2, weights, ww,wwo, xls, xre, yls, yre);

        Delloc3D(tau);
        Delloc3D(tau1);
        Delloc3D(tau2);
    }

    if(rank==0) printf("Allocate device media pointers and copy.\n");
    num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
    cudaMalloc((void**)&d_d1, num_bytes);
    cudaMemcpy(d_d1,&d1[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_lam, num_bytes);
    cudaMemcpy(d_lam,&lam[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_mu, num_bytes);
    cudaMemcpy(d_mu,&mu[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    CUCHK(cudaMalloc((void**)&d_qp, num_bytes));
    cudaMemcpy(d_qp,&qp[0][0][0],num_bytes,cudaMemcpyHostToDevice);

    num_bytes = sizeof(float)*(16);
    cudaMalloc((void**)&d_coeff, num_bytes);
    cudaMemcpy(d_coeff,&coeff[0],num_bytes,cudaMemcpyHostToDevice);
    num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);


    cudaMalloc((void**)&d_qs, num_bytes);
    cudaMemcpy(d_qs,&qs[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_vx1, num_bytes);
    cudaMemcpy(d_vx1,&vx1[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_vx2, num_bytes);
    cudaMemcpy(d_vx2,&vx2[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    num_bytes = sizeof(int)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
    cudaMalloc((void**)&d_ww, num_bytes);
    cudaMemcpy(d_ww,&ww[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
    cudaMalloc((void**)&d_wwo, num_bytes);
    CUCHK(cudaMemcpy(d_wwo,&wwo[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    BindArrayToTexture(d_vx1, d_vx2, d_ww,d_wwo, num_bytes);
    //    printf("ww,wwo %i %f\n",d_ww,d_wwo);



    if(NPC==0)
    {
        num_bytes = sizeof(float)*(nxt+4+ngsl2);
        cudaMalloc((void**)&d_dcrjx, num_bytes);
        cudaMemcpy(d_dcrjx,dcrjx,num_bytes,cudaMemcpyHostToDevice);
        num_bytes = sizeof(float)*(nyt+4+ngsl2);
        cudaMalloc((void**)&d_dcrjy, num_bytes);
        cudaMemcpy(d_dcrjy,dcrjy,num_bytes,cudaMemcpyHostToDevice);
        num_bytes = sizeof(float)*(nzt+2*align);
        cudaMalloc((void**)&d_dcrjz, num_bytes);
        cudaMemcpy(d_dcrjz,dcrjz,num_bytes,cudaMemcpyHostToDevice);
    }

    if(rank==0) printf("Allocate host velocity and stress pointers.\n");
    u1  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    v1  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    w1  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    xx  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    yy  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    zz  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    xy  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    yz  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    xz  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    if(NVE==1 || NVE==3)
    {
        r1  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
        r2  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
        r3  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
        r4  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
        r5  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
        r6  = Alloc3D(nxt+4+ngsl2, nyt+4+ngsl2, nzt+2*align);
    }

    source_step = 1;
    if(rank==srcproc)
    {
       addsrc(source_step, DH, DT, NST, npsrc, READ_STEP, maxdim, tpsrc, taxx, tayy, tazz, taxz, tayz, taxy, xx, yy, zz, xy, yz, xz);
    }

    if(rank==0) printf("Allocate device velocity and stress pointers and copy.\n");
    num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
    fprintf(stderr, "%d) Trying to alloc %lu MB (dev vel abd stress)\n", rank, num_bytes/1024/1024);
    CUCHK(cudaMalloc((void**)&d_u1, num_bytes)                          );
    CUCHK(cudaMemcpy(d_u1,&u1[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_v1, num_bytes)                          );
    CUCHK(cudaMemcpy(d_v1,&v1[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_w1, num_bytes)                          );
    CUCHK(cudaMemcpy(d_w1,&w1[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_xx, num_bytes)                          );
    CUCHK(cudaMemcpy(d_xx,&xx[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_yy, num_bytes)                          );
    CUCHK(cudaMemcpy(d_yy,&yy[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_zz, num_bytes)                          );
    CUCHK(cudaMemcpy(d_zz,&zz[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_xy, num_bytes)                          );
    CUCHK(cudaMemcpy(d_xy,&xy[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_xz, num_bytes                          ));
    CUCHK(cudaMemcpy(d_xz,&xz[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    CUCHK(cudaMalloc((void**)&d_yz, num_bytes)                          );
    CUCHK(cudaMemcpy(d_yz,&yz[0][0][0],num_bytes,cudaMemcpyHostToDevice));
    printf("%d) Done \n", rank);
    if(NVE==1 || NVE==3)
    {
      if(rank==0) fprintf(stderr, "Allocate additional device pointers (r) and copy.\n");
        cudaMalloc((void**)&d_r1, num_bytes);
        cudaMemcpy(d_r1,&r1[0][0][0],num_bytes,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_r2, num_bytes);
        cudaMemcpy(d_r2,&r2[0][0][0],num_bytes,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_r3, num_bytes);
        cudaMemcpy(d_r3,&r3[0][0][0],num_bytes,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_r4, num_bytes);
        cudaMemcpy(d_r4,&r4[0][0][0],num_bytes,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_r5, num_bytes);
        cudaMemcpy(d_r5,&r5[0][0][0],num_bytes,cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_r6, num_bytes);
        cudaMemcpy(d_r6,&r6[0][0][0],num_bytes,cudaMemcpyHostToDevice);
    }
    if(NVE==3){
      if(rank==0) printf("Allocate plasticity variables since NVE=3\n");
      num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
      fprintf(stderr, "%d) Trying to alloc %d MB (plas)\n", rank, num_bytes/1024/1024);
      CUCHK(cudaMalloc((void**)&d_sigma2, num_bytes)                               );
      CUCHK(cudaMemcpy(d_sigma2,&sigma2[0][0][0],num_bytes,cudaMemcpyHostToDevice)  );
      CUCHK(cudaMalloc((void**)&d_yldfac, num_bytes)                              );
      CUCHK(cudaMemcpy(d_yldfac,&yldfac[0][0][0],num_bytes,cudaMemcpyHostToDevice));
      /*CUCHK(cudaMalloc((void**)&d_cohes, num_bytes)                               );
      CUCHK(cudaMemcpy(d_cohes,&cohes[0][0][0],num_bytes,cudaMemcpyHostToDevice)  );
      CUCHK(cudaMalloc((void**)&d_phi, num_bytes)                                 );
      CUCHK(cudaMemcpy(d_phi,&phi[0][0][0],num_bytes,cudaMemcpyHostToDevice)      );*/
      CUCHK(cudaMalloc((void**)&d_neta, num_bytes)                                );
      CUCHK(cudaMemcpy(d_neta,&neta[0][0][0],num_bytes,cudaMemcpyHostToDevice)    );

      fprintf(stderr, "%d), msg_yldfac_size_x=%lu\n", rank, msg_yldfac_size_x);
      fprintf(stderr, "%d), msg_yldfac_size_y=%lu\n", rank, msg_yldfac_size_y);
      CUCHK(cudaMalloc((void**)&d_yldfac_L, msg_yldfac_size_x*sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_R, msg_yldfac_size_x*sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_F, msg_yldfac_size_y*sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_B, msg_yldfac_size_y*sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_FL, sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_FR, sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_BL, sizeof(float)));
      CUCHK(cudaMalloc((void**)&d_yldfac_BR, sizeof(float)));
    }
//  variable initialization ends
    if(rank==0) printf("Allocate buffers of #elements: %d\n",rec_nxt*rec_nyt*rec_nzt*WRITE_STEP);
    Bufx  = Alloc1D(rec_nxt*rec_nyt*rec_nzt*WRITE_STEP);
    Bufy  = Alloc1D(rec_nxt*rec_nyt*rec_nzt*WRITE_STEP);
    Bufz  = Alloc1D(rec_nxt*rec_nyt*rec_nzt*WRITE_STEP);
//  Allocate buffers for plasticity output
    if (NVE == 3){
       Bufeta   = Alloc1D(rec_nxt*rec_nyt*rec_nzt*WRITE_STEP);
    }
    num_bytes = sizeof(float)*3*(ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
    CUCHK(cudaMallocHost((void**)&SL_vel, num_bytes));
    CUCHK(cudaMallocHost((void**)&SR_vel, num_bytes));
    CUCHK(cudaMallocHost((void**)&RL_vel, num_bytes));
    CUCHK(cudaMallocHost((void**)&RR_vel, num_bytes));
    num_bytes = sizeof(float)*3*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
    CUCHK(cudaMallocHost((void**)&SF_vel, num_bytes));
    CUCHK(cudaMallocHost((void**)&SB_vel, num_bytes));
    CUCHK(cudaMallocHost((void**)&RF_vel, num_bytes));
    CUCHK(cudaMallocHost((void**)&RB_vel, num_bytes));
    num_bytes = sizeof(float)*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
    CUCHK(cudaMalloc((void**)&d_f_u1, num_bytes));
    CUCHK(cudaMalloc((void**)&d_f_v1, num_bytes));
    CUCHK(cudaMalloc((void**)&d_f_w1, num_bytes));
    CUCHK(cudaMalloc((void**)&d_b_u1, num_bytes));
    CUCHK(cudaMalloc((void**)&d_b_v1, num_bytes));
    CUCHK(cudaMalloc((void**)&d_b_w1, num_bytes));
    msg_v_size_x = 3*(ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
    msg_v_size_y = 3*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
    fprintf(stdout, "fmajor in main = %f\n", fmajor);
    SetDeviceConstValue(DH, DT, nxt, nyt, nzt, fmajor, fminor, Rz, RzT);
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);
    CUCHK(cudaStreamCreate(&stream_i));
    //Delloc3D(tau);  // FIXME: removed by Carl

    size_t  cmemfreeMin;
    cudaMemGetInfo(&cmemfree, &cmemtotal);
    if(sizeof(size_t)==8)
      MPI_Reduce(&cmemfree, &cmemfreeMin, 1, MPI_UINT64_T, MPI_MIN, 0, MCW);
    else
      MPI_Reduce(&cmemfree, &cmemfreeMin, 1, MPI_UINT32_T, MPI_MIN, 0, MCW);
    if(rank==0) printf("CUDA MEMORY: free = %ld\ttotal = %ld\n",cmemfreeMin,cmemtotal);


    if(rank==0){
      CUCHK(cudaMemGetInfo(&cmemfree, &cmemtotal));
      printf("CUDA MEMORY: Total=%ld\tAvailable=%ld\n",cmemtotal,cmemfree);
    }

    if(rank==0)
      fchk = fopen(CHKFILE,"a+");
//  Main Loop Starts
    if(NPC==0 && (NVE==1 || NVE==3))
    {
       time_un  -= gethrtime();
       //This loop has no loverlapping because there is source input
       //    for(cur_step=1;cur_step<=nt;cur_step++)
       for(cur_step=1;cur_step<=NST;cur_step++)
       {
         if(rank==0) printf("Time Step =                   %ld    OF  Total Timesteps = %ld\n", cur_step, nt);
         if(cur_step==100 && rank==0) printf("Time per timestep:\t%f seconds\n",(gethrtime()+time_un)/100);
         cerr = cudaGetLastError();
         if(cerr!=cudaSuccess) printf("CUDA ERROR! rank=%d before timestep: %s\n",rank,cudaGetErrorString(cerr));
         //pre-post MPI Message
         PostRecvMsg_Y(RF_vel, RB_vel, MCW, request_y, &count_y, msg_v_size_y, y_rank_F, y_rank_B);
         PostRecvMsg_X(RL_vel, RR_vel, MCW, request_x, &count_x, msg_v_size_x, x_rank_L, x_rank_R);
         /*if(NVE==3)
              PostRecvMsg_yldfac(RL_yldfac, RR_yldfac, RF_yldfac, RB_yldfac,
              RFL_yldfac, RFR_yldfac, RBL_yldfac, RBR_yldfac, MCW,
              request_yldfac, &count_yldfac, msg_yldfac_size_x, msg_yldfac_size_y,
              x_rank_L, x_rank_R, y_rank_F, y_rank_B,
              rank_FL, rank_FR, rank_BL, rank_BR);*/
         //velocity computation in y boundary, two ghost cell regions
         dvelcy_H(d_u1, d_v1, d_w1, d_xx,   d_yy,   d_zz,   d_xy,       d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz,
                  d_d1, nxt,  nzt,  d_f_u1, d_f_v1, d_f_w1, stream_i,   yfs,  yfe, y_rank_F);
         dvelcy_H(d_u1, d_v1, d_w1, d_xx,   d_yy,   d_zz,   d_xy,       d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz,
                  d_d1, nxt,  nzt,  d_b_u1, d_b_v1, d_b_w1, stream_i,   ybs,  ybe, y_rank_B);
         Cpy2Host_VY(d_f_u1, d_f_v1, d_f_w1,  SF_vel, nxt, nzt, stream_i, y_rank_F);
         Cpy2Host_VY(d_b_u1, d_b_v1, d_b_w1,  SB_vel, nxt, nzt, stream_i, y_rank_B);
         cudaThreadSynchronize();
         //velocity communication in y direction
         PostSendMsg_Y(SF_vel, SB_vel, MCW, request_y, &count_y, msg_v_size_y, y_rank_F, y_rank_B, rank, Both);
         MPI_Waitall(count_y, request_y, status_y);
         Cpy2Device_VY(d_u1,     d_v1,     d_w1,     d_f_u1, d_f_v1, d_f_w1, d_b_u1, d_b_v1, d_b_w1, RF_vel, RB_vel, nxt, nyt, nzt,
                       stream_i, stream_i, y_rank_F, y_rank_B);
         //velocity computation whole 3D Grid (nxt, nyt, nzt)
         dvelcx_H_opt(d_u1, d_v1, d_w1, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz, d_d1, nyt,  nzt,  stream_i,   xvs,  xve);
         Cpy2Host_VX(d_u1, d_v1, d_w1, SL_vel, nxt, nyt, nzt, stream_i, x_rank_L, Left);
         Cpy2Host_VX(d_u1, d_v1, d_w1, SR_vel, nxt, nyt, nzt, stream_i, x_rank_R, Right);
         cudaThreadSynchronize();
         //velocity communication in x direction
         PostSendMsg_X(SL_vel, SR_vel, MCW, request_x, &count_x, msg_v_size_x, x_rank_L, x_rank_R, rank, Both);
         MPI_Waitall(count_x, request_x, status_x);
         Cpy2Device_VX(d_u1, d_v1, d_w1, RL_vel, RR_vel, nxt, nyt, nzt, stream_i, stream_i, x_rank_L, x_rank_R);
         //stress computation whole 3D Grid (nxt+8, nyt+8, nzt)
         dstrqc_H_new(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz,
                  d_r1, d_r2, d_r3, d_r4, d_r5, d_r6,
                  d_u1, d_v1, d_w1, d_lam,
                  d_mu, d_qp,d_coeff, d_qs, d_dcrjx, d_dcrjy, d_dcrjz,
                  nyt,  nzt,  stream_i, d_lam_mu,
                  d_vx1, d_vx2, d_ww, d_wwo,
                  NX,   coord[0], coord[1],   xls,  xre,
                  yls,  yre);
         // plasticity related calls:
         if(NVE==3){
           cudaDeviceSynchronize();
           drprecpc_calc_H_opt(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu, d_d1,
              d_sigma2,
              d_yldfac,d_cohes, d_phi,
              d_neta,
              d_yldfac_L, d_yldfac_R, d_yldfac_F, d_yldfac_B,
              d_yldfac_FL, d_yldfac_FR, d_yldfac_BL, d_yldfac_BR,
              nzt, xlsp, xrep, ylsp, yrep, stream_i);
           cudaDeviceSynchronize();
           drprecpc_app_H(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu,
              d_sigma2,
              d_yldfac,
              nzt, xlsp, xrep, ylsp, yrep, stream_i);
           /*Cpy2Host_yldfac(d_yldfac_L, d_yldfac_R, d_yldfac_F, d_yldfac_B,
              d_yldfac_FL, d_yldfac_FR, d_yldfac_BL, d_yldfac_BR,
              SL_yldfac, SR_yldfac, SF_yldfac, SB_yldfac,
              SFL_yldfac, SFR_yldfac, SBL_yldfac, SBR_yldfac,
              stream_i, x_rank_L, x_rank_R, y_rank_F, y_rank_B,
              nxt, nyt);
           cudaThreadSynchronize();
           PostSendMsg_yldfac(SL_yldfac, SR_yldfac, SF_yldfac, SB_yldfac,
              SFL_yldfac, SFR_yldfac, SBL_yldfac, SBR_yldfac,
              MCW, request_yldfac, &count_yldfac,
              msg_yldfac_size_x, msg_yldfac_size_y, rank,
              x_rank_L, x_rank_R, y_rank_F, y_rank_B,
              rank_FL, rank_FR, rank_BL, rank_BR);
           MPI_Waitall(count_yldfac, request_yldfac, status_yldfac);
           Cpy2Device_yldfac(d_yldfac,
              d_yldfac_L, d_yldfac_R, d_yldfac_F, d_yldfac_B,
              d_yldfac_FL, d_yldfac_FR, d_yldfac_BL, d_yldfac_BR,
              RL_yldfac, RR_yldfac, RF_yldfac, RB_yldfac,
              RFL_yldfac, RFR_yldfac, RBL_yldfac, RBR_yldfac,
              stream_i, x_rank_L, x_rank_R, y_rank_F, y_rank_B,
              nxt, nyt, nzt);*/
         }

         //update source input
         if(rank==srcproc && cur_step<NST)
         {
            ++source_step;
            addsrc_H(source_step, READ_STEP_GPU, maxdim, d_tpsrc, npsrc, stream_i, d_taxx, d_tayy, d_tazz, d_taxz, d_tayz, d_taxy,
                     d_xx,       d_yy,      d_zz,   d_xy,    d_yz,  d_xz);
         }
         cudaThreadSynchronize();

         //apply free surface boundary conditions (Daniel)
         cudaDeviceSynchronize();
         fstr_H(d_zz, d_xz, d_yz, stream_i, xls, xre, yls, yre);

         if(cur_step%NTISKP == 0){
          num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
          if(!rank) time_gpuio_tmp = -gethrtime();
          cudaMemcpy(&u1[0][0][0],d_u1,num_bytes,cudaMemcpyDeviceToHost);
          cudaMemcpy(&v1[0][0][0],d_v1,num_bytes,cudaMemcpyDeviceToHost);
          cudaMemcpy(&w1[0][0][0],d_w1,num_bytes,cudaMemcpyDeviceToHost);
          //added for plasticity
          if (NVE == 3)
             cudaMemcpy(&neta[0][0][0],d_neta,num_bytes,cudaMemcpyDeviceToHost);
          idtmp = ((cur_step/NTISKP+WRITE_STEP-1)%WRITE_STEP);
          idtmp = idtmp*rec_nxt*rec_nyt*rec_nzt;
          tmpInd = idtmp;
          //if(rank==0) printf("idtmp=%ld\n", idtmp);
          // surface: k=nzt+align-1;
          for(k=nzt+align-1 - rec_nbgz; k>=nzt+align-1 - rec_nedz; k=k-NSKPZ)
            for(j=2+ngsl + rec_nbgy; j<=2+ngsl + rec_nedy; j=j+NSKPY)
              for(i=2+ngsl + rec_nbgx; i<=2+ngsl + rec_nedx; i=i+NSKPX)
              {
                //idx = (i-2-ngsl)/NSKPX;
                //idy = (j-2-ngsl)/NSKPY;
                //idz = ((nzt+align-1) - k)/NSKPZ;
                //tmpInd = idtmp + idz*rec_nxt*rec_nyt + idy*rec_nxt + idx;
                //if(rank==0) printf("%ld:%d,%d,%d\t",tmpInd,i,j,k);
                Bufx[tmpInd] = u1[i][j][k];
                Bufy[tmpInd] = v1[i][j][k];
                Bufz[tmpInd] = w1[i][j][k];
                if (NVE == 3) {
                   Bufeta[tmpInd] = neta[i][j][k];
                }
                tmpInd++;
              }
          if((cur_step/NTISKP)%WRITE_STEP == 0){
            cudaThreadSynchronize();
            sprintf(filename, "%s%07ld", filenamebasex, cur_step);
            err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
            err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
            err = MPI_File_write_all(fh, Bufx, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
            err = MPI_File_close(&fh);
            sprintf(filename, "%s%07ld", filenamebasey, cur_step);
            err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
            err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
            err = MPI_File_write_all(fh, Bufy, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
            err = MPI_File_close(&fh);
            sprintf(filename, "%s%07ld", filenamebasez, cur_step);
            err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
            err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
            err = MPI_File_write_all(fh, Bufz, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
            err = MPI_File_close(&fh);
            //saves the plastic shear work
            if (NVE == 3) {
               sprintf(filename, "%s%07ld", filenamebaseeta, cur_step);
               fprintf(stderr, "Saving shear work %s\n", filename);
               err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
               err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
               err = MPI_File_write_all(fh, Bufeta, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
               err = MPI_File_close(&fh);
            }
          }
          //else
            //cudaThreadSynchronize();

          // write-statistics to chk file:
          if(rank==0){
            i = ND+2+ngsl;
            j = i;
            k = nzt+align-1-ND;
            fprintf(fchk,"%ld :\t%e\t%e\t%e\n",cur_step,u1[i][j][k],v1[i][j][k],w1[i][j][k]);
            fflush(fchk);
          }
         }
         //else
          //cudaThreadSynchronize();

          if((cur_step<NST-1) && (IFAULT == 2) && ((cur_step+1)%READ_STEP_GPU == 0) && (rank==srcproc)){
            printf("%d) Read new source from CPU.\n",rank);
            if((cur_step+1)%READ_STEP == 0){
              printf("%d) Read new source from file.\n",rank);
              read_src_ifault_2(rank, READ_STEP,
                INSRC, INSRC_I2,
                maxdim, coord, NZ,
                nxt, nyt, nzt,
                &npsrc, &srcproc,
                &tpsrc, &taxx, &tayy, &tazz,
                &taxz, &tayz, &taxy, (cur_step+1)/READ_STEP+1);
            }
            printf("%d) SOURCE: taxx,xy,xz:%e,%e,%e\n",rank,
                taxx[cur_step%READ_STEP],taxy[cur_step%READ_STEP],taxz[cur_step%READ_STEP]);
            // Synchronous copy!
            Cpy2Device_source(npsrc, READ_STEP_GPU,
              ((cur_step+1)%READ_STEP),
              taxx, tayy, tazz,
              taxz, tayz, taxy,
              d_taxx, d_tayy, d_tazz,
              d_taxz, d_tayz, d_taxy);
            source_step = 0;
          }/*
          if((cur_step<NST) && (cur_step%25==0) && (rank==srcproc)){
            printf("%d) SOURCE: taxx,xy,xz:%e,%e,%e\n",rank,
                taxx[cur_step],taxy[cur_step],taxz[cur_step]);
          }*/
       }
       //This loop is purly stencil computation, overlapping are utilized here
       //time_un  -= gethrtime();
       for(cur_step=cur_step;cur_step<=nt;cur_step++)
       {
         if(rank==0) printf("Time Step =                   %ld    OF  Total Timesteps = %ld\n", cur_step, nt);
         //pre-post MPI Message
         PostRecvMsg_Y(RF_vel, RB_vel, MCW, request_y, &count_y, msg_v_size_y, y_rank_F, y_rank_B);
         PostRecvMsg_X(RL_vel, RR_vel, MCW, request_x, &count_x, msg_v_size_x, x_rank_L, x_rank_R);
         //velocity computation in y boundary, two ghost cell region, two different streams to control
         dvelcy_H(d_u1, d_v1, d_w1, d_xx,   d_yy,   d_zz,   d_xy,       d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz,
                  d_d1, nxt,  nzt,  d_f_u1, d_f_v1, d_f_w1, stream_1,   yfs,  yfe,  y_rank_F);
         Cpy2Host_VY(d_f_u1, d_f_v1, d_f_w1,  SF_vel, nxt, nzt, stream_1, y_rank_F);
         dvelcy_H(d_u1, d_v1, d_w1, d_xx,   d_yy,   d_zz,   d_xy,       d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz,
                  d_d1, nxt,  nzt,  d_b_u1, d_b_v1, d_b_w1, stream_2,   ybs,  ybe,  y_rank_B);
         Cpy2Host_VY(d_b_u1, d_b_v1, d_b_w1,  SB_vel, nxt, nzt, stream_2, y_rank_B);
         //Memory copy from GPU to CPU, and velocity computation at the same time
         dvelcx_H_opt(d_u1, d_v1, d_w1, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_dcrjx, d_dcrjy, d_dcrjz,
                      d_d1, nyt,  nzt,  stream_i,   xvs,  xve);

         //MPI overlapping velocity computation
         cudaStreamSynchronize(stream_1);
         PostSendMsg_Y(SF_vel, SB_vel, MCW, request_y, &count_y, msg_v_size_y, y_rank_F, y_rank_B, rank, Front);
         cudaStreamSynchronize(stream_2);
         PostSendMsg_Y(SF_vel, SB_vel, MCW, request_y, &count_y, msg_v_size_y, y_rank_F, y_rank_B, rank, Back);
         MPI_Waitall(count_y, request_y, status_y);
         Cpy2Device_VY(d_u1,     d_v1,     d_w1,     d_f_u1, d_f_v1, d_f_w1, d_b_u1, d_b_v1, d_b_w1, RF_vel, RB_vel, nxt, nyt, nzt,
                       stream_1, stream_2, y_rank_F, y_rank_B);
         cudaThreadSynchronize();
         //start stress computation in insider part
         dstrqc_H_new(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz,
                      d_r1, d_r2, d_r3, d_r4, d_r5, d_r6,
                      d_u1, d_v1, d_w1, d_lam,
                      d_mu, d_qp,d_coeff, d_qs, d_dcrjx, d_dcrjy, d_dcrjz,
                      nyt,  nzt,  stream_i, d_lam_mu,
                      d_vx1, d_vx2, d_ww, d_wwo,
                      NX,   coord[0], coord[1],   xss2, xse2,
                      yls,  yre);
          // plasticity related calls:
          if(NVE==3){
            cudaDeviceSynchronize();
            drprecpc_calc_H_opt(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu, d_d1,
               d_sigma2,
               d_yldfac,d_cohes, d_phi,
               d_neta,
               d_yldfac_L, d_yldfac_R, d_yldfac_F, d_yldfac_B,
               d_yldfac_FL, d_yldfac_FR, d_yldfac_BL, d_yldfac_BR,
               nzt, xss2, xse2, ylsp, yrep, stream_i);
            cudaDeviceSynchronize();
            drprecpc_app_H(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu,
               d_sigma2,
               d_yldfac,
               nzt, xss2, xse2, ylsp, yrep, stream_i);
          }
         Cpy2Host_VX(d_u1, d_v1, d_w1, SL_vel, nxt, nyt, nzt, stream_1, x_rank_L, Left);
         Cpy2Host_VX(d_u1, d_v1, d_w1, SR_vel, nxt, nyt, nzt, stream_2, x_rank_R, Right);
         //MPI overlapping stress computation
         cudaStreamSynchronize(stream_1);
         PostSendMsg_X(SL_vel, SR_vel, MCW, request_x, &count_x, msg_v_size_x, x_rank_L, x_rank_R, rank, Left);
         cudaStreamSynchronize(stream_2);
         PostSendMsg_X(SL_vel, SR_vel, MCW, request_x, &count_x, msg_v_size_x, x_rank_L, x_rank_R, rank, Right);
         MPI_Waitall(count_x, request_x, status_x);
         Cpy2Device_VX(d_u1, d_v1, d_w1, RL_vel, RR_vel, nxt, nyt, nzt, stream_1, stream_2, x_rank_L, x_rank_R);
         //stress computation in ghost cells
         dstrqc_H_new(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz,
                      d_r1, d_r2, d_r3, d_r4, d_r5, d_r6,
                      d_u1, d_v1, d_w1, d_lam,
                      d_mu, d_qp,d_coeff, d_qs, d_dcrjx, d_dcrjy, d_dcrjz,
                      nyt,  nzt,  stream_1, d_lam_mu,
                      d_vx1, d_vx2, d_ww, d_wwo,
                      NX,   coord[0], coord[1],   xss1, xse1,
                      yls,  yre);
          // plasticity related calls:
          if(NVE==3){
            cudaDeviceSynchronize();
            drprecpc_calc_H_opt(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu, d_d1,
               d_sigma2,
               d_yldfac,d_cohes, d_phi,
               d_neta,
               d_yldfac_L, d_yldfac_R, d_yldfac_F, d_yldfac_B,
               d_yldfac_FL, d_yldfac_FR, d_yldfac_BL, d_yldfac_BR,
               nzt, xss1, xse1, ylsp, yrep, stream_1);
            cudaDeviceSynchronize();
            drprecpc_app_H(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu,
               d_sigma2,
               d_yldfac,
               nzt, xss1, xse1, ylsp, yrep, stream_1);
          }
         dstrqc_H_new(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz,
                      d_r1, d_r2, d_r3, d_r4, d_r5, d_r6,
                     d_u1, d_v1, d_w1, d_lam,
                     d_mu, d_qp,d_coeff, d_qs, d_dcrjx, d_dcrjy, d_dcrjz,
                     nyt,  nzt,  stream_2, d_lam_mu,
                     d_vx1, d_vx2, d_ww, d_wwo,
                     NX,   coord[0], coord[1],   xss3, xse3,
                     yls,  yre);
         // plasticity related calls:
         if(NVE==3){
           cudaDeviceSynchronize();
           drprecpc_calc_H_opt(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu, d_d1,
              d_sigma2,
              d_yldfac,d_cohes, d_phi,
              d_neta,
              d_yldfac_L, d_yldfac_R, d_yldfac_F, d_yldfac_B,
              d_yldfac_FL, d_yldfac_FR, d_yldfac_BL, d_yldfac_BR,
              nzt, xss3, xse3, ylsp, yrep, stream_2);
           cudaDeviceSynchronize();
           drprecpc_app_H(d_xx, d_yy, d_zz, d_xy, d_xz, d_yz, d_mu,
              d_sigma2,
              d_yldfac,
              nzt, xss3, xse3, ylsp, yrep, stream_2);
         }

         if(cur_step%NTISKP == 0){
          num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
          cudaMemcpy(&u1[0][0][0],d_u1,num_bytes,cudaMemcpyDeviceToHost);
          cudaMemcpy(&v1[0][0][0],d_v1,num_bytes,cudaMemcpyDeviceToHost);
          cudaMemcpy(&w1[0][0][0],d_w1,num_bytes,cudaMemcpyDeviceToHost);
          idtmp = ((cur_step/NTISKP+WRITE_STEP-1)%WRITE_STEP);
          idtmp = idtmp*rec_nxt*rec_nyt*rec_nzt;
          tmpInd = idtmp;
          // surface: k=nzt+align-1;
          for(k=nzt+align-1 - rec_nbgz; k>=nzt+align-1 - rec_nedz; k=k-NSKPZ)
            for(j=2+ngsl + rec_nbgy; j<=2+ngsl + rec_nedy; j=j+NSKPY)
              for(i=2+ngsl + rec_nbgx; i<=2+ngsl + rec_nedx; i=i+NSKPX)
              {
                //idx = (i-2-ngsl)/NSKPX;
                //idy = (j-2-ngsl)/NSKPY;
                //idz = ((nzt+align-1) - k)/NSKPZ;
                //tmpInd = idtmp + idz*rec_nxt*rec_nyt + idy*rec_nxt + idx;
                Bufx[tmpInd] = u1[i][j][k];
                Bufy[tmpInd] = v1[i][j][k];
                Bufz[tmpInd] = w1[i][j][k];
                tmpInd++;
              }
          if(!rank){
          time_gpuio_tmp += gethrtime();
              time_gpuio += time_gpuio_tmp;
              printf("Output data buffered in (sec): %lf\n",time_gpuio_tmp);
           }
            if((cur_step/NTISKP)%WRITE_STEP == 0){
            cudaThreadSynchronize();
            //printf("I'm %d, my disp=%ld\n", rank, displacement);
            sprintf(filename, "%s%07ld", filenamebasex, cur_step);
              if(!rank) time_fileio_tmp = -gethrtime();
            err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
            err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
            err = MPI_File_write_all(fh, Bufx, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
            err = MPI_File_close(&fh);
            sprintf(filename, "%s%07ld", filenamebasey, cur_step);
            err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
            err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
            err = MPI_File_write_all(fh, Bufy, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
            err = MPI_File_close(&fh);
            sprintf(filename, "%s%07ld", filenamebasez, cur_step);
            err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
            err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype, "native", MPI_INFO_NULL);
            err = MPI_File_write_all(fh, Bufz, rec_nxt*rec_nyt*rec_nzt*WRITE_STEP, MPI_FLOAT, &filestatus);
            err = MPI_File_close(&fh);
            if(!rank){
                time_fileio_tmp += gethrtime();
             time_fileio += time_fileio_tmp;
                printf("Output data written in (sec): %lf\n",time_fileio_tmp);
              }
          }
          else
            cudaThreadSynchronize();
          if(rank==0){
            i = ND+2+ngsl;
            j = i;
            k = nzt+align-1-ND;
            fprintf(fchk,"%ld :\t%e\t%e\t%e\n",cur_step,u1[i][j][k],v1[i][j][k],w1[i][j][k]);
            fflush(fchk);
          }
         }
         else
           cudaThreadSynchronize();

       }
       time_un += gethrtime();
    }
    if(rank==0){
      fprintf(fchk,"END\n");
      fclose(fchk);
    }

    //This should save the final plastic strain tensor at the end of the simulation
    if (NVE == 3){
      fprintf(stdout, "copying plastic strain back to CPU\n");
      num_bytes = sizeof(float)*(nxt+4+ngsl2)*(nyt+4+ngsl2)*(nzt+2*align);
      cudaMemcpy(&neta[0][0][0],d_neta,num_bytes,cudaMemcpyDeviceToHost);
      tmpInd = 0;

      rec_NZ = (NEDZ_EP-NBGZ)/NSKPZ+1;
      calcRecordingPoints(&rec_nbgx, &rec_nedx, &rec_nbgy, &rec_nedy,
        &rec_nbgz, &rec_nedz, &rec_nxt, &rec_nyt, &rec_nzt, &displacement,
        (long int)nxt,(long int)nyt,(long int)nzt, rec_NX, rec_NY, rec_NZ,
        NBGX,NEDX,NSKPX, NBGY,NEDY,NSKPY, NBGZ,NEDZ_EP,NSKPZ, coord);
     printf("%d = (%d,%d)) NX,NY,NZ=%d,%d,%d\nnxt,nyt,nzt=%d,%d,%d\nrec_N=(%d,%d,%d)\nrec_nxt,=%d,%d,%d\nNBGX,SKP,END=(%d:%d:%d),(%d:%d:%d),(%d:%d:%d)\nrec_nbg,ed=(%d,%d),(%d,%d),(%d,%d)\ndisp=%ld\n",
        rank,coord[0],coord[1],NX,NY,NZ,nxt,nyt,nzt,
        rec_NX, rec_NY, rec_NZ, rec_nxt, rec_nyt, rec_nzt,
        NBGX,NSKPX,NEDX,NBGY,NSKPY,NEDY,NBGZ,NSKPZ,NEDZ_EP,
        rec_nbgx,rec_nedx,rec_nbgy,rec_nedy,rec_nbgz,rec_nedz,(long int)displacement);


      /*this should save the final plastic strain down to NEDZ_EP grip points*/
      Bufeta2  = Alloc1D(rec_nxt*rec_nyt*rec_nzt);

      for(k=nzt+align-1 - rec_nbgz; k>=nzt+align-1 - rec_nedz; k=k-NSKPZ)
        for(j=2+ngsl + rec_nbgy; j<=2+ngsl + rec_nedy; j=j+NSKPY)
          for(i=2+ngsl + rec_nbgx; i<=2+ngsl + rec_nedx; i=i+NSKPX) {
            if (tmpInd >= (rec_nxt*rec_nyt*rec_nzt))
               fprintf(stdout, "tmpind=%d (allocated %d)\n", tmpInd, (rec_nxt*rec_nyt*rec_nzt));
            Bufeta2[tmpInd] = neta[i][j][k];
            tmpInd++;
          }

      MPI_Datatype filetype2;


    maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>rec_NZ?maxNX_NY_NZ_WS:rec_NZ);
    int ones2[maxNX_NY_NZ_WS];
    MPI_Aint dispArray2[maxNX_NY_NZ_WS];
    for(i=0;i<maxNX_NY_NZ_WS;++i){
      ones2[i] = 1;
    }

      err = MPI_Type_contiguous(rec_nxt, MPI_FLOAT, &filetype2);
    err = MPI_Type_commit(&filetype2);
    for(i=0;i<rec_nyt;i++){
      dispArray2[i] = sizeof(float);
      dispArray2[i] = dispArray2[i]*rec_NX*i;
    }
    err = MPI_Type_create_hindexed(rec_nyt, ones2, dispArray2, filetype2, &filetype2);
    err = MPI_Type_commit(&filetype2);
    for(i=0;i<rec_nzt;i++){
      dispArray2[i] = sizeof(float);
      dispArray2[i] = dispArray2[i]*rec_NY*rec_NX*i;
    }
    err = MPI_Type_create_hindexed(rec_nzt, ones2, dispArray2, filetype2, &filetype2);
    err = MPI_Type_commit(&filetype2);
    MPI_Type_size(filetype2, &tmpSize);
    printf("filetype size (supposedly=rec_nxt*rec_nyt*rec_nzt*4=%d) =%d\n", rec_nxt*rec_nyt*rec_nzt*4,tmpSize);

      sprintf(filename, "Finaleta%07ld", cur_step);
      err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
      err = MPI_File_set_view(fh, displacement, MPI_FLOAT, filetype2, "native", MPI_INFO_NULL);
      if (err != MPI_SUCCESS) {
         fprintf(stderr, "MPI error in MPI_File_set_view():\n");
         char errstr[200];
         int strlen;
         MPI_Error_string(err, errstr, &strlen);
         fprintf(stderr, "MPI error in MPI_File_set_view(): %s\n", errstr);
      }
      err = MPI_File_write_all(fh, Bufeta2, rec_nxt*rec_nyt*rec_nzt, MPI_FLOAT, &filestatus);
      if (err != MPI_SUCCESS) {
         char errstr[200];
         int strlen;
         MPI_Error_string(err, errstr, &strlen);
         fprintf(stderr, "MPI error in MPI_File_write_all(): %s\n", errstr);
      }
      err = MPI_File_close(&fh);

    }

    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_i);
    cudaFreeHost(SL_vel);
    cudaFreeHost(SR_vel);
    cudaFreeHost(RL_vel);
    cudaFreeHost(RR_vel);
    cudaFreeHost(SF_vel);
    cudaFreeHost(SB_vel);
    cudaFreeHost(RF_vel);
    cudaFreeHost(RB_vel);
    if(NVE==3){
      cudaFreeHost(SL_yldfac);
      cudaFreeHost(SR_yldfac);
      cudaFreeHost(RL_yldfac);
      cudaFreeHost(RR_yldfac);
      cudaFreeHost(SF_yldfac);
      cudaFreeHost(SB_yldfac);
      cudaFreeHost(RF_yldfac);
      cudaFreeHost(RB_yldfac);
      cudaFreeHost(SFL_yldfac);
      cudaFreeHost(SFR_yldfac);
      cudaFreeHost(RFL_yldfac);
      cudaFreeHost(RFR_yldfac);
      cudaFreeHost(SBL_yldfac);
      cudaFreeHost(SBR_yldfac);
      cudaFreeHost(RBL_yldfac);
      cudaFreeHost(RBR_yldfac);
    }
    GFLOPS  = 1.0;
    if (NVE < 2) GFLOPS  = GFLOPS*307.0*(xre - xls)*(yre-yls)*nzt;
    else GFLOPS  = GFLOPS*511.0*(xre - xls)*(yre-yls)*nzt;
    GFLOPS  = GFLOPS/(1000*1000*1000);
    //time_un = time_un/(cur_step-READ_STEP);
    time_un /= cur_step;
    GFLOPS  = GFLOPS/time_un;
    MPI_Allreduce( &GFLOPS, &GFLOPS_SUM, 1, MPI_DOUBLE, MPI_SUM, MCW );
    if(rank==0)
    {
        printf("GPU benchmark size NX=%d, NY=%d, NZ=%d, ReadStep=%d\n", NX, NY, NZ, READ_STEP);
        printf("GPU computing flops=%1.18f GFLOPS, time = %1.18f secs per timestep\n", GFLOPS_SUM, time_un);
    }
//  Main Loop Ends

//  program ends, free all memories
    UnBindArrayFromTexture();
    Delloc3D(u1);
    Delloc3D(v1);
    Delloc3D(w1);
    Delloc3D(xx);
    Delloc3D(yy);
    Delloc3D(zz);
    Delloc3D(xy);
    Delloc3D(yz);
    Delloc3D(xz);
    Delloc3D(vx1);
    Delloc3D(vx2);
    Delloc3Dww(ww);
    Delloc3D(wwo);

    cudaFree(d_u1);
    cudaFree(d_v1);
    cudaFree(d_w1);
    cudaFree(d_f_u1);
    cudaFree(d_f_v1);
    cudaFree(d_f_w1);
    cudaFree(d_b_u1);
    cudaFree(d_b_v1);
    cudaFree(d_b_w1);
    cudaFree(d_xx);
    cudaFree(d_yy);
    cudaFree(d_zz);
    cudaFree(d_xy);
    cudaFree(d_yz);
    cudaFree(d_xz);
    cudaFree(d_vx1);
    cudaFree(d_vx2);

    if(NVE==1 || NVE==3)
    {
       Delloc3D(r1);
       Delloc3D(r2);
       Delloc3D(r3);
       Delloc3D(r4);
       Delloc3D(r5);
       Delloc3D(r6);
       cudaFree(d_r1);
       cudaFree(d_r2);
       cudaFree(d_r3);
       cudaFree(d_r4);
       cudaFree(d_r5);
       cudaFree(d_r6);

       Delloc3D(qp);
       Delloc1D(coeff);
       Delloc3D(qs);
       cudaFree(d_qp);
       cudaFree(d_coeff);
       cudaFree(d_qs);
    }
    if(NVE==3){
      Delloc3D(sigma2);
      Delloc3D(cohes);
      Delloc3D(phi);
      Delloc3D(yldfac);
      Delloc3D(neta);
    }

    if(NPC==0)
    {
        Delloc1D(dcrjx);
        Delloc1D(dcrjy);
        Delloc1D(dcrjz);
        cudaFree(d_dcrjx);
        cudaFree(d_dcrjy);
        cudaFree(d_dcrjz);
    }

    Delloc3D(d1);
    Delloc3D(mu);
    Delloc3D(lam);
    Delloc3D(lam_mu);
    cudaFree(d_d1);
    cudaFree(d_mu);
    cudaFree(d_lam);
    cudaFree(d_lam_mu);

    if(rank==srcproc)
    {
       Delloc1D(taxx);
       Delloc1D(tayy);
       Delloc1D(tazz);
       Delloc1D(taxz);
       Delloc1D(tayz);
       Delloc1D(taxy);
       cudaFree(d_taxx);
       cudaFree(d_tayy);
       cudaFree(d_tazz);
       cudaFree(d_taxz);
       cudaFree(d_tayz);
       cudaFree(d_taxy);
       Delloc1P(tpsrc);
       cudaFree(d_tpsrc);
    }

    MPI_Comm_free( &MC1 );
    MPI_Finalize();
    return (0);
}

// Calculates recording points for each core
// rec_nbgxyz rec_nedxyz...
// WARNING: Assumes NPZ = 1! Only surface outputs are needed!
void calcRecordingPoints(int *rec_nbgx, int *rec_nedx,
  int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz,
  int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
  long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
  int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY,
  int NBGZ, int NEDZ, int NSKPZ, int *coord){

  *displacement = 0;

  if(NBGX > nxt*(coord[0]+1))     *rec_nxt = 0;
  else if(NEDX < nxt*coord[0]+1)  *rec_nxt = 0;
  else{
    if(nxt*coord[0] >= NBGX){
      *rec_nbgx = (nxt*coord[0]+NBGX-1)%NSKPX;
      *displacement += (nxt*coord[0]-NBGX)/NSKPX+1;
    }
    else
      *rec_nbgx = NBGX-nxt*coord[0]-1;  // since rec_nbgx is 0-based
    if(nxt*(coord[0]+1) <= NEDX)
      *rec_nedx = (nxt*(coord[0]+1)+NBGX-1)%NSKPX-NSKPX+nxt;
    else
      *rec_nedx = NEDX-nxt*coord[0]-1;
    *rec_nxt = (*rec_nedx-*rec_nbgx)/NSKPX+1;
  }

  if(NBGY > nyt*(coord[1]+1))     *rec_nyt = 0;
  else if(NEDY < nyt*coord[1]+1)  *rec_nyt = 0;
  else{
    if(nyt*coord[1] >= NBGY){
      *rec_nbgy = (nyt*coord[1]+NBGY-1)%NSKPY;
      *displacement += ((nyt*coord[1]-NBGY)/NSKPY+1)*rec_NX;
    }
    else
      *rec_nbgy = NBGY-nyt*coord[1]-1;  // since rec_nbgy is 0-based
    if(nyt*(coord[1]+1) <= NEDY)
      *rec_nedy = (nyt*(coord[1]+1)+NBGY-1)%NSKPY-NSKPY+nyt;
    else
      *rec_nedy = NEDY-nyt*coord[1]-1;
    *rec_nyt = (*rec_nedy-*rec_nbgy)/NSKPY+1;
  }

  if(NBGZ > nzt) *rec_nzt = 0;
  else{
    *rec_nbgz = NBGZ-1;  // since rec_nbgz is 0-based
    *rec_nedz = NEDZ-1;
    *rec_nzt = (*rec_nedz-*rec_nbgz)/NSKPZ+1;
  }

  if(*rec_nxt == 0 || *rec_nyt == 0 || *rec_nzt == 0){
    *rec_nxt = 0;
    *rec_nyt = 0;
    *rec_nzt = 0;
  }

  // displacement assumes NPZ=1!
  *displacement *= sizeof(float);

  return;
}
