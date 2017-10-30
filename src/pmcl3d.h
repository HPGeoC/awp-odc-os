/*  
********************************************************************************
* pmcl3d.h                                                                     *
* programming in C language                                                    *
* all pmcl3d data types are defined here                                       * 
********************************************************************************
 
 @author Jun Zhou
 @author Daniel Roten
 @author Kyle Withers
 
 @section LICENSE
 Copyright (c) 2013-2017, Regents of the University of California
 Copyright (c) 2015-2017, San Diego State University Research Foundation
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _PMCL3D_H
#define _PMCL3D_H

#ifdef AWP_USE_MPI
#include <mpi.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include "pmcl3d_cons.h"
#include "constants.hpp"

void SetDeviceConstValue( float DH,     float DT,     int nxt,    int nyt, int nzt,
                          float fmajor, float fminor, float *Rz,  float *RzT );

void BindArrayToTexture( float *vx1, float *vx2, int *ww, float *wwo, int memsize );

void UnBindArrayFromTexture();

void dvelcx_H( float *u1,     float *v1,    float *w1,    float *xx,  float *yy,  float *zz,  float *xy,        float *xz,  float *yz,
               float *dcrjx,  float *dcrjy, float *dcrjz, float *d_1, int nyt,    int nzt,    cudaStream_t St,  int s_i,    int e_i );

void dvelcx_H_opt( float *u1,       float *v1,    float *w1,
                   float *xx,       float *yy,    float *zz, float *xy,
                   float *xz,       float *yz,
                   float *dcrjx,    float *dcrjy, float *dcrjz,
                   float *d_1,      int nyt,      int nzt,
                   cudaStream_t St, int s_i,      int e_i );

void dvelcy_H( float *u1,       float *v1,    float *w1,    float *xx,  float *yy,  float *zz,  float *xy,    float *xz,    float *yz,
               float *dcrjx,    float *dcrjy, float *dcrjz, float *d_1, int nxt,    int nzt,    float *s_u1,  float *s_v1,  float *s_w1,
               cudaStream_t St, int s_j,      int e_j,      int rank );

void dstrqc_H( float *xx,       float *yy,      float *zz,    float *xy,    float *xz,  float *yz,
               float *r1,       float *r2,      float *r3,    float *r4,    float *r5,  float *r6,
               float *u1,       float *v1,      float *w1,    float *lam,   float *mu,  float *qp, float *coeff,
               float *qs,       float *dcrjx,   float *dcrjy, float *dcrjz, int nyt,    int nzt,
               cudaStream_t St, float *lam_mu,  int NX,       int rankx,    int ranky,  int s_i,
               int e_i,         int s_j,        int e_j );

void dstrqc_H_new( float *xx,       float *yy,    float *zz,    float *xy,    float *xz,  float *yz,
                   float *r1,       float *r2,    float *r3,    float *r4,    float *r5,  float *r6,
                   float *u1,       float *v1,    float *w1,    float *lam,   float *mu,  float *qp, float *coeff,
                   float *qs,       float *dcrjx, float *dcrjy, float *dcrjz, int nyt,    int nzt,
                   cudaStream_t St, float *lam_mu,
                   float *vx1,      float *vx2,   int *ww,      float *wwo,
                   int NX,          int rankx,    int ranky,    int  s_i,
                   int e_i,         int s_j,      int e_j );

void fstr_H( float *zz, float *xz, float *yz, cudaStream_t St, int s_i, int e_i, int s_j, int e_j );

void drprecpc_calc_H( float *xx,      float *yy,    float *zz,  float *xy,  float *xz, float *yz,
                      float *mu,      float *d1,    float *sigma2,
                      float *yldfac,  float *cohes, float *phi,
                      float *neta,
                      int nzt,
                      int xls,        int xre,      int yls,    int yre,    cudaStream_t St );

void drprecpc_calc_H_opt( float *xx,      float *yy,    float *zz,  float *xy,  float *xz, float *yz,
                          float *mu,      float *d1,    float *sigma2,
                          float *yldfac,  float *cohes, float *phi,
                          float *neta,
                          int nzt,
                          int xls,        int xre,      int yls,    int yre,    cudaStream_t St );

void drprecpc_app_H( float *xx,     float *yy,  float *zz,  float *xy,  float *xz,  float *yz,
                     float *mu,
                     float *sigma2, float *yldfac,
                     int nzt,       int xls,    int xre,    int yls,    int yre,    cudaStream_t St );

void addsrc_H( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,  cudaStream_t St,
               float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
               float *xx,   float *yy,      float *zz,  float *xy,  float *yz,  float *xz );

void frcvel_H( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,  int tskp, cudaStream_t St,
               float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
               float *u1,   float *v1,      float *w1,  int ymin,   int ymax );

void fvel_H( float *u1, float *v1,  float *w1,  cudaStream_t St, float *lam_mu, int NX, int rankx, int ranky,
             int s_i,   int e_i,    int s_j,    int e_j );

void calcRecordingPoints( int *rec_nbgx,  int *rec_nedx,
                          int *rec_nbgy,  int *rec_nedy,  int *rec_nbgz,  int *rec_nedz,
                          int *rec_nxt,   int *rec_nyt,   int *rec_nzt,   MPI_Offset *displacement,
                          long int nxt,   long int nyt,   long int nzt,   int rec_NX, int rec_NY, int rec_NZ,
                          int NBGX,       int NEDX,       int NSKPX,      int NBGY,   int NEDY,   int NSKPY,
                          int NBGZ,       int NEDZ,       int NSKPZ,      int *coord );

void command( int argc,       char **argv,
              float *TMAX,    float *DH,        float *DT,  float *ARBC,  float *PHT,
              int *NPC,       int *ND,          int *NSRC,  int *NST,     int *NVAR,
              int *NVE,       int *MEDIASTART,  int *IFAULT,
              int *READ_STEP, int *READ_STEP_GPU,
              int *NTISKP,    int *WRITE_STEP,
              int *NX,        int *NY,          int *NZ,    int *PX,      int *PY,
              int *NBGX,      int *NEDX,        int *NSKPX,
              int *NBGY,      int *NEDY,        int *NSKPY,
              int *NBGZ,      int *NEDZ,        int *NSKPZ,
              float *FAC,     float *Q0,        float *EX,  float *FP,    int *IDYNA, int *SoCalQ,
              char *INSRC,    char *INVEL,      char *OUT,  char *INSRC_I2,
              char *CHKFILE,  char *INRCVR,     char *OUTRCVR );

int read_src_ifault_2( int rank,      int READ_STEP,
                       char *INSRC,   char *INSRC_I2,
                       int maxdim,    int *coords,  int NZ,
                       int nxt,       int nyt,      int nzt,
                       int *NPSRC,    int *SRCPROC,
                       PosInf *psrc,  Grid1D *axx,  Grid1D *ayy, Grid1D *azz,
                       Grid1D *axz,   Grid1D *ayz,  Grid1D *axy,
                       int idx );

int read_src_ifault_4( int rank,          int READ_STEP,  char *INSRC,
                       int maxdim,        int *coords,    int NZ,       int nxt,          int nyt,        int nzt,
                       int *NPSRC,        int *SRCPROC,   PosInf *psrc, Grid1D *axx,      Grid1D *ayy,    Grid1D *azz,
                       int idx,           int *fbc_ext,   int *fbc_off, char *fbc_pmask,  int *fbc_extl,  int *fbc_dim,
                       int *fbc_seismio,  int *fbc_tskp,  int nst,      int size );

int inisource( int rank,        int IFAULT,     int NSRC,       int READ_STEP,
               int NST,         int *SRCPROC,   int NZ,
               MPI_Comm MCW,    int nxt,        int nyt,        int nzt,
               int *coords,     int maxdim,     int *NPSRC,
               PosInf *ptpsrc,  Grid1D *ptaxx,  Grid1D *ptayy,  Grid1D *ptazz,
               Grid1D *ptaxz,   Grid1D *ptayz,  Grid1D *ptaxy,  char *INSRC, char *INSRC_I2 );

void addsrc( int i,       float DH,   float DT,   int NST,    int npsrc,  int READ_STEP, int dim, PosInf psrc,
             Grid1D axx,  Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
             Grid3D xx,   Grid3D yy,  Grid3D zz,  Grid3D xy,  Grid3D yz,  Grid3D xz );

void frcvel( int i,       float DH,   float DT,   int NST,    int npsrc,  int tpsrc, int READ_STEP, int dim, PosInf psrc,
             Grid1D axx,  Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
             Grid3D u1,   Grid3D v1,  Grid3D w1,  int rank );

void inimesh( int rank,   int MEDIASTART, Grid3D d1,    Grid3D mu,  Grid3D lam, Grid3D qp,  Grid3D qs,  float *taumax,  float *taumin,
              Grid3D tau, Grid3D weights, Grid1D coeff,
              int nvar,   float FP,       float FAC,    float Q0,   float EX,   int nxt,    int nyt,    int nzt,        int PX, int PY, int NX, int NY,
              int NZ,     int *coords,    MPI_Comm MCW, int IDYNA,  int NVE,    int SoCalQ, char *INVEL,
              float *vse, float *vpe,     float *dde );

void inidrpr_hoekbrown_light( int nxt,        int nyt,    int nzt,        int nve, int *coords,
                              float dh,       int rank,
                              Grid3D mu,      Grid3D lam, Grid3D d1,
                              Grid3D sigma2,
                              Grid3D cohes,   Grid3D phi, float *fmajor,  float *fminor,
                              float *strike,  float *dip, MPI_Comm MCW );

void rotation_matrix( float *strike, float *dip, float *Rz, float *RzT );

int writeCHK( char *chkfile,  int ntiskp, float dt, float dh,
              int nxt,        int nyt,    int nzt,
              int nt,         float arbc, int npc,  int nve,
              float fac,      float q0,   float ex, float fp,
              float *vse,     float *vpe, float *dde);

void mediaswap( Grid3D d1,  Grid3D mu,    Grid3D lam,   Grid3D qp,    Grid3D qs,
                int rank,   int x_rank_L, int x_rank_R, int y_rank_F, int y_rank_B,
                int nxt,    int nyt,      int nzt,      MPI_Comm MCW );

void tausub( Grid3D tau, float taumin,float taumax );

void weights_sub( Grid3D weights, Grid1D coeff, float ex, float fac );

void inicrj( float ARBC, int *coords, int nxt, int nyt, int nzt, int NX, int NY, int ND, Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz );

void init_texture( int nxt,         int nyt,      int nzt,  Grid3D tau1, Grid3D tau2, Grid3D vx1, Grid3D vx2,
                   Grid3D weights,  Grid3Dww ww,  Grid3D wwo,
                   int xls,         int xre,      int yls,  int yre );

void PostRecvMsg_X( float *RL_M, float *RR_M, MPI_Comm MCW, MPI_Request *request, int *count, int msg_size, int rank_L, int rank_R );

void PostRecvMsg_Y( float *RF_M, float *RB_M, MPI_Comm MCW, MPI_Request *request, int *count, int msg_size, int rank_F, int rank_B );

void Cpy2Device_source( int npsrc,      int READ_STEP,
                        int index_offset,
                        Grid1D taxx,    Grid1D tayy,    Grid1D tazz,
                        Grid1D taxz,    Grid1D tayz,    Grid1D taxy,
                        float *d_taxx,  float *d_tayy,  float *d_tazz,
                        float *d_taxz,  float *d_tayz,  float *d_taxy, int IFAULT );

void Cpy2Host_VX( float *u1, float *v1, float *w1, float *h_m, int nxt, int nyt, int nzt, cudaStream_t St, int rank, int flag );

void Cpy2Host_VY( float *s_u1, float *s_v1, float *s_w1, float *h_m, int nxt, int nzt, cudaStream_t St, int rank );

void Cpy2Device_VX( float *u1,  float *v1,  float *w1,        float *L_m,       float *R_m, int nxt,
                    int nyt,    int nzt,    cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R );

void Cpy2Device_VY( float *u1,    float *v1,  float *w1,  float *f_u1,  float *f_v1,  float *f_w1,  float *b_u1,      float *b_v1,
                    float *b_w1,  float *F_m, float *B_m, int nxt,      int nyt,      int nzt,      cudaStream_t St1, cudaStream_t St2,
                    int rank_F,   int rank_B );

void Cpy2Host_yldfac( float *d_L,       float *d_R,   float *d_F,   float *d_B,
                      float *d_FL,      float *d_FR,  float *d_BL,  float *d_BR,
                      float *SL,        float *SR,    float *SF,    float *SB,
                      float *SFL,       float *SFR,   float *SBL,   float *SBR,
                      cudaStream_t St,  int rank_L,   int rank_R,   int rank_F, int rank_B,
                      int nxt,          int nyt );

void Cpy2Device_yldfac( float *d_yldfac,
                        float *d_L,       float *d_R,   float *d_F,   float *d_B,
                        float *d_FL,      float *d_FR,  float *d_BL,  float *d_BR,
                        float *RL,        float *RR,    float *RF,    float *RB,
                        float *RFL,       float *RFR,   float *RBL,   float *RBR,
                        cudaStream_t St,  int rank_L,   int rank_R,   int rank_F, int rank_B,
                        int nxt,          int nyt,      int nzt );

void PostSendMsg_X( float *SL_M,  float *SR_M,  MPI_Comm MCW, MPI_Request *request, int *count, int msg_size,
                    int rank_L,   int rank_R,   int rank,     int flag );

void PostSendMsg_Y( float *SF_M,  float *SB_M,  MPI_Comm MCW, MPI_Request *request, int *count, int msg_size,
                    int rank_F,   int rank_B,   int rank,     int flag );

void PostRecvMsg_yldfac( float *RL_M,     float *RR_M,          float *RF_M,  float *RB_M,
                         float *RFL_M,    float *RFR_M,         float *RBL_M, float *RBR_M,
                         MPI_Comm MCW,    MPI_Request *request, int *count,
                         int msg_size_x,  int msg_size_y,
                         int rank_L,      int rank_R,           int rank_F,   int rank_B,
                         int rank_FL,     int rank_FR,          int rank_BL,  int rank_BR );

void PostSendMsg_yldfac( float *SL_M,     float *SR_M,          float *SF_M,  float *SB_M,
                         float *SFL_M,    float *SFR_M,         float *SBL_M, float *SBR_M,
                         MPI_Comm MCW,    MPI_Request *request, int *count,
                         int msg_size_x,  int msg_size_y,       int rank,
                         int rank_L,      int rank_R,           int rank_F,   int rank_B,
                         int rank_FL,     int rank_FR,          int rank_BL,  int rank_BR );

#include "data/grid.h"

void Cpy2Host_yldfac_X( float *yldfac,    float *buf_L,     float *buf_R, float *d_buf_L, float *d_buf_R, int nyt, int nzt,
                        cudaStream_t St1, cudaStream_t St2, int rank_L,   int rank_R,     int meshtp );

void Cpy2Device_yldfac_X( float *yldfac,    float *buf_L,     float *buf_R, float *d_buf_L, float *d_buf_R, int nyt, int nzt,
                          cudaStream_t St1, cudaStream_t St2, int rank_L,   int rank_R,     int meshtp );

void Cpy2Host_yldfac_Y( float *yldfac,    float *buf_F,     float *buf_B, float *d_buf_F, float *d_buf_B, int nxt, int nzt,
                        cudaStream_t St1, cudaStream_t St2, int rank_F,   int rank_B,     int meshtp );

void Cpy2Device_yldfac_Y( float *yldfac,    float *buf_F,     float *buf_B, float *d_buf_F, float *d_buf_B, int nxt, int nzt,
                          cudaStream_t St1, cudaStream_t St2, int rank_F,   int rank_B,     int meshtp );

int background_velocity_reader( int rank, int size, int NST, int READ_STEP, MPI_Comm MCS );

void update_yldfac_buffer_x_H( float *yldfac, float *buf_L, float *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2,
                               int rank_L,    int rank_R,   int meshtp );

void update_yldfac_buffer_y_H( float *yldfac,     float *buf_F,     float *buf_B, int nxt,    int nzt,
                               cudaStream_t St1,  cudaStream_t St2, int rank_F,   int rank_B, int meshtp );

#endif
