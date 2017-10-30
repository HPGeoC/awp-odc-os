/**
 @author Daniel Roten
 @author Kyle Withers
 
 @section LICENSE
 Copyright (c) 2013-2016, Regents of the University of California
 Copyright (c) 2015-2017, San Diego State University Research Foundation
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _KERNEL_H
#define _KERNEL_H

__global__ void dvelcx( float *u1,    float *v1,    float *w1,    float *xx,  float *yy,  float *zz, float *xy, float *xz, float *yz,
                        float *dcrjx, float *dcrjy, float *dcrjz, float *d_1, int s_i,    int e_i );

__global__ void dvelcy( float *u1,    float *v1,    float *w1,    float *xx,  float *yy,    float *zz,    float *xy,    float *xz,  float *yz,
                        float *dcrjx, float *dcrjy, float *dcrjz, float *d_1, float *s_u1,  float *s_v1,  float *s_w1,  int s_j,    int e_j );

__global__ void update_boundary_y( float *u1, float *v1, float *w1, float *s_u1, float *s_v1, float *s_w1, int rank, int flag );

__global__ void update_yldfac( float *yldfac,
                               float *buf_L,  float *buf_R,   float *buf_F,   float *buf_B,
                               float *buf_FL, float *buf_FR,  float *buf_BL,  float *buf_BR );

__global__ void fvelxyz( float *u1, float *v1, float *w1, float *lam_mu, int xls, int NX, int rankx );

__global__ void dstrqc( float *xx,  float *yy,    float *zz,    float *xy,    float *xz,      float *yz,
                        float *r1,  float *r2,    float *r3,    float *r4,    float *r5,      float *r6,
                        float *u1,  float *v1,    float *w1,    float *lam,   float *mu,      float *qp, float *coeff,
                        float *qs,  float *dcrjx, float *dcrjy, float *dcrjz, float *lam_mu,  int NX,
                        int rankx,  int ranky,    int s_i,      int e_i,      int s_j );

__global__ void fstr( float *zz, float *xz, float *yz, int s_i, int e_i, int s_j );

__global__ void drprecpc_calc( float *xx,     float *yy,      float *zz,
                               float *xy,     float *xz,      float *yz,    float *mu,  float *d1,
                               float *sigma2, float *yldfac,  float *cohes, float *phi, float *neta,
                               int s_i,       int e_i,        int s_j );

__global__ void drprecpc_app( float *xx,      float *yy,  float *zz,
                              float *xy,      float *xz,  float *yz, float *mu,
                              float *sigma2,  float *yldfac,
                              int s_i,        int e_i,    int s_j );

__global__ void addsrc_cu( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,
                           float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
                           float *xx,   float *yy,      float *zz,  float *xy,  float *yz,  float *xz );

__global__ void frcvel_cu( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,  int tskp,
                           float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
                           float *u1,   float *v1,      float *w1,  int xmin,   int xmax );

__global__ void fvel( float *u1, float *v1, float *w1, float *lam_mu, int NX, int rankx, int ranky, int s_i, int e_i, int s_j );

template< int BLOCK_Z, int BLOCK_Y >
__global__ void dvelcx_opt( float * __restrict__ u1,
                            float * __restrict__ v1,
                            float * __restrict__ w1,
                            const float *xx,    const float *yy,    const float *zz,
                            const float *xy,    const float *xz,    const float *yz,
                            const float *dcrjx, const float *dcrjy, const float *dcrjz,
                            const float *d_1,
                            const int s_i,
                            const int e_i );

template< int BLOCKX, int BLOCKY >
__global__ void
__launch_bounds__( 512, 2 )
dstrqc_new( float *__restrict__ xx, float *__restrict__ yy, float *__restrict__ zz,
            float *__restrict__ xy, float *__restrict__ xz, float *__restrict__ yz,
            float *__restrict__ r1, float *__restrict__ r2, float *__restrict__ r3,
            float *__restrict__ r4, float *__restrict__ r5, float *__restrict__ r6,
            float *__restrict__ u1,
            float *__restrict__ v1,
            float *__restrict__ w1,
            float *lam,
            float *mu,
            float *qp,
            float *coeff,
            float *qs,
            float *dcrjx, float *dcrjy, float *dcrjz, float *lam_mu,
            //float *d_vx1, float *d_vx2, float *d_ww, float *d_wwo, //pengs version
            float *d_vx1, float *d_vx2, int *d_ww,    float *d_wwo,
            int NX,       int rankx,    int ranky,    int nzt, int s_i, int e_i, int s_j, int e_j );

__global__ void
__launch_bounds__( 512, 2 )
drprecpc_calc_opt( float *xx,     float *yy,    float *zz,
                   const float *__restrict__ xy,
                   const float *__restrict__ xz,
                   const float *__restrict__ yz,
                   float *mu,     float *d1,
                   float *sigma2,
                   float *yldfac, float *cohes, float *phi,
                   float *neta,
                   int nzt,       int s_i,      int e_i, int s_j, int e_j );

__global__ void update_yldfac_buffer_x( float *yldfac, float *buf, int rank, int flag, int meshtp );

__global__ void update_yldfac_data_x( float *yldfac, float *buf, int rank, int flag, int meshtp );

__global__ void update_yldfac_buffer_y( float *yldfac, float *buf, int rank, int flag, int meshtp );

__global__ void update_yldfac_data_y( float *yldfac, float *buf, int rank, int flag, int meshtp );

#endif
