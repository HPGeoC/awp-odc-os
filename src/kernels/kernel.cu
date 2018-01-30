/**
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

#include <stdio.h>
#include <math.h>
#include "kernel.h"
#include "pmcl3d_cons.h"
#include <cuda.h>

__constant__ float d_c1;
__constant__ float d_c2;
__constant__ float d_dth;
__constant__ float d_dt1;
__constant__ float d_dh1;
__constant__ float d_DT;
__constant__ float d_DH;
__constant__ int   d_nxt;
__constant__ int   d_nyt;
__constant__ int   d_nzt;
__constant__ int   d_slice_1;
__constant__ int   d_slice_2;
__constant__ int   d_yline_1;
__constant__ int   d_yline_2;

#define LDG(x) x

texture< float, 1, cudaReadModeElementType > p_vx1;
texture< float, 1, cudaReadModeElementType > p_vx2;
texture< int,   1, cudaReadModeElementType > p_ww;
texture< float, 1, cudaReadModeElementType > p_wwo;

//! Compute initial stress on GPU (Daniel)
__constant__ float d_fmajor;
__constant__ float d_fminor;
__constant__ float d_Rz[9];
__constant__ float d_RzT[9];

__device__ void matmul3( register float *a, register float *b, register float *c ) {
  register int i, j, k;

  for( i = 0; i < 3; i++ )
    for(j = 0; j < 3; j++ )
      for(k = 0; k < 3; k++ )
        c[i*3+j] += a[i*3+k] * b[k*3+j];
}

__device__ void rotate_principal( register float sigma2, register float pfluid, register float *ssp ) {
  register float ss[9], tmp[9];
  register int k;

  for( k = 0; k < 9; k++ )
    ss[k] = tmp[k] = ssp[k] = 0.;

  ss[0] = (sigma2 + pfluid) * d_fmajor;
  ss[4] = sigma2 + pfluid;
  ss[8] = (sigma2 + pfluid) * d_fminor;

  matmul3( d_RzT, ss, tmp );
  matmul3( tmp, d_Rz, ssp );

  ssp[0] -= pfluid;
  ssp[4] -= pfluid;
  ssp[8] -= pfluid;

  /*cuPrintf("ssp:%5.2e %5.2e %5.2e %5.2e %5.2e %5.2e\n",
     ssp[0], ssp[4], ssp[8],
     ssp[1], ssp[2], ssp[5]);*/
}

//! end of routines for on-GPU initial stress computation (Daniel)

extern "C"
void SetDeviceConstValue( float DH,     float DT,     int nxt,    int nyt, int nzt,
                          float fmajor, float fminor, float *Rz,  float *RzT ) {
  float h_c1, h_c2, h_dth, h_dt1, h_dh1;
  int   slice_1, slice_2, yline_1, yline_2;

  h_c1  = 9.0 / 8.0;
  h_c2  = -1.0 / 24.0;
  h_dth = DT / DH;
  h_dt1 = 1.0 / DT;
  h_dh1 = 1.0 / DH;
  slice_1  = (nyt + 4 + 8 * LOOP) * (nzt + 2 * ALIGN);
  slice_2  = (nyt + 4 + 8 * LOOP) * (nzt + 2 * ALIGN) * 2;
  yline_1  = nzt + 2 * ALIGN;
  yline_2  = (nzt + 2 * ALIGN) * 2;

  cudaMemcpyToSymbol( d_c1,       &h_c1,    sizeof( float ) );
  cudaMemcpyToSymbol( d_c2,       &h_c2,    sizeof( float ) );
  cudaMemcpyToSymbol( d_dth,      &h_dth,   sizeof( float ) );
  cudaMemcpyToSymbol( d_dt1,      &h_dt1,   sizeof( float ) );
  cudaMemcpyToSymbol( d_dh1,      &h_dh1,   sizeof( float ) );
  cudaMemcpyToSymbol( d_DT,       &DT,      sizeof( float ) );
  cudaMemcpyToSymbol( d_DH,       &DH,      sizeof( float ) );
  cudaMemcpyToSymbol( d_nxt,      &nxt,     sizeof( int )   );
  cudaMemcpyToSymbol( d_nyt,      &nyt,     sizeof( int )   );
  cudaMemcpyToSymbol( d_nzt,      &nzt,     sizeof( int )   );
  cudaMemcpyToSymbol( d_slice_1,  &slice_1, sizeof( int )   );
  cudaMemcpyToSymbol( d_slice_2,  &slice_2, sizeof( int )   );
  cudaMemcpyToSymbol( d_yline_1,  &yline_1, sizeof( int )   );
  cudaMemcpyToSymbol( d_yline_2,  &yline_2, sizeof( int )   );

  //! Compute initial stress on GPU (Daniel)
  cudaMemcpyToSymbol( d_fmajor, &fmajor,  sizeof( float )     );
  cudaMemcpyToSymbol( d_fminor, &fminor,  sizeof( float )     );
  cudaMemcpyToSymbol( d_Rz,     Rz,       9 * sizeof( float ) );
  cudaMemcpyToSymbol( d_RzT,    RzT,      9 * sizeof( float ) );

  return;
}

extern "C"
void BindArrayToTexture( float* vx1, float* vx2,int* ww, float* wwo, int memsize ) {
  cudaBindTexture( 0, p_vx1,  vx1,  memsize );
  cudaBindTexture( 0, p_vx2,  vx2,  memsize );
  cudaBindTexture( 0, p_ww,   ww,   memsize );
  cudaBindTexture( 0, p_wwo,  wwo,  memsize );
  cudaThreadSynchronize ();

  return;
}

extern "C"
void UnBindArrayFromTexture() {
  cudaUnbindTexture( p_vx1 );
  cudaUnbindTexture( p_vx2 );
  cudaUnbindTexture( p_ww );
  cudaUnbindTexture( p_wwo );

  return;
}

extern "C"
void dvelcx_H( float *u1,     float *v1,    float *w1,    float *xx,  float *yy,  float *zz,  float *xy,        float *xz,  float *yz,
               float *dcrjx,  float *dcrjy, float *dcrjz, float *d_1, int nyt,    int nzt,    cudaStream_t St,  int s_i,    int e_i ) {
  dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
  dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, (nyt + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( dvelcx, cudaFuncCachePreferL1 );
  dvelcx<<<grid, block, 0, St>>>( u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_i, e_i );

  return;
}

template< int BLOCKX, int BLOCKY >
__global__ void
__launch_bounds__( 640, 2 )
dvelcx_opt( float * __restrict__ u1,
            float * __restrict__ v1,
            float * __restrict__ w1,
            const float *xx,    const float *yy,    const float *zz,
            const float *xy,    const float *xz,    const float *yz,
            const float *dcrjx, const float *dcrjy, const float *dcrjz,
            const float *d_1,
            const int s_i,
            const int e_i ) {
  float f_xx,   xx_im1,   xx_ip1,   xx_im2;
  float f_xy,   xy_ip1,   xy_ip2,   xy_im1;
  float f_xz,   xz_ip1,   xz_ip2,   xz_im1;
  float f_dcrj, f_dcrjy,  f_dcrjz,  f_yz;

  const int k = blockIdx.x * BLOCKX + threadIdx.x + ALIGN;
  const int j = blockIdx.y * BLOCKY + threadIdx.y + 2 + 4 * LOOP;
  int pos     = e_i * d_slice_1 + j * d_yline_1 + k;
  int tx      = threadIdx.x, ty = threadIdx.y;

  __shared__ float s_xy[BLOCKX][BLOCKY+3], s_xz[BLOCKX+3][BLOCKY],
                   s_yy[BLOCKX][BLOCKY+3], s_yz[BLOCKX+3][BLOCKY+3], s_zz[BLOCKX+3][BLOCKY];
  __shared__ float s_d_1[BLOCKX+1][BLOCKY+1];

  f_xx    = xx[pos+d_slice_1];
  xx_im1  = xx[pos];
  xx_im2  = xx[pos-d_slice_1];
  xy_ip1  = xy[pos+d_slice_2];
  f_xy    = xy[pos+d_slice_1];
  xy_im1  = xy[pos];
  xz_ip1  = xz[pos+d_slice_2];
  f_xz    = xz[pos+d_slice_1];
  xz_im1  = xz[pos];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

  float f_d_1     = d_1[pos+d_slice_1];           //f_d_1_ip1 will get this value
  float f_d_1_km1 = d_1[pos+d_slice_1-1];         //f_d_1_ik1 will get this value
  float f_d_1_jm1 = d_1[pos+d_slice_1-d_yline_1];

  for( int i = e_i; i >= s_i; i-- ) {
    pos = i * d_slice_1 + j * d_yline_1 + k;
//    const int pos_km2  = pos - 2;
//    const int pos_km1  = pos - 1;
//    const int pos_kp1  = pos + 1;
//    const int pos_kp2  = pos + 2;
//    const int pos_jm2  = pos - d_yline_2;
//    const int pos_jm1  = pos - d_yline_1;
//    const int pos_jp1  = pos + d_yline_1;
//    const int pos_jp2  = pos + d_yline_2;
    const int pos_im1  = pos - d_slice_1;
    const int pos_im2  = pos - d_slice_2;
    //const int pos_ip1  = pos+d_slice_1;
//    const int pos_jk1  = pos - d_yline_1 - 1;
    //const int pos_ik1  = pos+d_slice_1-1;
//    const int pos_ijk  = pos + d_slice_1 - d_yline_1;

    //! xx pipeline
    xx_ip1  = f_xx;
    f_xx    = xx_im1;
    xx_im1  = xx_im2;
    xx_im2  = xx[pos_im2];

    //! xy pipeline
    xy_ip2  = xy_ip1;
    xy_ip1  = f_xy;
    f_xy    = xy_im1;
    xy_im1  = xy[pos_im1];

    //! xz pipeline
    xz_ip2  = xz_ip1;
    xz_ip1  = f_xz;
    f_xz    = xz_im1;
    xz_im1  = xz[pos_im1];

    f_yz    = yz[pos];

    __syncthreads();
    s_xy[tx][ty+2]    = f_xy;
    s_xz[tx+2][ty]    = f_xz;
    s_yy[tx][ty+1]    = yy[pos];
    s_yz[tx+2][ty+2]  = f_yz;
    s_zz[tx+1][ty]    = zz[pos];
    s_d_1[tx+1][ty+1] = d_1[pos];

    if( ty == 0 ) {   //! y boundary
      s_xy[tx][0] = xy[pos-2*d_yline_1];
      s_xy[tx][1] = xy[pos-d_yline_1];
      s_xy[tx][BLOCKY+2]  = xy[pos+BLOCKY];

      s_yy[tx][0] = yy[pos-d_yline_1];
      s_yy[tx][BLOCKY+1]  = yy[pos+BLOCKY*d_yline_1];
      s_yy[tx][BLOCKY+2]  = yy[pos+(BLOCKY+1)*d_yline_1];

      s_yz[tx+2][0] = yz[pos-2*d_yline_1];
      s_yz[tx+2][1] = yz[pos-d_yline_1];
      s_yz[tx+2][BLOCKY+2] = yz[pos+BLOCKY*d_yline_1];

      s_d_1[tx+1][0] = d_1[pos-d_yline_1];
    }

    if( tx == 0 ) {   //! x boundary
      s_xz[0][ty] = xz[pos-2];
      s_xz[1][ty] = xz[pos-1];
      s_xz[BLOCKX+2][ty]  = xz[pos+BLOCKX];

      s_yz[0][ty+2] = yz[pos-2];
      s_yz[1][ty+2] = yz[pos-1];
      s_yz[BLOCKX+2][ty+2] = yz[pos+BLOCKX];

      s_zz[0][ty] = zz[pos-1];
      s_zz[BLOCKX+1][ty] = zz[pos+BLOCKX];
      s_zz[BLOCKX+2][ty] = zz[pos+BLOCKX+1];

      s_d_1[0][ty+1] = d_1[pos-1];
    }

    if( tx == 0 && ty == 0 ) {
      s_d_1[0][0] = d_1[pos-1-d_yline_1];
    }

    __syncthreads();

    //! d_1 pipeline
    float f_d_1_ip1 = f_d_1;
    f_d_1 = s_d_1[tx+1][ty+1];

    float f_d_1_ik1 = f_d_1_km1;
    f_d_1_km1 = s_d_1[tx][ty+1];

    float f_d_1_ijk = f_d_1_jm1;
    f_d_1_jm1 = s_d_1[tx+1][ty];

    f_dcrj   = dcrjx[i] * f_dcrjy * f_dcrjz;
    //f_d1     = 0.25*(d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
    //f_d2     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
    //f_d3     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);
    float f_d1  = 0.25f * (f_d_1 + f_d_1_jm1 + f_d_1_km1 + s_d_1[tx][ty]);
    float f_d2  = 0.25f * (f_d_1 + f_d_1_ip1 + f_d_1_km1 + f_d_1_ik1);
    float f_d3  = 0.25f * (f_d_1 + f_d_1_ip1 + f_d_1_jm1 + f_d_1_ijk);

    f_d1  = d_dth / f_d1;
    f_d2  = d_dth / f_d2;
    f_d3  = d_dth / f_d3;

    u1[pos]  = (u1[pos] + f_d1 * (d_c1 * (f_xx      - xx_im1)
                        + d_c2 * (xx_ip1            - xx_im2)
                        + d_c1 * (f_xy              - s_xy[tx][ty+1])
                        + d_c2 * (s_xy[tx][ty+3]    - s_xy[tx][ty])
                        + d_c1 * (f_xz              - s_xz[tx+1][ty])
                        + d_c2 * (s_xz[tx+3][ty]    - s_xz[tx][ty]) )) * f_dcrj;

    v1[pos]  = (v1[pos] + f_d2 * (d_c1 * (xy_ip1    - f_xy)
                        + d_c2 * (xy_ip2            - xy_im1)
                        + d_c1 * (s_yy[tx][ty+2]    - s_yy[tx][ty+1])
                        + d_c2 * (s_yy[tx][ty+3]    - s_yy[tx][ty])
                        + d_c1 * (f_yz              - s_yz[tx+1][ty+2])
                        + d_c2 * (s_yz[tx+3][ty+2]  - s_yz[tx][ty+2]) )) * f_dcrj;

    w1[pos]  = (w1[pos] + f_d3 * (d_c1 * (xz_ip1    - f_xz)
                        + d_c2 * (xz_ip2            - xz_im1)
                        + d_c1 * (f_yz              - s_yz[tx+2][ty+1])
                        + d_c2 * (s_yz[tx+2][ty+3]  - s_yz[tx+2][ty])
                        + d_c1 * (s_zz[tx+2][ty]    - s_zz[tx+1][ty])
                        + d_c2 * (s_zz[tx+3][ty]    - s_zz[tx][ty]) )) * f_dcrj;
  }

  return;
}

extern "C"
void dvelcx_H_opt( float *u1,       float *v1,    float *w1,
                   float *xx,       float *yy,    float *zz, float *xy, float *xz, float *yz,
                   float *dcrjx,    float *dcrjy, float *dcrjz,
                   float *d_1,      int nyt,      int nzt,
                   cudaStream_t St, int s_i,      int e_i ) {
  if( (nzt % 64) == 0 || (nyt % 10) == 0 ) {
    const int blockx = 64, blocky = 10;
    dim3 block( blockx, blocky, 1 );
    dim3 grid( nzt / block.x, nyt / block.y, 1 );
    dvelcx_opt< blockx, blocky ><<<grid, block, 0, St>>>( u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_i, e_i );
  } else {
    dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
    dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, (nyt + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
    cudaFuncSetCacheConfig( dvelcx_opt< BLOCK_SIZE_Z, BLOCK_SIZE_Y >, cudaFuncCachePreferL1 );
    dvelcx_opt< BLOCK_SIZE_Z, BLOCK_SIZE_Y ><<<grid, block, 0, St>>>( u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_i, e_i );
  }

  CUCHK( cudaGetLastError() );

  return;
}

extern "C"
void dvelcy_H( float *u1,       float *v1,    float *w1,    float *xx,  float *yy, float *zz, float *xy,   float *xz,   float *yz,
               float *dcrjx,    float *dcrjy, float *dcrjz, float *d_1, int nxt,   int nzt,   float *s_u1, float *s_v1, float *s_w1,
               cudaStream_t St, int s_j,      int e_j,      int rank ) {
  if( rank == -1 )
    return;

  dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
  dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, (nxt + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( dvelcy, cudaFuncCachePreferL1 );
  dvelcy<<<grid, block, 0, St>>>( u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_u1, s_v1, s_w1, s_j, e_j );

  return;
}

extern "C"
void update_bound_y_H( float *u1,   float *v1,  float *w1,  float *f_u1,      float *f_v1,      float *f_w1,  float *b_u1, float *b_v1,
                       float *b_w1, int nxt,    int nzt,    cudaStream_t St1, cudaStream_t St2, int rank_f,   int rank_b ) {
  if( rank_f == -1 && rank_b == -1 )
    return;

  dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
  dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, (nxt + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( update_boundary_y, cudaFuncCachePreferL1 );
  update_boundary_y<<<grid, block, 0, St1>>>( u1, v1, w1, f_u1, f_v1, f_w1, rank_f, Front );
  update_boundary_y<<<grid, block, 0, St2>>>( u1, v1, w1, b_u1, b_v1, b_w1, rank_b, Back );

  return;
}

extern "C"
void dstrqc_H( float *xx,       float *yy,      float *zz,    float *xy,    float *xz,  float *yz,
               float *r1,       float *r2,      float *r3,    float *r4,    float *r5,  float *r6,
               float *u1,       float *v1,      float *w1,    float *lam,   float *mu,  float *qp, float *coeff,
               float *qs,       float *dcrjx,   float *dcrjy, float *dcrjz, int nyt,    int nzt,
               cudaStream_t St, float *lam_mu,  int NX,       int rankx,    int ranky,  int  s_i,
               int e_i,         int s_j,        int e_j ) {
  dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
  dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, (e_j - s_j + 1 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( dstrqc, cudaFuncCachePreferL1 );
  dstrqc<<<grid, block, 0, St>>>( xx,     yy,     zz,   xy,   xz, yz, r1,     r2, r3,     r4,     r5,     r6,
                                  u1,     v1,     w1,   lam,  mu, qp, coeff,  qs, dcrjx,  dcrjy,  dcrjz,  lam_mu, NX,
                                  rankx,  ranky,  s_i,  e_i,  s_j );
  return;
}

template< int BLOCKX, int BLOCKY >
__global__ void
__launch_bounds__( 1024, 1 )
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
            int NX,       int rankx,    int ranky,    int nzt, int s_i, int e_i, int s_j, int e_j ) {
#define SMEM
#define REGQ

  register int    i, j, k, g_i;
  register int    pos, pos_im2, pos_im1;
  register int    pos_km1, pos_kp1, pos_kp2;
  register int    pos_jm1, pos_jp1;
  register int    f_ww;
  register float  vs1, vs2, vs3, a1, tmp, vx1,f_wwo;
  register float  xl, xm, xmu1, xmu2, xmu3;
  register float  qpa, h, h1, h2, h3;
  register float  qpaw, hw, h1w, h2w, h3w;
  register float  f_vx1, f_vx2, f_dcrj, f_r, f_dcrjy, f_dcrjz;
  register float  f_rtmp;
  register float  f_u1, u1_ip1, u1_ip2, u1_im1;
  register float  f_v1, v1_im1, v1_ip1, v1_im2;
  register float  f_w1, w1_im1, w1_im2, w1_ip1;
  float           f_xx, f_yy, f_zz, f_xy, f_xz, f_yz;

#ifdef REGQ
  float mu_i, mu_ip1, lam_i, lam_ip1, qp_i, qp_ip1, qs_i, qs_ip1;
  float mu_jk1, mu_ijk1, lam_jk1, lam_ijk1, qp_jk1, qp_ijk1, qs_jk1, qs_ijk1;
  float mu_jm1, mu_ijk, lam_jm1, lam_ijk, qp_jm1, qp_ijk, qs_jm1, qs_ijk;
  float mu_km1, mu_ik1, lam_km1, lam_ik1, qp_km1, qp_ik1, qs_km1, qs_ik1;
#endif

  int tx = threadIdx.x, ty = threadIdx.y;

  k    = blockIdx.x * blockDim.x + threadIdx.x + ALIGN;
  j    = blockIdx.y * blockDim.y + threadIdx.y + s_j;

  if( k >= nzt + ALIGN || j > e_j )
    return;

#ifdef SMEM
  __shared__ float s_u1[BLOCKX+3][BLOCKY+3], s_v1[BLOCKX+3][BLOCKY+3], s_w1[BLOCKX+3][BLOCKY+3];
  __shared__ float s_mu[BLOCKX+1][BLOCKY+1], s_lam[BLOCKX+1][BLOCKY+1],
                   s_qp[BLOCKX+1][BLOCKY+1], s_qs[BLOCKX+1][BLOCKY+1];
#endif

  i       = e_i;
  pos     = i * d_slice_1+j * d_yline_1 + k;

  u1_ip1  = u1[pos+d_slice_2];
  f_u1    = u1[pos+d_slice_1];
  u1_im1  = u1[pos];
  f_v1    = v1[pos+d_slice_1];
  v1_im1  = v1[pos];
  v1_im2  = v1[pos-d_slice_1];
  f_w1    = w1[pos+d_slice_1];
  w1_im1  = w1[pos];
  w1_im2  = w1[pos-d_slice_1];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

#ifdef REGQ
  mu_i    = mu [pos+d_slice_1];
  lam_i   = lam[pos+d_slice_1];
  qp_i    = qp [pos+d_slice_1];
  qs_i    = qs [pos+d_slice_1];

  mu_jk1  = mu [pos+d_slice_1-d_yline_1-1];
  lam_jk1 = lam[pos+d_slice_1-d_yline_1-1];
  qp_jk1  = qp [pos+d_slice_1-d_yline_1-1];
  qs_jk1  = qs [pos+d_slice_1-d_yline_1-1];

  mu_jm1  = mu [pos+d_slice_1-d_yline_1];
  lam_jm1 = lam[pos+d_slice_1-d_yline_1];
  qp_jm1  = qp [pos+d_slice_1-d_yline_1];
  qs_jm1  = qs [pos+d_slice_1-d_yline_1];

  mu_km1  = mu [pos+d_slice_1-1];
  lam_km1 = lam[pos+d_slice_1-1];
  qp_km1  = qp [pos+d_slice_1-1];
  qs_km1  = qs [pos+d_slice_1-1];
#endif

  for( i = e_i; i >= s_i; i-- ) {
    // f_vx1    = tex1Dfetch(p_vx1, pos);
    // f_vx2    = tex1Dfetch(p_vx2, pos);
    // f_ww     = tex1Dfetch(p_ww, pos);
    // f_wwo     = tex1Dfetch(p_wwo, pos);
    f_vx1     = d_vx1[pos];
    f_vx2     = d_vx2[pos];
    f_ww      = d_ww[pos];
    f_wwo     = d_wwo[pos];

#ifdef REGQ
    mu_ip1    = mu_i;
    lam_ip1   = lam_i;
    qp_ip1    = qp_i;
    qs_ip1    = qs_i;
    mu_ijk1   = mu_jk1;
    lam_ijk1  = lam_jk1;
    qp_ijk1   = qp_jk1;
    qs_ijk1   = qs_jk1;
    mu_ijk    = mu_jm1;
    lam_ijk   = lam_jm1;
    qp_ijk    = qp_jm1;
    qs_ijk    = qs_jm1;
    mu_ik1    = mu_km1;
    lam_ik1   = lam_km1;
    qp_ik1    = qp_km1;
    qs_ik1    = qs_km1;

    __syncthreads();

    s_mu[tx+1][ty+1]  = mu [pos];
    s_lam[tx+1][ty+1] = lam[pos];
    s_qp[tx+1][ty+1]  = qp [pos];
    s_qs[tx+1][ty+1]  = qs [pos];

    if( ty == 0 ) {
      s_mu[tx+1][0]   = mu [pos-d_yline_1];
      s_lam[tx+1][0]  = lam[pos-d_yline_1];
      s_qp[tx+1][0]   = qp [pos-d_yline_1];
      s_qs[tx+1][0]   = qs [pos-d_yline_1];
    }

    if( tx == 0 ) {
      s_mu[0][ty+1]   = mu [pos-1];
      s_lam[0][ty+1]  = lam[pos-1];
      s_qp[0][ty+1]   = qp [pos-1];
      s_qs[0][ty+1]   = qs [pos-1];
    }

    if( tx == 0 && ty == 0 ) {
      s_mu[0][0]      = mu [pos-d_yline_1-1];
      s_lam[0][0]     = lam[pos-d_yline_1-1];
      s_qp[0][0]      = qp [pos-d_yline_1-1];
      s_qs[0][0]      = qs [pos-d_yline_1-1];
    }

    __syncthreads();

    mu_i    = s_mu [tx+1][ty+1];
    lam_i   = s_lam[tx+1][ty+1];
    qp_i    = s_qp [tx+1][ty+1];
    qs_i    = s_qs [tx+1][ty+1];
    mu_jk1  = s_mu [tx][ty];
    lam_jk1 = s_lam[tx][ty];
    qp_jk1  = s_qp [tx][ty];
    qs_jk1  = s_qs [tx][ty];
    mu_jm1  = s_mu [tx+1][ty];
    lam_jm1 = s_lam[tx+1][ty];
    qp_jm1  = s_qp [tx+1][ty];
    qs_jm1  = s_qs [tx+1][ty];
    mu_km1  = s_mu [tx][ty+1];
    lam_km1 = s_lam[tx][ty+1];
    qp_km1  = s_qp [tx][ty+1];
    qs_km1  = s_qs [tx][ty+1];

    // mu_i    = LDG(mu [pos]);
    // lam_i   = LDG(lam[pos]);
    // qp_i    = LDG(qp [pos]);
    // qs_i    = LDG(qs [pos]);
    // mu_jk1  = LDG(mu [pos-d_yline_1-1]);
    // lam_jk1 = LDG(lam[pos-d_yline_1-1]);
    // qp_jk1  = LDG(qp [pos-d_yline_1-1]);
    // qs_jk1  = LDG(qs [pos-d_yline_1-1]);
    // mu_jm1  = LDG(mu [pos-d_yline_1]);
    // lam_jm1 = LDG(lam[pos-d_yline_1]);
    // qp_jm1  = LDG(qp [pos-d_yline_1]);
    // qs_jm1  = LDG(qs [pos-d_yline_1]);
    // mu_km1  = LDG(mu [pos-1]);
    // lam_km1 = LDG(lam[pos-1]);
    // qp_km1  = LDG(qp [pos-1]);
    // qs_km1  = LDG(qs [pos-1]);
#endif

/*
  if(f_wwo!=f_wwo){
  xx[pos] = yy[pos] = zz[pos] = xy[pos] = xz[pos] = yz[pos] = 1.0;
  r1[pos] = r2[pos] = r3[pos] = r4[pos] = r5[pos] = r6[pos] = 1.0;
  return;
  }
*/
    f_dcrj    = dcrjx[i] * f_dcrjy * f_dcrjz;

#ifndef REGQ
    register int pos_ip1  = pos + d_slice_1;
    register int pos_ik1  = pos + d_slice_1 - 1;
    register int pos_jk1  = pos - d_yline_1 - 1;
    register int pos_ijk  = pos + d_slice_1 - d_yline_1;
    register int pos_ijk1 = pos + d_slice_1 - d_yline_1 - 1;
#endif

#ifndef SMEM
    register int pos_km2  = pos - 2;
    register int pos_jm2  = pos - d_yline_2;
    register int pos_jp2  = pos + d_yline_2;
#endif

    pos_km1   = pos - 1;
    pos_kp1   = pos + 1;
    pos_kp2   = pos + 2;
    pos_jm1   = pos - d_yline_1;
    pos_jp1   = pos + d_yline_1;
    pos_im2   = pos - d_slice_2;
    pos_im1   = pos - d_slice_1;

#ifdef REGQ
    xl        = 8.0f / (lam_i + lam_ip1 + lam_jm1 + lam_ijk
                      + lam_km1 + lam_ik1 + lam_jk1 + lam_ijk1);
    xm        = 16.0f / (mu_i + mu_ip1 + mu_jm1 + mu_ijk
                      + mu_km1 + mu_ik1 + mu_jk1 + mu_ijk1);
    xmu1      = 2.0f / (mu_i + mu_km1);
    xmu2      = 2.0f / (mu_i + mu_jm1);
    xmu3      = 2.0f / (mu_i + mu_ip1);
    xl        = xl + xm;
    qpa       = 0.0625f * (qp_i + qp_ip1 + qp_jm1 + qp_ijk
                        + qp_km1 + qp_ik1 + qp_jk1 + qp_ijk1);
#else
    xl        = 8.0f / (LDG( lam[pos] ) + LDG( lam[pos_ip1] ) + LDG( lam[pos_jm1] ) + LDG( lam[pos_ijk] )
                      + LDG( lam[pos_km1] ) + LDG( lam[pos_ik1] ) + LDG( lam[pos_jk1] ) + LDG( lam[pos_ijk1] ));
    xm        = 16.0f / (LDG( mu[pos] ) + LDG( mu[pos_ip1] ) + LDG( mu[pos_jm1] ) + LDG( mu[pos_ijk] )
                      + LDG( mu[pos_km1] ) + LDG( mu[pos_ik1] ) + LDG( mu[pos_jk1] ) + LDG( mu[pos_ijk1] ));
    xmu1      = 2.0f / (LDG( mu[pos] ) + LDG( mu[pos_km1] ));
    xmu2      = 2.0f / (LDG( mu[pos] ) + LDG( mu[pos_jm1] ));
    xmu3      = 2.0f / (LDG( mu[pos] ) + LDG( mu[pos_ip1] ));
    xl        = xl + xm;
    qpa       = 0.0625f * (LDG( qp[pos] ) + LDG( qp[pos_ip1] ) + LDG( qp[pos_jm1] ) + LDG( qp[pos_ijk] )
                        + LDG( qp[pos_km1] ) + LDG( qp[pos_ik1] ) + LDG( qp[pos_jk1] ) + LDG( qp[pos_ijk1] ));
#endif

//                        www=f_ww;
    if( 1.0f / (qpa * 2.0f) <= 200.0f ) {
//      printf("coeff[f_ww*2-2] %g\n",coeff[f_ww*2-2]);
      qpaw  = coeff[f_ww*2-2] * (2. * qpa) * (2. * qpa) + coeff[f_ww*2-1] * (2.*qpa);
//              qpaw=coeff[www*2-2]*(2.*qpa)*(2.*qpa)+coeff[www*2-1]*(2.*qpa);
//                qpaw=qpaw/2.;
    } else {
      qpaw  = f_wwo * qpa;
    }
//                 printf("qpaw %f\n",qpaw);
//              printf("qpaw1 %g\n",qpaw);
    qpaw    = qpaw / f_wwo;
//      printf("qpaw2 %g\n",qpaw);

#ifdef REGQ
    h       = 0.0625f * (qs_i + qs_ip1 + qs_jm1 + qs_ijk
                      + qs_km1 + qs_ik1 + qs_jk1 + qs_ijk1);
#else
    h       = 0.0625f * (LDG( qs[pos] ) + LDG( qs[pos_ip1] ) + LDG( qs[pos_jm1] ) + LDG( qs[pos_ijk] )
                      + LDG( qs[pos_km1] ) + LDG( qs[pos_ik1] ) + LDG( qs[pos_jk1] ) + LDG( qs[pos_ijk1] ));
#endif

    if( 1.0f / (h * 2.0f) <= 200.0f ) {
      hw    = coeff[f_ww*2-2] * (2.0f * h) * (2.0f * h) + coeff[f_ww*2-1] * (2.0f * h);
      //                  hw=hw/2.0f;
    } else {
      hw    = f_wwo * h;
    }
    hw      = hw / f_wwo;

    // h1       = 0.250f*(  qs[pos]     + qs[pos_km1] );
    h1      = 0.25f * (qs_i + qs_km1);

    if( 1.0f / (h1 * 2.0f) <= 200.0f ) {
      h1w   = coeff[f_ww*2-2] * (2.0f * h1) * (2.0f * h1) + coeff[f_ww*2-1] * (2.0f * h1);
      //                  h1w=h1w/2.0f;
    } else {
      h1w   = f_wwo * h1;
    }
    h1w     = h1w / f_wwo;

    // h2       = 0.250f*(  qs[pos]     + qs[pos_jm1] );
    h2    = 0.25f * (qs_i + qs_jm1);
    if( 1.0f / (h2 * 2.0f) <= 200.0f ) {
      h2w   = coeff[f_ww*2-2] * (2.0f * h2) * (2.0f * h2) + coeff[f_ww*2-1] * (2.0f * h2);
      //                  h2w=h2w/2.;
    } else {
      h2w   = f_wwo * h2;
    }
    h2w     = h2w / f_wwo;

    // h3       = 0.250f*(  qs[pos]     + qs[pos_ip1] );
    h3      = 0.25f * (qs_i + qs_ip1);
    if( 1.0f / (h3 * 2.0f) <= 200.0f ) {
      h3w   = coeff[f_ww*2-2] * (2.0f * h3) * (2.0f * h3) + coeff[f_ww*2-1] * (2.0f * h3);
      //                  h3w=h3w/2.0f;
    } else {
      h3w   = f_wwo * h3;
    }
    h3w     = h3w / f_wwo;

    h       = -xm * hw * d_dh1;
    h1      = -xmu1 * h1w * d_dh1;
    h2      = -xmu2 * h2w * d_dh1;
    h3      = -xmu3 * h3w * d_dh1;

    //        h1       = -xmu1*hw1*d_dh1;
    //h2       = -xmu2*hw2*d_dh1;
    //h3       = -xmu3*hw3*d_dh1;

    qpa     = -qpaw * xl * d_dh1;
    //        qpa      = -qpaw*xl*d_dh1;

    xm      = xm * d_dth;
    xmu1    = xmu1 * d_dth;
    xmu2    = xmu2 * d_dth;
    xmu3    = xmu3 * d_dth;
    xl      = xl * d_dth;
    //  f_vx2    = f_vx2*f_vx1;
    h       = h * f_vx1;
    h1      = h1 * f_vx1;
    h2      = h2 * f_vx1;
    h3      = h3 * f_vx1;
    qpa     = qpa * f_vx1;

    xm      = xm + d_DT * h;
    xmu1    = xmu1 + d_DT * h1;
    xmu2    = xmu2 + d_DT * h2;
    xmu3    = xmu3 + d_DT * h3;
    vx1     = d_DT * (1 + f_vx2 * f_vx1);

    u1_ip2  = u1_ip1;
    u1_ip1  = f_u1;
    f_u1    = u1_im1;
    u1_im1  = u1[pos_im1];
    v1_ip1  = f_v1;
    f_v1    = v1_im1;
    v1_im1  = v1_im2;
    v1_im2  = v1[pos_im2];
    w1_ip1  = f_w1;
    f_w1    = w1_im1;
    w1_im1  = w1_im2;
    w1_im2  = w1[pos_im2];

    if( k == d_nzt + ALIGN - 1 ) {
      u1[pos_kp1] = f_u1 - (f_w1 - w1_im1);
      v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);

      g_i   = d_nxt * rankx + i - 4 * LOOP - 1;

      if( g_i < NX )
        vs1 = u1_ip1 - (w1_ip1 - f_w1);
      else
        vs1 = 0.0f;

      g_i   = d_nyt * ranky + j - 4 * LOOP - 1;
      if( g_i > 1 )
        vs2 = v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
      else
        vs2 = 0.0f;

      w1[pos_kp1] = w1[pos_km1] - lam_mu[i*(d_nyt+4+8*LOOP)+j] * ((vs1 - u1[pos_kp1]) + (u1_ip1 - f_u1)
                                                          + (v1[pos_kp1] - vs2) + (f_v1 - v1[pos_jm1]));
    } else if( k == d_nzt + ALIGN - 2 ) {
      u1[pos_kp2] = u1[pos_kp1] - (w1[pos_kp1] - w1[pos_im1+1]);
      v1[pos_kp2] = v1[pos_kp1] - (w1[pos_jp1+1] - w1[pos_kp1]);
    }

#ifdef SMEM
    __threadfence_block();
    __syncthreads();

    s_u1[tx+1][ty+1]  = u1[pos];
    s_v1[tx+1][ty+2]  = v1[pos];
    s_w1[tx+2][ty+1]  = w1[pos];

    if( tx == 0 ) {   //! k halo
      s_u1[0][ty+1]         = u1[pos-1];
      s_u1[BLOCKX+1][ty+1]  = u1[pos+BLOCKX];
      s_u1[BLOCKX+2][ty+1]  = u1[pos+BLOCKX+1];

      s_v1[0][ty+2]         = v1[pos-1];
      s_v1[BLOCKX+1][ty+2]  = v1[pos+BLOCKX];
      s_v1[BLOCKX+2][ty+2]  = v1[pos+BLOCKX+1];

      s_w1[0][ty+1]  = w1[pos-2];
      s_w1[1][ty+1]  = w1[pos-1];
      s_w1[BLOCKX+2][ty+1]  = w1[pos+BLOCKX];
    }

    if( ty == 0 ) {   //! j halo
      s_u1[tx+1][0]         = u1[pos-d_yline_1];
      s_u1[tx+1][BLOCKY+1]  = u1[pos+BLOCKY*d_yline_1];
      s_u1[tx+1][BLOCKY+2]  = u1[pos+(BLOCKY+1)*d_yline_1];

      s_v1[tx+1][0]         = v1[pos-2*d_yline_1];
      s_v1[tx+1][1]         = v1[pos-d_yline_1];
      s_v1[tx+1][BLOCKY+2]  = v1[pos+BLOCKY*d_yline_1];

      s_w1[tx+2][0]         = w1[pos-d_yline_1];
      s_w1[tx+2][BLOCKY+1]  = w1[pos+BLOCKY*d_yline_1];
      s_w1[tx+2][BLOCKY+2]  = w1[pos+(BLOCKY+1)*d_yline_1];
    }

    __syncthreads();
#endif

    vs1     = d_c1 * (u1_ip1 - f_u1) + d_c2 * (u1_ip2 - u1_im1);

#ifdef SMEM
    vs2     = d_c1 * (f_v1 - s_v1[tx+1][ty+1]) +
              d_c2 * (s_v1[tx+1][ty+3] - s_v1[tx+1][ty]);
    vs3     = d_c1 * (f_w1 - s_w1[tx+1][ty+1]) +
              d_c2 * (s_w1[tx+3][ty+1] - s_w1[tx][ty+1]);
#else
    vs2     = d_c1 * (f_v1 - v1[pos_jm1]) + d_c2 * (v1[pos_jp1] - v1[pos_jm2]);
    vs3     = d_c1 * (f_w1 - w1[pos_km1]) + d_c2 * (w1[pos_kp1] - w1[pos_km2]);
#endif

    tmp     = xl * (vs1 + vs2 + vs3);
    a1      = qpa * (vs1 + vs2 + vs3);
    tmp     = tmp + d_DT * a1;

    f_r     = r1[pos];
    f_rtmp  = -h * (vs2 + vs3) + a1;
    f_xx    = xx[pos] + tmp - xm * (vs2 + vs3) + vx1 * f_r;
    r1[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp  = f_rtmp * (f_wwo - 1.0f) + f_vx2 * f_r * (1.0f - f_vx1);
    xx[pos] = (f_xx + d_DT * f_rtmp) * f_dcrj;

    f_r     = r2[pos];
    f_rtmp  = -h * (vs1 + vs3) + a1;
    f_yy    = (yy[pos] + tmp - xm * (vs1 + vs3) + vx1 * f_r) * f_dcrj;

    r2[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp  = f_rtmp * (f_wwo - 1.0f) + f_vx2 * f_r * (1.0f - f_vx1);
    yy[pos] = (f_yy + d_DT * f_rtmp) * f_dcrj;

    f_r     = r3[pos];
    f_rtmp  = -h * (vs1 + vs2) + a1;
    f_zz    = (zz[pos] + tmp - xm * (vs1 + vs2) + vx1 * f_r) * f_dcrj;
    r3[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp  = f_rtmp * (f_wwo - 1.0f) + f_vx2 * f_r * (1.0f - f_vx1);
    zz[pos] = (f_zz + d_DT * f_rtmp) * f_dcrj;

#ifdef SMEM
    vs1     = d_c1 * (s_u1[tx+1][ty+2] - f_u1) +
              d_c2 * (s_u1[tx+1][ty+3] - s_u1[tx+1][ty]);
#else
    vs1     = d_c1 * (u1[pos_jp1] - f_u1) + d_c2 * (u1[pos_jp2] - u1[pos_jm1]);
#endif

    vs2     = d_c1 * (f_v1 - v1_im1) + d_c2 * (v1_ip1 - v1_im2);
    f_r     = r4[pos];
    f_rtmp  = h1 * (vs1 + vs2);
    f_xy    = xy[pos] + xmu1 * (vs1 + vs2) + vx1 * f_r;
    r4[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp  = f_rtmp * (f_wwo - 1.0f) + f_vx2 * f_r * (1.0f - f_vx1);
    xy[pos] = (f_xy + d_DT * f_rtmp) * f_dcrj;

    //moved to separate subroutine fstr, to be executed after plasticity (Daniel)
    /*if(k == d_nzt+ALIGN-1)
      {
      zz[pos+1] = -zz[pos];
      xz[pos]   = 0.0f;
      yz[pos]   = 0.0f;
      }
      else
      {*/

#ifdef SMEM
    vs1     = d_c1 * (s_u1[tx+2][ty+1] - f_u1) +
              d_c2 * (s_u1[tx+3][ty+1] - s_u1[tx][ty+1]);
#else
    vs1     = d_c1 * (u1[pos_kp1] - f_u1) + d_c2 * (u1[pos_kp2] - u1[pos_km1]);
#endif

    vs2     = d_c1 * (f_w1 - w1_im1) + d_c2 * (w1_ip1 - w1_im2);
    f_r     = r5[pos];
    f_rtmp  = h2 * (vs1 + vs2);
    f_xz    = xz[pos] + xmu2 * (vs1 + vs2) + vx1 * f_r;
    r5[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp  = f_rtmp * (f_wwo - 1.0f) + f_vx2 * f_r * (1.0f - f_vx1);
    xz[pos] = (f_xz + d_DT * f_rtmp) * f_dcrj;

#ifdef SMEM
    vs2     = d_c1 * (s_w1[tx+2][ty+2] - f_w1) +
              d_c2 * (s_w1[tx+2][ty+3] - s_w1[tx+2][ty]);
    vs1     = d_c1 * (s_v1[tx+2][ty+2] - f_v1) +
              d_c2 * (s_v1[tx+3][ty+2] - s_v1[tx][ty+2]);
#else
    vs1     = d_c1 * (v1[pos_kp1] - f_v1) + d_c2 * (v1[pos_kp2] - v1[pos_km1]);
    vs2     = d_c1 * (w1[pos_jp1] - f_w1) + d_c2 * (w1[pos_jp2] - w1[pos_jm1]);
#endif

    f_r     = r6[pos];
    f_rtmp  = h3 * (vs1 + vs2);
    f_yz    = yz[pos] + xmu3 * (vs1 + vs2) + vx1 * f_r;
    r6[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp  = f_rtmp * (f_wwo - 1.0f) + f_vx2 * f_r * (1.0f - f_vx1);
    yz[pos] = (f_yz + d_DT * f_rtmp) * f_dcrj;

    // also moved to fstr (Daniel)
    /*if(k == d_nzt+ALIGN-2)
      {
      zz[pos+3] = -zz[pos];
      xz[pos+2] = -xz[pos];
      yz[pos+2] = -yz[pos];
      }
      else if(k == d_nzt+ALIGN-3)
      {
      xz[pos+4] = -xz[pos];
      yz[pos+4] = -yz[pos];
      }*/
    /*}*/
    pos     = pos_im1;
  }

  return;
}

extern "C"
void dstrqc_H_new( float *xx,       float *yy,    float *zz,    float *xy,    float *xz,  float *yz,
                   float *r1,       float *r2,    float *r3,    float *r4,    float *r5,  float *r6,
                   float *u1,       float *v1,    float *w1,    float *lam,   float *mu,  float *qp, float *coeff,
                   float *qs,       float *dcrjx, float *dcrjy, float *dcrjz, int nyt,    int nzt,
                   cudaStream_t St, float *lam_mu,
                   //float *vx1, float *vx2, float *ww, float *wwo, //peng's version
                   float *vx1,      float *vx2,   int *ww,      float *wwo,
                   int NX,          int rankx,    int ranky,    int  s_i,
                   int e_i,         int s_j,      int e_j ) {
  if( (nzt % 64) == 0 && ((e_j - s_j + 1) % 8) == 0 ) {
    const int blockx = 64, blocky = 8;
    dim3 block( blockx, blocky, 1 );
    dim3 grid( (nzt + block.x - 1) / block.x, (e_j - s_j + 1 + block.y - 1) / block.y, 1 );
    CUCHK( cudaFuncSetCacheConfig( dstrqc_new< blockx, blocky >, cudaFuncCachePreferShared ) );
    dstrqc_new< blockx, blocky ><<<grid, block, 0, St>>>( xx,   yy,     zz,     xy,   xz,   yz,   r1,     r2, r3,     r4,     r5,     r6,
                                                          u1,   v1,     w1,     lam,  mu,   qp,   coeff,  qs, dcrjx,  dcrjy,  dcrjz,  lam_mu,
                                                          vx1,  vx2,    ww,     wwo,
                                                          NX,   rankx,  ranky,  nzt,  s_i,  e_i,  s_j,    e_j );
  } else {
    const int blockx = BLOCK_SIZE_Z, blocky = BLOCK_SIZE_Y;
    dim3 block( blockx, blocky, 1 );
    dim3 grid( (nzt + block.x - 1) / block.x, (e_j - s_j + 1 + block.y - 1) / block.y, 1 );
    CUCHK( cudaFuncSetCacheConfig( dstrqc_new<blockx,blocky>, cudaFuncCachePreferShared ) );
    dstrqc_new< blockx, blocky ><<<grid, block, 0, St>>>( xx,   yy,     zz,     xy,   xz,   yz,   r1,     r2, r3,     r4,     r5,     r6,
                                                          u1,   v1,     w1,     lam,  mu,   qp,   coeff,  qs, dcrjx,  dcrjy,  dcrjz,  lam_mu,
                                                          vx1,  vx2,    ww,     wwo,
                                                          NX,   rankx,  ranky,  nzt,  s_i,  e_i,  s_j,    e_j );
  }

  CUCHK( cudaGetLastError() );

  return;
}

/* kernel function to apply free-surface B.C. to stresses - (Daniel) */
extern "C"
void fstr_H( float* zz, float* xz, float* yz, cudaStream_t St, int s_i, int e_i, int s_j, int e_j ) {
  dim3 block( 2, BLOCK_SIZE_Y, 1 );
  dim3 grid( 1, (e_j - s_j + 1 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( fstr, cudaFuncCachePreferL1 );
  fstr<<<grid, block, 0, St>>>( zz, xz, yz, s_i, e_i, s_j );

  return;
}

template< int BLOCKX,int BLOCKY >
__global__ void
__launch_bounds__( 640, 2 )
drprecpc_calc_opt( float *xx,     float *yy,    float *zz,
                   const float *__restrict__ xy,
                   const float *__restrict__ xz,
                   const float *__restrict__ yz,
                   float *mu,     float *d1,
                   float *sigma2,
                   float *yldfac, float *cohes, float *phi,
                   float *neta,
                   int nzt,       int s_i,      int e_i, int s_j, int e_j ) {
  register int    i, j, k, pos;
//  register int    jj, kk;
  register int    pos_im1, pos_jm1, pos_km1;
//  register int pos_ip1, pos_ip1jm1;
//  register int pos_ip1km1, pos_jm1km1;

  register float  Sxx, Syy, Szz, Sxy, Sxz, Syz;
  register float  Sxxp, Syyp, Szzp, Sxyp, Sxzp, Syzp;
  register float  depxx, depyy, depzz, depxy, depxz, depyz;
  register float  SDxx, SDyy, SDzz;
  register float  iyldfac, Tv, sigma_m, taulim, taulim2, rphi;
  register float  xm, iixx, iiyy, iizz;
  register float  secinv, sqrtSecinv;
//  register float  mu_;

  //! Compute initial stress on GPU (Daniel)
  register float  ini[9], ini_ip1[9];
  register float  depth, pfluid;
  register int    srfpos;
  __shared__ float s_yz[BLOCKX+1][BLOCKY+1];
  int tx = threadIdx.x, ty = threadIdx.y;

  k    = blockIdx.x * blockDim.x + threadIdx.x + ALIGN;
  j    = blockIdx.y * blockDim.y + threadIdx.y + s_j;

  if( k >= nzt + ALIGN || j > e_j )
    return;

  i       = e_i;
  pos     = i * d_slice_1 + j * d_yline_1 + k;

//  kk      = k - ALIGN;
//  jj      = j - (2 + 4 * LOOP);

  srfpos  = d_nzt + ALIGN - 1;
  depth   = (float) (srfpos - k) * d_DH;

  if(depth > 0)
    pfluid = (depth + d_DH * 0.5) * 9.81e3;
  else
    pfluid = d_DH / 2. * 9.81e3;

  //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k, depth, pfluid);

//  float yz_ip1, yz_i;
  float sigma2_ip1, sigma2_i;
  float xy_ip1, xy_i, xz_ip1, xz_i;
  float mu_ip1, mu_i;
  float xz_km1, xz_ip1km1, xy_jm1, xy_ip1jm1;

  sigma2_i  = sigma2[pos+d_slice_1];
  xy_i      = xy[pos+d_slice_1];
  xz_i      = xz[pos+d_slice_1];
  mu_i      = mu[pos+d_slice_1];
  xz_km1    = xz[pos+d_slice_1-1];
  xy_jm1    = xy[pos+d_slice_1-d_yline_1];

  for( i = e_i; i >= s_i; --i ) {
    sigma2_ip1  = sigma2_i;
    xy_ip1      = xy_i;
    xz_ip1      = xz_i;
    mu_ip1      = mu_i;
    xz_ip1km1   = xz_km1;
    xy_ip1jm1   = xy_jm1;

    pos_im1     = pos - d_slice_1;
//    pos_ip1     = pos + d_slice_1;
    pos_jm1     = pos - d_yline_1;
    pos_km1     = pos - 1;
//    pos_ip1jm1  = pos_ip1 - d_yline_1;
//    pos_ip1km1  = pos_ip1 - 1;
//    pos_jm1km1  = pos_jm1 - 1;

    sigma2_i    = sigma2[pos];
    xy_i        = xy[pos];
    xy_jm1      = xy[pos_jm1];
    xz_i        = xz[pos];
    xz_km1      = xz[pos_km1];
    mu_i        = mu[pos];

    // mu_ = mu[pos];

    //! start drprnn
    rotate_principal( sigma2_i, pfluid, ini );
    rotate_principal( sigma2_ip1, pfluid, ini_ip1 );
    /*cuPrintf("ini[8] = %5.2e, ini[4]=%5.2e sigma2=%5.2e pfluid=%5.2e\n",
         ini[8], ini[4], sigma2[pos], pfluid);*/
    /*iixx  = 0.5f*(inixx_i + inixx_ip1);
    iiyy  = 0.5f*(iniyy_i + iniyy_ip1);
    iizz  = 0.5f*(inizz_i + inizz_ip1);*/
    iixx  = 0.5f * (ini[0] + ini_ip1[0]);
    iiyy  = 0.5f * (ini[4] + ini_ip1[4]);
    iizz  = 0.5f * (ini[8] + ini_ip1[8]);

    __syncthreads();

    s_yz[tx+1][ty+1] = yz[pos];
    if( ty == 0 ) {
      s_yz[tx+1][0] = yz[pos-d_yline_1];
    }

    if( tx == 0 ) {
      s_yz[0][ty+1] = yz[pos-1];
    }

    if( tx == 0 && ty == 0 ) {
      s_yz[0][0] = yz[pos-1-d_yline_1];
    }

    __syncthreads();

    Sxx   = xx[pos] + iixx;
    Syy   = yy[pos] + iiyy;
    Szz   = zz[pos] + iizz;
    // Sxz   = 0.25f*(xz_i + xz_ip1 + xz[pos_km1] + xz[pos_ip1km1])
    //   //+ 0.5f*(inixz_i + inixz_ip1);
    //           + 0.5f*(ini[2] + ini_ip1[2]);
    // Syz   = 0.25f*(yz[pos] + yz[pos_jm1] + yz[pos_km1] + yz[pos_jm1km1])
    //   //+ 0.5f*(iniyz_i + iniyz_ip1);
    //           + 0.5f*(ini[5] + ini_ip1[5]);
    // Sxy   = 0.25f*(xy_i + xy_ip1 + xy[pos_jm1] + xy[pos_ip1jm1])
    //   //+ 0.5f*(inixy_i + inixy_ip1);
    //           + 0.5f*(ini[1] + ini_ip1[1]);
    Sxz   = 0.25f * (xz_i + xz_ip1 + xz_km1 + xz_ip1km1)
            + 0.5f * (ini[2] + ini_ip1[2]);
    Syz   = 0.25f * (s_yz[tx+1][ty+1] + s_yz[tx+1][ty] + s_yz[tx][ty+1] + s_yz[tx][ty])
            + 0.5f * (ini[5] + ini_ip1[5]);
    Sxy   = 0.25f * (xy_i + xy_ip1 + xy_jm1 + xy_ip1jm1)
            + 0.5f * (ini[1] + ini_ip1[1]);

    Tv = d_DH / sqrt( 1.0f / (mu_i * d1[pos]) );

    Sxxp  = Sxx;
    Syyp  = Syy;
    Szzp  = Szz;
    Sxyp  = Sxy;
    Sxzp  = Sxz;
    Syzp  = Syz;

    //! drucker_prager function:
    rphi        = phi[pos] * 0.017453292519943295f;
    // rphi = 45.f * 0.017453292519943295f;
    sigma_m     = (Sxx + Syy + Szz) / 3.0f;
    SDxx        = Sxx - sigma_m;
    SDyy        = Syy - sigma_m;
    SDzz        = Szz - sigma_m;
    secinv      = 0.5f * (SDxx * SDxx + SDyy * SDyy + SDzz * SDzz)
                  + Sxz * Sxz + Sxy * Sxy + Syz * Syz;
    sqrtSecinv  = sqrt( secinv );
    taulim2     = cohes[pos] * cos( rphi ) - (sigma_m + pfluid) * sin( rphi );
    // taulim2 = 0.f*cos(rphi) - (sigma_m + pfluid)*sin(rphi);

    if( taulim2 > 0.0f )
      taulim    = taulim2;
    else
      taulim    = 0.0f;

    if( sqrtSecinv > taulim ) {
      iyldfac   = taulim / sqrtSecinv
                  + (1.0f - taulim / sqrtSecinv) * exp( -d_DT / Tv );
      Sxx       = SDxx * iyldfac + sigma_m;
      Syy       = SDyy * iyldfac + sigma_m;
      Szz       = SDzz * iyldfac + sigma_m;
      Sxz       = Sxz * iyldfac;
      Sxy       = Sxy * iyldfac;
      Syz       = Syz * iyldfac;
    } else
      iyldfac   = 1.0f;

    yldfac[pos] = iyldfac;
    //! end drucker_prager function

    if( yldfac[pos] < 1.0f ) {
      xm        = 2.0f / (mu_i + mu_ip1);
      depxx     = (Sxx - Sxxp) / xm;
      depyy     = (Syy - Syyp) / xm;
      depzz     = (Szz - Szzp) / xm;
      depxy     = (Sxy - Sxyp) / xm;
      depxz     = (Sxz - Sxzp) / xm;
      depyz     = (Syz - Syzp) / xm;

      neta[pos] = neta[pos] + sqrt( 0.5f * (depxx * depxx + depyy * depyy + depzz * depzz) +
                                    2.0f * (depxy * depxy + depxz * depxz + depyz * depyz) );
    } else
      yldfac[pos] = 1.0f;
    //! end drprnn

// DEBUG if neta/EPxx etc are set
    //neta[pos] = 1.E+2;
    //EPxx[pos] = 1.E+2;
// DEBUG - end

    pos = pos_im1;
  }

  return;
}

//! drprecpc is for plasticity computation for cerjan and wave propagation
extern "C"
void drprecpc_calc_H_opt( float *xx,      float *yy,    float *zz,  float *xy,  float *xz, float *yz,
                          float *mu,      float *d1,    float *sigma2,
                          float *yldfac,  float *cohes, float *phi,
                          float *neta,
                          int nzt,
                          int xls,        int xre,      int yls,    int yre,    cudaStream_t St ) {
  if( (nzt % 64) == 0 && ((yre - yls + 1) % 10) == 0 ) {
    const int blockx = 64, blocky = 10;
    dim3 block( blockx, blocky, 1 );
    dim3 grid( nzt / blockx, (yre - yls + 1) / blocky, 1 );

    //!split into tho routines, one for the normal, one for shear stress components (Daniel)
    drprecpc_calc_opt<blockx,blocky><<<grid, block, 0, St>>>( xx, yy, zz, xy, xz, yz, mu, d1,
                                                              sigma2, yldfac, cohes, phi, neta,
                                                              nzt, xls, xre, yls, yre );
  } else {
    dim3 block(BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
    dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, ((yre - yls + 1) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
    cudaFuncSetCacheConfig( drprecpc_calc_opt< BLOCK_SIZE_Z,BLOCK_SIZE_Y >, cudaFuncCachePreferL1 );
    drprecpc_calc_opt< BLOCK_SIZE_Z,BLOCK_SIZE_Y ><<<grid, block, 0, St>>>( xx, yy, zz, xy, xz, yz, mu, d1,
                                                                            sigma2, yldfac, cohes, phi, neta,
                                                                            nzt, xls, xre, yls, yre );
  }

  CUCHK( cudaGetLastError() );

  return;
}

//! drprecpc is for plasticity computation for cerjan and wave propagation
extern "C"
void drprecpc_calc_H( float *xx,      float *yy,    float *zz,  float *xy,  float *xz, float *yz,
                      float *mu,      float *d1,    float *sigma2,
                      float *yldfac,  float *cohes, float *phi, float *neta,
                      int nzt,
                      int xls,        int xre,      int yls,    int yre,    cudaStream_t St ) {
  dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
  dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, ((yre - yls + 1) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( drprecpc_calc, cudaFuncCachePreferL1 );

  //! split into tho routines, one for the normal, one for shear stress components (Daniel)
  drprecpc_calc<<<grid, block, 0, St>>>( xx, yy, zz, xy, xz, yz, mu, d1,
                                         sigma2, yldfac, cohes, phi, neta,
                                         xls, xre, yls );

  return;
}

extern "C"
void drprecpc_app_H( float *xx, float *yy,      float *zz,
                     float *xy, float *xz,      float *yz,
                     float *mu, float *sigma2,  float *yldfac,
                     int nzt,   int xls,        int xre, int yls, int yre, cudaStream_t St ) {
  dim3 block( BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1 );
  dim3 grid( (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z, ((yre - yls + 1) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( drprecpc_app, cudaFuncCachePreferL1 );
  drprecpc_app<<<grid, block, 0, St>>>( xx, yy, zz, xy, xz, yz, mu,
                                        sigma2, yldfac, xls, xre, yls );

  return;
}

extern "C"
void addsrc_H( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,  cudaStream_t St,
               float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
               float *xx,   float *yy,      float *zz,  float *xy,  float *yz,  float *xz ) {
  dim3 grid, block;
  if( npsrc < 256 ) {
    block.x = npsrc;
    grid.x  = 1;
  } else {
    block.x = 256;
    grid.x  = int( (npsrc + 255) / 256 );
  }

  cudaError_t cerr;
  cerr = cudaGetLastError();
  if( cerr != cudaSuccess )
    printf( "CUDA ERROR: addsrc before kernel: %s\n", cudaGetErrorString( cerr ) );

  addsrc_cu<<<grid, block, 0, St>>>( i,  READ_STEP, dim, psrc, npsrc, axx, ayy, azz, axz, ayz, axy,
                                     xx, yy,        zz,  xy,   yz,    xz );
  cerr = cudaGetLastError();
  if( cerr != cudaSuccess )
    printf( "CUDA ERROR: addsrc after kernel: %s\n", cudaGetErrorString( cerr ) );

  return;
}

__global__ void dvelcx( float *u1,    float *v1,    float *w1,    float *xx,  float *yy,  float *zz, float *xy, float *xz, float *yz,
                        float *dcrjx, float *dcrjy, float *dcrjz, float *d_1, int s_i,    int e_i ) {
  register int   i, j, k, pos,     pos_im1, pos_im2;
  register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
  register int   pos_jm2, pos_jm1, pos_jp1, pos_jp2;
  register int   pos_ip1, pos_jk1, pos_ik1, pos_ijk;
  register float f_xx,    xx_im1,  xx_ip1,  xx_im2;
  register float f_xy,    xy_ip1,  xy_ip2,  xy_im1;
  register float f_xz,    xz_ip1,  xz_ip2,  xz_im1;
  register float f_d1,    f_d2,    f_d3,    f_dcrj, f_dcrjy, f_dcrjz, f_yz;

  k    = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  j    = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + 2 + 4 * LOOP;
  i    = e_i;
  pos  = i * d_slice_1 + j * d_yline_1 + k;

  f_xx    = xx[pos+d_slice_1];
  xx_im1  = xx[pos];
  xx_im2  = xx[pos-d_slice_1]; 
  xy_ip1  = xy[pos+d_slice_2];
  f_xy    = xy[pos+d_slice_1];
  xy_im1  = xy[pos];
  xz_ip1  = xz[pos+d_slice_2];
  f_xz    = xz[pos+d_slice_1];
  xz_im1  = xz[pos];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

  for( i = e_i; i >= s_i; i-- ) {
    pos_km2  = pos - 2;
    pos_km1  = pos - 1;
    pos_kp1  = pos + 1;
    pos_kp2  = pos + 2;
    pos_jm2  = pos - d_yline_2;
    pos_jm1  = pos - d_yline_1;
    pos_jp1  = pos + d_yline_1;
    pos_jp2  = pos + d_yline_2;
    pos_im1  = pos - d_slice_1;
    pos_im2  = pos - d_slice_2;
    pos_ip1  = pos + d_slice_1;
    pos_jk1  = pos - d_yline_1 - 1;
    pos_ik1  = pos + d_slice_1 - 1;
    pos_ijk  = pos + d_slice_1 - d_yline_1;

    xx_ip1   = f_xx;
    f_xx     = xx_im1;
    xx_im1   = xx_im2;
    xx_im2   = xx[pos_im2];
    xy_ip2   = xy_ip1;
    xy_ip1   = f_xy;
    f_xy     = xy_im1;
    xy_im1   = xy[pos_im1];
    xz_ip2   = xz_ip1;
    xz_ip1   = f_xz;
    f_xz     = xz_im1;
    xz_im1   = xz[pos_im1];
    f_yz     = yz[pos];

    f_dcrj   = dcrjx[i] * f_dcrjy * f_dcrjz;
    f_d1     = 0.25 * (d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
    f_d2     = 0.25 * (d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
    f_d3     = 0.25 * (d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);

    f_d1     = d_dth / f_d1;
    f_d2     = d_dth / f_d2;
    f_d3     = d_dth / f_d3;

    u1[pos]  = (u1[pos] + f_d1 * (d_c1 * (f_xx        - xx_im1)      + d_c2 * (xx_ip1      - xx_im2)
                                + d_c1 * (f_xy        - xy[pos_jm1]) + d_c2 * (xy[pos_jp1] - xy[pos_jm2])
                                + d_c1 * (f_xz        - xz[pos_km1]) + d_c2 * (xz[pos_kp1] - xz[pos_km2]))) * f_dcrj;
    v1[pos]  = (v1[pos] + f_d2 * (d_c1 * (xy_ip1      - f_xy)        + d_c2 * (xy_ip2      - xy_im1)
                                + d_c1 * (yy[pos_jp1] - yy[pos])     + d_c2 * (yy[pos_jp2] - yy[pos_jm1])
                                + d_c1 * (f_yz        - yz[pos_km1]) + d_c2 * (yz[pos_kp1] - yz[pos_km2]))) * f_dcrj;

    w1[pos]  = (w1[pos] + f_d3 * (d_c1 * (xz_ip1      - f_xz)        + d_c2 * (xz_ip2      - xz_im1)
                                + d_c1 * (f_yz        - yz[pos_jm1]) + d_c2 * (yz[pos_jp1] - yz[pos_jm2])
                                + d_c1 * (zz[pos_kp1] - zz[pos])     + d_c2 * (zz[pos_kp2] - zz[pos_km1]))) * f_dcrj;
    pos      = pos_im1;
  }

  return;
}

__global__ void dvelcy( float *u1,    float *v1,    float *w1,    float *xx,  float *yy,    float *zz,    float *xy,    float *xz,  float *yz,
                        float *dcrjx, float *dcrjy, float *dcrjz, float *d_1, float *s_u1,  float *s_v1,  float *s_w1,  int s_j,    int e_j ) {
  register int   i, j, k, pos,     j2,      pos2, pos_jm1, pos_jm2;
  register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
  register int   pos_im2, pos_im1, pos_ip1, pos_ip2;
  register int   pos_jk1, pos_ik1, pos_ijk;
  register float f_xy,    xy_jp1,  xy_jm1,  xy_jm2;
  register float f_yy,    yy_jp2,  yy_jp1,  yy_jm1;
  register float f_yz,    yz_jp1,  yz_jm1,  yz_jm2;
  register float f_d1,    f_d2,    f_d3,    f_dcrj, f_dcrjx, f_dcrjz, f_xz;

  k     = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  i     = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + 2 + 4 * LOOP;
  j     = e_j;
  j2    = 4 * LOOP - 1;
  pos   = i * d_slice_1 + j * d_yline_1 + k;
  pos2  = i * 4 * LOOP * d_yline_1 + j2 * d_yline_1 + k;

  f_xy    = xy[pos+d_yline_1];
  xy_jm1  = xy[pos];
  xy_jm2  = xy[pos-d_yline_1];
  yy_jp1  = yy[pos+d_yline_2];
  f_yy    = yy[pos+d_yline_1];
  yy_jm1  = yy[pos];
  f_yz    = yz[pos+d_yline_1];
  yz_jm1  = yz[pos];
  yz_jm2  = yz[pos-d_yline_1];
  f_dcrjz = dcrjz[k];
  f_dcrjx = dcrjx[i];

  for( j = e_j; j >= s_j; j-- ) {
    pos_km2  = pos - 2;
    pos_km1  = pos - 1;
    pos_kp1  = pos + 1;
    pos_kp2  = pos + 2;
    pos_jm2  = pos - d_yline_2;
    pos_jm1  = pos - d_yline_1;
    pos_im1  = pos - d_slice_1;
    pos_im2  = pos - d_slice_2;
    pos_ip1  = pos + d_slice_1;
    pos_ip2  = pos + d_slice_2;
    pos_jk1  = pos - d_yline_1 - 1;
    pos_ik1  = pos + d_slice_1 - 1;
    pos_ijk  = pos + d_slice_1 - d_yline_1;

    xy_jp1   = f_xy;
    f_xy     = xy_jm1;
    xy_jm1   = xy_jm2;
    xy_jm2   = xy[pos_jm2];
    yy_jp2   = yy_jp1;
    yy_jp1   = f_yy;
    f_yy     = yy_jm1;
    yy_jm1   = yy[pos_jm1];
    yz_jp1   = f_yz;
    f_yz     = yz_jm1;
    yz_jm1   = yz_jm2;
    yz_jm2   = yz[pos_jm2];
    f_xz     = xz[pos];

    f_dcrj   = f_dcrjx * dcrjy[j] * f_dcrjz;
    f_d1     = 0.25 * (d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
    f_d2     = 0.25 * (d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
    f_d3     = 0.25 * (d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);

    f_d1     = d_dth / f_d1;
    f_d2     = d_dth / f_d2;
    f_d3     = d_dth / f_d3;

    s_u1[pos2] = (u1[pos] + f_d1 * (d_c1 * (xx[pos]     - xx[pos_im1]) + d_c2 * (xx[pos_ip1] - xx[pos_im2])
                                  + d_c1 * (f_xy        - xy_jm1)      + d_c2 * (xy_jp1      - xy_jm2)
                                  + d_c1 * (f_xz        - xz[pos_km1]) + d_c2 * (xz[pos_kp1] - xz[pos_km2]))) * f_dcrj;
    s_v1[pos2] = (v1[pos] + f_d2 * (d_c1 * (xy[pos_ip1] - f_xy)        + d_c2 * (xy[pos_ip2] - xy[pos_im1])
                                  + d_c1 * (yy_jp1      - f_yy)        + d_c2 * (yy_jp2      - yy_jm1)
                                  + d_c1 * (f_yz        - yz[pos_km1]) + d_c2 * (yz[pos_kp1] - yz[pos_km2]))) * f_dcrj;
    s_w1[pos2] = (w1[pos] + f_d3 * (d_c1 * (xz[pos_ip1] - f_xz)        + d_c2 * (xz[pos_ip2] - xz[pos_im1])
                                  + d_c1 * (f_yz        - yz_jm1)      + d_c2 * (yz_jp1      - yz_jm2)
                                  + d_c1 * (zz[pos_kp1] - zz[pos])     + d_c2 * (zz[pos_kp2] - zz[pos_km1]))) * f_dcrj;

    pos        = pos_jm1;
    pos2       = pos2 - d_yline_1;
  }

  return;
}

__global__ void update_boundary_y( float *u1, float *v1, float *w1, float *s_u1, float *s_v1, float *s_w1, int rank, int flag ) {
  register int i, j, k, pos, posj;

  k = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  i = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + 2 + 4 * LOOP;

  if( flag == Front && rank != -1 ) {
    j     = 2;
    pos   = i * d_slice_1 + j * d_yline_1 + k;
    posj  = i * 4 * LOOP * d_yline_1 + k;

    for( j = 2; j < 2 + 4 * LOOP; j++ ) {
      u1[pos] = s_u1[posj];
      v1[pos] = s_v1[posj];
      w1[pos] = s_w1[posj];
      pos     = pos  + d_yline_1;
      posj    = posj + d_yline_1;
    }
  }

  if( flag == Back && rank != -1 ) {
    j     = d_nyt + 4 * LOOP + 2;
    pos   = i * d_slice_1 + j * d_yline_1 + k;
    posj  = i * 4 * LOOP * d_yline_1 + k;

    for( j = d_nyt + 4 * LOOP + 2; j < d_nyt + 8 * LOOP + 2; j++ ) {
      u1[pos] = s_u1[posj];
      v1[pos] = s_v1[posj];
      w1[pos] = s_w1[posj];
      pos     = pos  + d_yline_1;
      posj    = posj + d_yline_1;
    }
  }

  return;
}

/* kernel functions to apply free-surface B.C.s to stress */
__global__ void fstr( float *zz, float *xz, float *yz, int s_i, int e_i, int s_j ) {
  register int i, j, k;
  register int pos, pos_im1;

  k    = d_nzt + ALIGN - 1;
  j    = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + s_j;
  i    = e_i;
  pos  = i * d_slice_1 + j * d_yline_1 + k;

  for( i = e_i; i >= s_i; i-- ) {
    pos_im1   = pos - d_slice_1;

    //! asymmetry reflection above free surface
    zz[pos+1] = -zz[pos];
    zz[pos+2] = -zz[pos-1];

    xz[pos+1] = -xz[pos-1];
    xz[pos+2] = -xz[pos-2];

    yz[pos+1] = -yz[pos-1];
    yz[pos+2] = -yz[pos-2];

    //! horizontal shear stresses on free surface
    xz[pos]   = 0.0;
    yz[pos]   = 0.0;

    pos     = pos_im1;
  }
}

__global__ void dstrqc( float *xx,  float *yy,    float *zz,    float *xy,    float *xz,      float *yz,
                        float *r1,  float *r2,    float *r3,    float *r4,    float *r5,      float *r6,
                        float *u1,  float *v1,    float *w1,    float *lam,   float *mu,      float *qp, float *coeff,
                        float *qs,  float *dcrjx, float *dcrjy, float *dcrjz, float *lam_mu,  int NX,
                        int rankx,  int ranky,    int s_i,      int e_i,      int s_j ) {
  register int    i,  j,  k,  g_i;
  register int    pos,     pos_ip1, pos_im2, pos_im1;
  register int    pos_km2, pos_km1, pos_kp1, pos_kp2;
  register int    pos_jm2, pos_jm1, pos_jp1, pos_jp2;
  register int    pos_ik1, pos_jk1, pos_ijk, pos_ijk1,f_ww;
  register float  vs1, vs2, vs3, a1, tmp, vx1,f_wwo;
  register float  xl,  xm,  xmu1, xmu2, xmu3;
  register float  qpa, h,   h1,   h2,   h3;
  register float  qpaw,hw,h1w,h2w,h3w;
  register float  f_vx1, f_vx2,  f_dcrj, f_r,  f_dcrjy, f_dcrjz;
  register float  f_rtmp;
  register float  f_u1, u1_ip1, u1_ip2, u1_im1;
  register float  f_v1, v1_im1, v1_ip1, v1_im2;
  register float  f_w1, w1_im1, w1_im2, w1_ip1;
  
  k    = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  j    = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + s_j;
  i    = e_i;
  pos  = i * d_slice_1 + j * d_yline_1 + k;

  u1_ip1 = u1[pos+d_slice_2];
  f_u1   = u1[pos+d_slice_1];
  u1_im1 = u1[pos];    
  f_v1   = v1[pos+d_slice_1];
  v1_im1 = v1[pos];
  v1_im2 = v1[pos-d_slice_1];
  f_w1   = w1[pos+d_slice_1];
  w1_im1 = w1[pos];
  w1_im2 = w1[pos-d_slice_1];
  f_dcrjz = dcrjz[k];
  f_dcrjy = dcrjy[j];

  for( i = e_i; i >= s_i; i-- ) {
    f_vx1   = tex1Dfetch( p_vx1, pos );
    f_vx2   = tex1Dfetch( p_vx2, pos );
    f_ww    = tex1Dfetch( p_ww, pos );
    f_wwo   = tex1Dfetch( p_wwo, pos );
/*
    if(f_wwo!=f_wwo){ 
      xx[pos] = yy[pos] = zz[pos] = xy[pos] = xz[pos] = yz[pos] = 1.0;
      r1[pos] = r2[pos] = r3[pos] = r4[pos] = r5[pos] = r6[pos] = 1.0;
      return;
    }
*/
    f_dcrj   = dcrjx[i] * f_dcrjy *f_dcrjz;

    pos_km2  = pos - 2;
    pos_km1  = pos - 1;
    pos_kp1  = pos + 1;
    pos_kp2  = pos + 2;
    pos_jm2  = pos - d_yline_2;
    pos_jm1  = pos - d_yline_1;
    pos_jp1  = pos + d_yline_1;
    pos_jp2  = pos + d_yline_2;
    pos_im2  = pos - d_slice_2;
    pos_im1  = pos - d_slice_1;
    pos_ip1  = pos + d_slice_1;
    pos_jk1  = pos - d_yline_1 - 1;
    pos_ik1  = pos + d_slice_1 - 1;
    pos_ijk  = pos + d_slice_1 - d_yline_1;
    pos_ijk1 = pos + d_slice_1 - d_yline_1 - 1;

    xl       = 8.0 / (lam[pos]      + lam[pos_ip1] + lam[pos_jm1] + lam[pos_ijk]
                    + lam[pos_km1]  + lam[pos_ik1] + lam[pos_jk1] + lam[pos_ijk1]);
    xm       = 16.0 / (mu[pos]       + mu[pos_ip1]  + mu[pos_jm1]  + mu[pos_ijk]
                     + mu[pos_km1]   + mu[pos_ik1]  + mu[pos_jk1]  + mu[pos_ijk1]);
    xmu1     = 2.0 / (mu[pos]       + mu[pos_km1]);
    xmu2     = 2.0 / (mu[pos]       + mu[pos_jm1]);
    xmu3     = 2.0 / (mu[pos]       + mu[pos_ip1]);
    xl       = xl + xm;
    qpa      = 0.0625 * (qp[pos]     + qp[pos_ip1] + qp[pos_jm1] + qp[pos_ijk]
                       + qp[pos_km1] + qp[pos_ik1] + qp[pos_jk1] + qp[pos_ijk1]);
//  www=f_ww;
    if( 1. / (qpa * 2.0) <= 200.0 ) {
//printf("coeff[f_ww*2-2] %g\n",coeff[f_ww*2-2]);
      qpaw  = coeff[f_ww*2-2] * (2. * qpa) * (2. * qpa) + coeff[f_ww*2-1] * (2. * qpa);
//        qpaw=coeff[www*2-2]*(2.*qpa)*(2.*qpa)+coeff[www*2-1]*(2.*qpa);
//  qpaw=qpaw/2.;
    } else {
      qpaw  = f_wwo * qpa;
    }
//           printf("qpaw %f\n",qpaw);
//printf("qpaw1 %g\n",qpaw);  
    qpaw    = qpaw / f_wwo;
//printf("qpaw2 %g\n",qpaw);

    h       = 0.0625 * (qs[pos]     + qs[pos_ip1] + qs[pos_jm1] + qs[pos_ijk]
                      + qs[pos_km1] + qs[pos_ik1] + qs[pos_jk1] + qs[pos_ijk1]);

    if( 1. / (h * 2.0) <= 200.0 ) {
      hw    = coeff[f_ww*2-2] * (2. * h) * (2. * h) + coeff[f_ww*2-1] * (2. * h);
  //                  hw=hw/2.;
    } else {
      hw    = f_wwo * h;
    }
    hw      = hw / f_wwo;

    h1      = 0.250 * (qs[pos] + qs[pos_km1]);

    if( 1. / (h1 * 2.0) <= 200.0 ) {
      h1w   = coeff[f_ww*2-2] * (2. * h1) * (2. * h1) + coeff[f_ww*2-1] * (2. * h1);
  //                  h1w=h1w/2.;
    } else {
      h1w   = f_wwo * h1;
    }
    h1w     =h1w / f_wwo;

    h2      = 0.250 * (qs[pos] + qs[pos_jm1]);

    if( 1. / (h2 * 2.0) <= 200.0 ) {
      h2w   = coeff[f_ww*2-2] * (2. * h2) * (2. * h2) + coeff[f_ww*2-1] * (2. * h2);
  //                  h2w=h2w/2.;
    } else {
      h2w   = f_wwo * h2;
    }
    h2w     = h2w / f_wwo;

    h3      = 0.250 * (qs[pos] + qs[pos_ip1]);

    if( 1. / (h3 * 2.0) <= 200.0 ) {
      h3w   = coeff[f_ww*2-2] * (2. * h3) * (2. * h3) + coeff[f_ww*2-1] * (2. * h3);
  //                  h3w=h3w/2.;
    } else {
      h3w   = f_wwo * h3;
    }
    h3w     = h3w / f_wwo;

    h        = -xm * hw * d_dh1;
    h1       = -xmu1 * h1w * d_dh1;
    h2       = -xmu2 * h2w * d_dh1;
    h3       = -xmu3 * h3w * d_dh1;

//        h1       = -xmu1*hw1*d_dh1;
    //h2       = -xmu2*hw2*d_dh1;
    //h3       = -xmu3*hw3*d_dh1;

    qpa      = -qpaw * xl * d_dh1;
//        qpa      = -qpaw*xl*d_dh1;
    xm       = xm * d_dth;
    xmu1     = xmu1 * d_dth;
    xmu2     = xmu2 * d_dth;
    xmu3     = xmu3 * d_dth;
    xl       = xl * d_dth;
    //f_vx2    = f_vx2*f_vx1;
    h        = h * f_vx1;
    h1       = h1 * f_vx1;
    h2       = h2 * f_vx1;
    h3       = h3 * f_vx1;

    qpa      = qpa * f_vx1;

    xm       = xm + d_DT * h;
    xmu1     = xmu1 + d_DT * h1;
    xmu2     = xmu2 + d_DT * h2;
    xmu3     = xmu3 + d_DT * h3;
    vx1      = d_DT * (1 + f_vx2 * f_vx1);

    u1_ip2   = u1_ip1;
    u1_ip1   = f_u1;
    f_u1     = u1_im1;
    u1_im1   = u1[pos_im1];
    v1_ip1   = f_v1;
    f_v1     = v1_im1;
    v1_im1   = v1_im2;
    v1_im2   = v1[pos_im2];
    w1_ip1   = f_w1;
    f_w1     = w1_im1;
    w1_im1   = w1_im2;
    w1_im2   = w1[pos_im2];

    if( k == d_nzt + ALIGN - 1 ) {
      u1[pos_kp1] = f_u1 - (f_w1 - w1_im1);
      v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);

      g_i   = d_nxt * rankx + i - 4 * LOOP - 1;
      if( g_i < NX )
        vs1 = u1_ip1 - (w1_ip1 - f_w1);
      else
        vs1 = 0.0;

      g_i   = d_nyt * ranky + j - 4 * LOOP - 1;
      if( g_i > 1 )
        vs2 = v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
      else
        vs2 = 0.0;

      w1[pos_kp1] = w1[pos_km1] - lam_mu[i*(d_nyt+4+8*LOOP)+j] * ((vs1         - u1[pos_kp1]) + (u1_ip1 - f_u1)
                                        +                         (v1[pos_kp1] - vs2)         + (f_v1   - v1[pos_jm1]) );
    } else if( k == d_nzt + ALIGN - 2 ) {
      u1[pos_kp2] = u1[pos_kp1] - (w1[pos_kp1] - w1[pos_im1+1]);
      v1[pos_kp2] = v1[pos_kp1] - (w1[pos_jp1+1] - w1[pos_kp1]);
    }

    vs1      = d_c1 * (u1_ip1 - f_u1)        + d_c2 * (u1_ip2      - u1_im1);
    vs2      = d_c1 * (f_v1   - v1[pos_jm1]) + d_c2 * (v1[pos_jp1] - v1[pos_jm2]);
    vs3      = d_c1 * (f_w1   - w1[pos_km1]) + d_c2 * (w1[pos_kp1] - w1[pos_km2]);

    tmp      = xl * (vs1 + vs2 + vs3);
    a1       = qpa * (vs1 + vs2 + vs3);
    tmp      = tmp + d_DT * a1;

    // modified for q(f)
//           f_wwo     = f_wwo*2.;
//	     a1=a1*2.;
//	     h=h*2.;
//	     h1=h1*2.;
//	     h2=h2*2.;
//	     h3=h3*2.;

    f_r      = r1[pos];
    f_rtmp   = -h * (vs2 + vs3) + a1;
    xx[pos]  = xx[pos] + tmp - xm * (vs2 + vs3) + vx1 * f_r;
    r1[pos]  = f_vx2 * f_r + f_wwo * f_rtmp;
//KBW          r1[pos]  = f_vx2*f_r + f_rtmp;
    //r1[pos]  = f_vx2*f_r - f_wwo*h*(vs2+vs3)        + f_wwo*a1;
    f_rtmp   = f_rtmp * (f_wwo - 1) + f_vx2 * f_r * (1 - f_vx1);
    xx[pos]  = (xx[pos] + d_DT * f_rtmp) * f_dcrj;

    f_r      = r2[pos];
    f_rtmp   = -h * (vs1 + vs3) + a1;
    yy[pos]  = yy[pos] + tmp - xm * (vs1 + vs3) + vx1 * f_r;
    r2[pos]  = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp   = f_rtmp * (f_wwo - 1) + f_vx2 * f_r * (1 - f_vx1);
    yy[pos]  = (yy[pos] + d_DT * f_rtmp) * f_dcrj;

    f_r      = r3[pos];
    f_rtmp   = -h * (vs1 + vs2) + a1;
    zz[pos]  = zz[pos] + tmp - xm * (vs1 + vs2) + vx1 * f_r;
    r3[pos]  = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp   = f_rtmp * (f_wwo - 1) + f_vx2 * f_r * (1 - f_vx1);
    zz[pos]  = (zz[pos] + d_DT * f_rtmp) * f_dcrj;

    vs1      = d_c1 * (u1[pos_jp1] - f_u1)   + d_c2 * (u1[pos_jp2] - u1[pos_jm1]);
    vs2      = d_c1 * (f_v1        - v1_im1) + d_c2 * (v1_ip1      - v1_im2);

    f_r      = r4[pos];
    f_rtmp   = h1 * (vs1 + vs2);
    xy[pos]  = xy[pos] + xmu1 * (vs1 + vs2) + vx1 * f_r;
    r4[pos]  = f_vx2 * f_r + f_wwo * f_rtmp;
    f_rtmp   = f_rtmp * (f_wwo - 1) + f_vx2 * f_r * (1 - f_vx1);
    xy[pos]  = (xy[pos] + d_DT * f_rtmp) * f_dcrj;

    if( k == d_nzt + ALIGN - 1 ) {
      zz[pos+1] = -zz[pos];
      xz[pos]   = 0.0;
      yz[pos]   = 0.0;
    } else {
      //! modified for q(f)
      vs1     = d_c1 * (u1[pos_kp1] - f_u1)   + d_c2 * (u1[pos_kp2] - u1[pos_km1]);
      vs2     = d_c1 * (f_w1        - w1_im1) + d_c2 * (w1_ip1      - w1_im2);
      f_r     = r5[pos];
      f_rtmp  = h2 * (vs1 + vs2);
      xz[pos] = xz[pos] + xmu2 * (vs1 + vs2) + vx1 * f_r;
      r5[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
//          f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
//kBW	        r5[pos] = f_vx2*f_r + f_rtmp;
      f_rtmp  = f_rtmp * (f_wwo - 1) + f_vx2 * f_r * (1 - f_vx1);
      xz[pos] = (xz[pos] + d_DT * f_rtmp) * f_dcrj;

      vs1     = d_c1 * (v1[pos_kp1] - f_v1) + d_c2 * (v1[pos_kp2] - v1[pos_km1]);
      vs2     = d_c1 * (w1[pos_jp1] - f_w1) + d_c2 * (w1[pos_jp2] - w1[pos_jm1]);

      f_r     = r6[pos];
      f_rtmp  = h3 * (vs1 + vs2);
      yz[pos] = yz[pos]  + xmu3 * (vs1 + vs2) + vx1 * f_r;
    	r6[pos] = f_vx2 * f_r + f_wwo * f_rtmp;
      f_rtmp  = f_rtmp * (f_wwo - 1) + f_vx2 * f_r * (1 - f_vx1);
      yz[pos] = (yz[pos] + d_DT * f_rtmp) * f_dcrj;

      if( k == d_nzt + ALIGN - 2 ) {
        zz[pos+3] = -zz[pos];
        xz[pos+2] = -xz[pos];
        yz[pos+2] = -yz[pos];
      } else if( k == d_nzt + ALIGN - 3 ) {
        xz[pos+4] = -xz[pos];
        yz[pos+4] = -yz[pos];
      }
    }

    pos       = pos_im1;
  }

  return;
}

__global__ void drprecpc_calc( float *xx,     float *yy,      float *zz,
                               float *xy,     float *xz,      float *yz,    float *mu,  float *d1,
                               float *sigma2, float *yldfac,  float *cohes, float *phi, float *neta,
                               int s_i,       int e_i,        int s_j ) {
  register int    i,j,k,pos;
  register int    pos_im1,pos_ip1,pos_jm1,pos_km1;
  register int    pos_ip1jm1;
  register int    pos_ip1km1,pos_jm1km1;
  register float  Sxx, Syy, Szz, Sxy, Sxz, Syz;
  register float  Sxxp, Syyp, Szzp, Sxyp, Sxzp, Syzp;
  register float  depxx, depyy, depzz, depxy, depxz, depyz;
  register float  SDxx, SDyy, SDzz;
  register float  iyldfac, Tv, sigma_m, taulim, taulim2, rphi;
  register float  xm, iixx, iiyy, iizz;
  register float  mu_, secinv, sqrtSecinv;
//  register int   jj, kk;

  //! Compute initial stress on GPU (Daniel)
  register float  ini[9], ini_ip1[9];
  register float  depth, pfluid;
  register int    srfpos;

  k    = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  j    = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + s_j;
  i    = e_i;
  pos  = i * d_slice_1 + j * d_yline_1 + k;

//  kk   = k - ALIGN;
//  jj   = j - (2 + 4 * LOOP);

  srfpos  = d_nzt + ALIGN - 1;
  depth   = (float) (srfpos - k) * d_DH;

  if( depth > 0 )
    pfluid = (depth + d_DH * 0.5) * 9.81e3;
  else
    pfluid = d_DH / 2. * 9.81e3;

  //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k, depth, pfluid);

  for( i = e_i; i >= s_i; --i ) {
    pos_im1     = pos - d_slice_1;
    pos_ip1     = pos + d_slice_1;
    pos_jm1     = pos - d_yline_1;
    pos_km1     = pos - 1;
    pos_ip1jm1  = pos_ip1 - d_yline_1;
    pos_ip1km1  = pos_ip1 - 1;
    pos_jm1km1  = pos_jm1 - 1;

    mu_   = mu[pos];

// start drprnn
    rotate_principal( sigma2[pos], pfluid, ini );
    rotate_principal( sigma2[pos_ip1], pfluid, ini_ip1 );
    /*cuPrintf("ini[8] = %5.2e, ini[4]=%5.2e sigma2=%5.2e pfluid=%5.2e\n",
       ini[8], ini[4], sigma2[pos], pfluid);*/

    /*iixx  = 0.5*(inixx[pos] + inixx[pos_ip1]);
    iiyy  = 0.5*(iniyy[pos] + iniyy[pos_ip1]);
    iizz  = 0.5*(inizz[pos] + inizz[pos_ip1]);*/
    iixx  = 0.5 * (ini[0] + ini_ip1[0]);
    iiyy  = 0.5 * (ini[4] + ini_ip1[4]);
    iizz  = 0.5 * (ini[8] + ini_ip1[8]);

    Sxx   = xx[pos] + iixx;
    Syy   = yy[pos] + iiyy;
    Szz   = zz[pos] + iizz;
    Sxz   = 0.25 * (xz[pos] + xz[pos_ip1] + xz[pos_km1] + xz[pos_ip1km1])
            //+ 0.5*(inixz[pos] + inixz[pos_ip1]);
            + 0.5 * (ini[2] + ini_ip1[2]);
    Syz   = 0.25 * (yz[pos] + yz[pos_jm1] + yz[pos_km1] + yz[pos_jm1km1])
            //+ 0.5*(iniyz[pos] + iniyz[pos_ip1]);
            + 0.5 * (ini[5] + ini_ip1[5]);
    Sxy   = 0.25 * (xy[pos] + xy[pos_ip1] + xy[pos_jm1] + xy[pos_ip1jm1])
            //+ 0.5*(inixy[pos] + inixy[pos_ip1]);
            + 0.5 * (ini[1] + ini_ip1[1]);

    Tv = d_DH / sqrt( 1. / (mu_ * d1[pos]) );

    Sxxp = Sxx;
    Syyp = Syy;
    Szzp = Szz;
    Sxyp = Sxy;
    Sxzp = Sxz;
    Syzp = Syz;

    //! drucker_prager function:
    rphi        = phi[pos] * 0.017453292519943295;
    sigma_m     = (Sxx + Syy + Szz) / 3.;
    SDxx        = Sxx - sigma_m;
    SDyy        = Syy - sigma_m;
    SDzz        = Szz - sigma_m;
    secinv      = 0.5 * (SDxx * SDxx + SDyy * SDyy + SDzz * SDzz)
                  + Sxz * Sxz + Sxy * Sxy + Syz * Syz;
    sqrtSecinv  = sqrt( secinv );
    taulim2     = cohes[pos] * cos( rphi ) - (sigma_m + pfluid) * sin( rphi );
    if( taulim2 > 0. )
      taulim  = taulim2;
    else
      taulim  = 0.;

    if( sqrtSecinv > taulim ) {
      iyldfac = taulim / sqrtSecinv
                + (1. - taulim / sqrtSecinv) * exp( -d_DT / Tv );
      Sxx     = SDxx * iyldfac + sigma_m;
      Syy     = SDyy * iyldfac + sigma_m;
      Szz     = SDzz * iyldfac + sigma_m;
      Sxz     = Sxz * iyldfac;
      Sxy     = Sxy * iyldfac;
      Syz     = Syz * iyldfac;
    } else
      iyldfac = 1.;

    yldfac[pos] = iyldfac;
    //! end drucker_prager function

    if( yldfac[pos] < 1. ) {
      xm        = 2. / (mu_ + mu[pos_ip1]);
      depxx     = (Sxx - Sxxp) / xm;
      depyy     = (Syy - Syyp) / xm;
      depzz     = (Szz - Szzp) / xm;
      depxy     = (Sxy - Sxyp) / xm;
      depxz     = (Sxz - Sxzp) / xm;
      depyz     = (Syz - Syzp) / xm;

      neta[pos] = neta[pos] + sqrt( 0.5 * (depxx*depxx + depyy*depyy + depzz*depzz)
                                   + 2. * (depxy*depxy + depxz*depxz + depyz*depyz) );
    }
    else
      yldfac[pos] = 1.;
    //! end drprnn

// DEBUG if neta/EPxx etc are set
    //neta[pos] = 1.E+2;
    //EPxx[pos] = 1.E+2;
// DEBUG - end

    pos = pos_im1;
  }

return;
}

//! treatment of shear stress components moved to separate kernel code (Daniel)
__global__ void drprecpc_app( float *xx,      float *yy,  float *zz,
                              float *xy,      float *xz,  float *yz,
                              float *mu,      float *sigma2,
                              float *yldfac,  int s_i,    int e_i, int s_j ) {
  register int    i,j,k,pos;
  register int    pos_im1,pos_ip1,pos_jp1,pos_kp1;
  register int    pos_im1jp1,pos_im1kp1,pos_ip1jp1;
  register int    pos_ip1kp1,pos_jp1kp1,pos_ip1jp1kp1;
  register float  iyldfac;
  register float  iist;
//  register float xm, tst;
//  register float mu_;
  register float  Sxx, Syy, Szz;
  register float  iixx, iiyy, iizz, SDxx, SDyy, SDzz, sigma_m;

  register float  ini[9], ini_ip1[9], ini_kp1[9], ini_ip1kp1[9];
  register float  ini_jp1[9], ini_ip1jp1[9], ini_jp1kp1[9], ini_ip1jp1kp1[9];
  register int    srfpos;
  register float  depth, pfluid, depth_kp1, pfluid_kp1;

  k    = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  j    = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + s_j;
  i    = e_i;
  pos  = i * d_slice_1 + j * d_yline_1 + k;

  srfpos    = d_nzt + ALIGN - 1;
  depth     = (float) (srfpos - k) * d_DH;
  depth_kp1 = (float) (srfpos - k + 1) * d_DH;

  if( depth > 0 )
    pfluid  = (depth + d_DH / 2.) * 9.81e3;
  else
    pfluid  = d_DH / 2. * 9.81e3;

  if( depth_kp1 > 0 )
    pfluid_kp1  = (depth_kp1 + d_DH / 2.) * 9.81e3;
  else
    pfluid_kp1  = d_DH / 2. * 9.81e3;

  //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k, depth, pfluid);
  //cuPrintf("k=%d, depth=%f, pfluid=%e\n", k+1, depth_kp1, pfluid_kp1);

  for( i = e_i; i >= s_i; --i ) {
    pos_im1       = pos - d_slice_1;
    pos_ip1       = pos + d_slice_1;
    pos_jp1       = pos + d_yline_1;
    pos_kp1       = pos + 1;  //! changed from -1 to +1 (Daniel)
    pos_im1jp1    = pos_im1 + d_yline_1;
    pos_im1kp1    = pos_im1 + 1;
    pos_ip1jp1    = pos_ip1 + d_yline_1;
    pos_ip1kp1    = pos_ip1 + 1;
    pos_jp1kp1    = pos_jp1 + 1;
    pos_ip1jp1kp1 = pos_ip1 + d_yline_1 + 1;

//    mu_           = mu[pos];

    //! start drprnn
    if( yldfac[pos] < 1. ) {
      //! compute initial stress at pos and pos_ip1
      rotate_principal( sigma2[pos], pfluid, ini );
      rotate_principal( sigma2[pos_ip1], pfluid_kp1, ini_ip1 );
      iixx    = 0.5 * (ini[0] + ini_ip1[0]);
      iiyy    = 0.5 * (ini[4] + ini_ip1[4]);
      iizz    = 0.5 * (ini[8] + ini_ip1[8]);

      Sxx     = xx[pos] + iixx;
      Syy     = yy[pos] + iiyy;
      Szz     = zz[pos] + iizz;

      sigma_m = (Sxx + Syy + Szz) / 3.;
      SDxx    = xx[pos] + iixx - sigma_m;
      SDyy    = yy[pos] + iiyy - sigma_m;
      SDzz    = zz[pos] + iizz - sigma_m;

      xx[pos] = SDxx * yldfac[pos] + sigma_m - iixx;
      yy[pos] = SDyy * yldfac[pos] + sigma_m - iiyy;
      zz[pos] = SDzz * yldfac[pos] + sigma_m - iizz;
    }

    //! start drprxz
    iyldfac   = 0.25 * (yldfac[pos_im1] + yldfac[pos]
                      + yldfac[pos_im1kp1] + yldfac[pos_kp1]);
    if( iyldfac < 1. ) {
      //! compute initial stress at pos and pos_kp1
      rotate_principal( sigma2[pos], pfluid, ini );
      rotate_principal( sigma2[pos_kp1], pfluid_kp1, ini_kp1 );

      iist    = 0.5 * (ini[2] + ini_kp1[2]);
//      tst     = xz[pos];
      xz[pos] = (xz[pos] + iist) * iyldfac - iist;
//      xm      = 2. / (mu_ + mu[pos_kp1]);
    }
    //! end drprxz / start drpryz

    iyldfac   = 0.25 * (yldfac[pos] + yldfac[pos_jp1]
                      + yldfac[pos_jp1kp1] + yldfac[pos_kp1]);
    if( iyldfac < 1. ) {
      //! compute initial stress at 8 positions
      rotate_principal( sigma2[pos], pfluid, ini );
      rotate_principal( sigma2[pos_ip1], pfluid, ini_ip1 );
      rotate_principal( sigma2[pos_kp1], pfluid_kp1, ini_kp1 );
      rotate_principal( sigma2[pos_ip1kp1], pfluid_kp1, ini_ip1kp1 );
      rotate_principal( sigma2[pos_jp1], pfluid, ini_jp1 );
      rotate_principal( sigma2[pos_ip1jp1], pfluid, ini_ip1jp1 );
      rotate_principal( sigma2[pos_jp1kp1], pfluid_kp1, ini_jp1kp1 );
      rotate_principal( sigma2[pos_ip1jp1kp1], pfluid_kp1, ini_ip1jp1kp1 );

      iist    = 0.125 * (ini[5] + ini_ip1[5]
                       + ini_kp1[5] + ini_ip1kp1[5]
                       + ini_jp1[5] + ini_ip1jp1[5]
                       + ini_jp1kp1[5] + ini_ip1jp1kp1[5]);

//      tst     = yz[pos];
      yz[pos] = (yz[pos] + iist) * iyldfac - iist;
//      xm      = 8. / (mu_ + mu[pos_ip1] + mu[pos_kp1]
//                    + mu[pos_ip1kp1] + mu[pos_jp1] + mu[pos_ip1jp1]
//                    + mu[pos_jp1kp1] + mu[pos_ip1jp1kp1]);
    }
    //! end drpryz / start drprxy

    iyldfac   = 0.25 * (yldfac[pos] + yldfac[pos_jp1]
                      + yldfac[pos_im1] + yldfac[pos_im1jp1]);
    if( iyldfac < 1. ) {
      rotate_principal( sigma2[pos], pfluid, ini );
      rotate_principal( sigma2[pos_jp1], pfluid, ini_jp1 );
      iist    = 0.5 * (ini[1] + ini_jp1[1]);
//      tst     = xy[pos];
      xy[pos] = (xy[pos] + iist) * iyldfac - iist;
//      xm      = 2. / (mu_ + mu[pos_jp1]);
    }

    pos = pos_im1;
  }

  return;
}

__global__ void update_yldfac( float *yldfac,
                               float *buf_L,  float *buf_R,   float *buf_F,   float *buf_B,
                               float *buf_FL, float *buf_FR,  float *buf_BL,  float *buf_BR ) {
  register int k,ind,tind;
  register int xs, xe, ys, ye;

  k    = blockIdx.x * BLOCK_SIZE_Z + threadIdx.x + ALIGN;
  ind  = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;

  //! ghost cell indices
  xs   = 1 + 4 * LOOP;
  xe   = xs + d_nxt + 1;
  ys   = xs;
  ye   = ys + d_nyt + 1;

  if( ind < 4 ) {
    //! 0: FL
    if( ind == 0 )
      yldfac[xs*d_slice_1+ys*d_yline_1+k] = buf_FL[k-ALIGN];
    //! 1: FR
    else if( ind == 1 )
      yldfac[xe*d_slice_1+ys*d_yline_1+k] = buf_FR[k-ALIGN];
    //! 2: BL
    else if( ind == 2 )
      yldfac[xs*d_slice_1+ye*d_yline_1+k] = buf_BL[k-ALIGN];
    //! 3: BR
    else if( ind == 3 )
      yldfac[xe*d_slice_1+ye*d_yline_1+k] = buf_BR[k-ALIGN];
  } else if( ind < 4 + d_nxt ) {  //! Y direction
    tind  = ind - 4;
    yldfac[tind*d_slice_1+ys*d_yline_1+k] = buf_F[tind*d_nzt+k-ALIGN];
    yldfac[tind*d_slice_1+ye*d_yline_1+k] = buf_B[tind*d_nzt+k-ALIGN];
  } else {
    tind  = ind - 4 - d_nxt;
    yldfac[xs*d_slice_1+tind*d_yline_1+k] = buf_L[tind*d_nzt+k-ALIGN];
    yldfac[xe*d_slice_1+tind*d_yline_1+k] = buf_R[tind*d_nzt+k-ALIGN];
  }

  return;
}

__global__ void addsrc_cu( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,
                           float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
                           float *xx,   float *yy,      float *zz,  float *xy,  float *yz,  float *xz ) {
  register float vtst;
  register int idx, idy, idz, j, pos;

  j = blockIdx.x * blockDim.x + threadIdx.x;
  if( j >= npsrc )
    return;

  vtst    = (float) d_DT / (d_DH * d_DH * d_DH);

  i       = i - 1;
  idx     = psrc[j*dim]   + 1 + 4*LOOP;
  idy     = psrc[j*dim+1] + 1 + 4*LOOP;
  idz     = psrc[j*dim+2] + ALIGN - 1;
  pos     = idx * d_slice_1 + idy * d_yline_1 + idz;

  xx[pos] = xx[pos] - vtst * axx[j*READ_STEP+i];
  yy[pos] = yy[pos] - vtst * ayy[j*READ_STEP+i];
  zz[pos] = zz[pos] - vtst * azz[j*READ_STEP+i];
  xz[pos] = xz[pos] - vtst * axz[j*READ_STEP+i];
  yz[pos] = yz[pos] - vtst * ayz[j*READ_STEP+i];
  xy[pos] = xy[pos] - vtst * axy[j*READ_STEP+i];

  return;
}

extern "C"
void frcvel_H( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,  int tskp, cudaStream_t St,
               float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
               float *u1,   float *v1,      float *w1,  int ymin,   int ymax ) {
  dim3 grid, block;

  if( npsrc < 256 ) {
    block.x = npsrc;
    grid.x  = 1;
  } else {
    block.x = 256;
    grid.x  = int( (npsrc + 255) / 256 );
  }

  cudaError_t cerr;
  cerr = cudaGetLastError();
  if( cerr != cudaSuccess )
    printf( "CUDA ERROR: addsrc before kernel: %s\n", cudaGetErrorString( cerr ) );
  //cudaPrintfInit();
  frcvel_cu<<<grid, block, 0, St>>>( i,  READ_STEP, dim,  psrc, npsrc, tskp, axx, ayy, azz, axz, ayz, axy,
                                     u1, v1,        w1,   ymin, ymax );

  cerr = cudaGetLastError();
  //cudaPrintfDisplay(stdout, 1);
  //cudaPrintfEnd();
  if( cerr != cudaSuccess )
    printf( "CUDA ERROR: addsrc after kernel: %s\n", cudaGetErrorString( cerr ) );

  return;
}

__global__ void frcvel_cu( int i,       int READ_STEP,  int dim,    int *psrc,  int npsrc,  int tskp,
                           float *axx,  float *ayy,     float *azz, float *axz, float *ayz, float *axy,
                           float *u1,   float *v1,      float *w1,  int xmin,   int xmax ) {
  register int    idx, idy, idz, j, pos;
  register int    i0, i1;
  register float  u1_p, u1_n, v1_p, v1_n, w1_p, w1_n;
  register float  u1_i, v1_i, w1_i, pfact;
  /*register int pos_jm1, pos_jp1, pos_jm2;*/
  bool            abvmin, blwmax;

  j     = blockIdx.x * blockDim.x + threadIdx.x;
  if( j >= npsrc )
    return;

  i     = i - 1;
  i0    = int( tskp * floorf( float( i + 1. ) / float( tskp ) ) );
  i1    = int( tskp * ceilf( float( i + 1.) / float( tskp ) ) );

  //! Linear interpolation
  pfact = float (i + 1 - i0) / float( tskp );
  //! Cosine interpolation
  //pfact = -cosf(float(i + 1 - i0) / float(tskp) * M_PI)/2 + 0.5;
  //if (j==0) cuPrintf("inside frcvel_cu: i=%d, i0=%d i1=%d\n", i, i0, i1);

  i0    /= tskp;
  i1    /= tskp;

  idx   = psrc[j*dim]   + 1 + 4 * LOOP;
  idy   = psrc[j*dim+1] + 1 + 4 * LOOP;
  idz   = psrc[j*dim+2] + ALIGN - 1;
  pos   = idx * d_slice_1 + idy * d_yline_1 + idz;
  //cuPrintf("%d %d %d\n", psrc[j*dim], psrc[j*dim+1], psrc[j*dim+2]);

  /* only add velocities inside a given zone */
  if( (xmin == -1) || (idx >= xmin) )
    abvmin  = 1;
  else
    abvmin  = 0;

  if( (xmax == -1) || (idx <= xmax) )
    blwmax  = 1;
  else
    blwmax  = 0;

  if( abvmin && blwmax ) {
    if( i < (READ_STEP * tskp) ) {
      u1_p = axx[i0*npsrc+j];
      v1_p = ayy[i0*npsrc+j];
      w1_p = azz[i0*npsrc+j];

      u1_n = axx[i1*npsrc+j];
      v1_n = ayy[i1*npsrc+j];
      w1_n = azz[i1*npsrc+j];

      u1_i = u1_p + (u1_n - u1_p) * pfact;
      v1_i = v1_p + (v1_n - v1_p) * pfact;
      w1_i = w1_p + (w1_n - w1_p) * pfact;

      /*if (j==0){
    cuPrintf("u1[%d]=%e, u1[%d]=%e, u1_i=%e\n", i0, u1_p, i1, u1_n, u1_i);
      }*/

      if (i == (READ_STEP * tskp - 1)){
        //if (j==0) cuPrintf("inside frcvel_cu: last step at i=%d\n", i);
        /* copy last value back to beginning of array */
        axx[j]  = axx[i1*npsrc+j];
        ayy[j]  = ayy[i1*npsrc+j];
        azz[j]  = azz[i1*npsrc+j];

        /* save restoring force to source array in case this is the last
          time step where velocity is prescribed  */
        axx[i1*npsrc+j] = (u1_n - u1[pos]) / d_DT;
        ayy[i1*npsrc+j] = (v1_n - v1[pos]) / d_DT;
        azz[i1*npsrc+j] = (w1_n - w1[pos]) / d_DT;
      }

      u1[pos] = u1_i;
      v1[pos] = v1_i;
      w1[pos] = w1_i;

      /*if (((psrc[j*dim] == 90) && (psrc[j*dim+1] == 10)) && (psrc[j*dim+2] == 180)){
    cuPrintf("dbg1>> i=%d, pos=%ld, %e, %e, %e\n", i, pos,
      axx[i*npsrc+j], ayy[i*npsrc+j], azz[i*npsrc+j]);
      }*/
    }

    /* we keep applying the static force needed to stabilize discontinuity */
    else {
      u1[pos] += axx[READ_STEP*npsrc+j] * d_DT;
      v1[pos] += ayy[READ_STEP*npsrc+j] * d_DT;
      w1[pos] += azz[READ_STEP*npsrc+j] * d_DT;
      /*if (((psrc[j*dim] == 90) && (psrc[j*dim+1] == 10)) && (psrc[j*dim+2] == 180)){
    cuPrintf("i=%d, pos=%ld, %e, %e, %e\n", i, pos,
      axx[(READ_STEP-1)*npsrc+j], ayy[(READ_STEP-1)*npsrc+j], azz[(READ_STEP-1)*npsrc+j]);
      }*/
    }
  }

  return;
}


/* kernel function to apply free-surface B.C. to velocities - (Daniel) */
extern "C"
void fvel_H( float *u1, float *v1,  float *w1,  cudaStream_t St, float *lam_mu, int NX, int rankx, int ranky,
             int s_i,   int e_i,    int s_j,    int e_j ) {
  dim3 block( 2, BLOCK_SIZE_Y, 1 );
  dim3 grid( 1, (e_j - s_j + 1 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1 );
  cudaFuncSetCacheConfig( fstr, cudaFuncCachePreferL1 );
  fvel<<<grid, block, 0, St>>>( u1, v1, w1, lam_mu, NX, rankx, ranky, s_i, e_i, s_j );

  return;
}

/* kernel functions to apply free-surface B.C.s to velocity */
__global__ void fvel( float *u1, float *v1, float *w1, float *lam_mu, int NX, int rankx, int ranky, int s_i, int e_i, int s_j ) {
  register int    i, j, k;
  //register float w1_im1, w1_im2, u1_ip1, f_u1, f_v1, f_w1;
  //register int pos, pos_km1, pos_kp1, pos_kp2, pos_jm1, pos_jp1, pos_im1;
  register int    g_i;
  register float  vs1, vs2;

//  register int    pos_ip1, pos_km2, pos_kp2, pos_jm2, pos_jp2, pos_ik1, pos_jk1;
  register int    pos,     pos_im2, pos_im1;
  register int    pos_km1, pos_kp1;
  register int    pos_jm1, pos_jp1;
//  register int    pos_ijk, pos_ijk1, f_ww;

//  register float  u1_ip2, v1_ip1;
  register float  f_u1, u1_ip1, u1_im1;
  register float  f_v1, v1_im1, v1_im2;
  register float  f_w1, w1_im1, w1_im2, w1_ip1;

  k    = d_nzt + ALIGN - 1;
  j    = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y + s_j;
  i    = e_i;
  pos  = i * d_slice_1 + j * d_yline_1 + k;

  u1_ip1 = u1[pos+d_slice_2];
  f_u1   = u1[pos+d_slice_1];
  u1_im1 = u1[pos];
  f_v1   = v1[pos+d_slice_1];
  v1_im1 = v1[pos];
  v1_im2 = v1[pos-d_slice_1];
  f_w1   = w1[pos+d_slice_1];
  w1_im1 = w1[pos];
  w1_im2 = w1[pos-d_slice_1];

  for( i = e_i; i >= s_i; i-- ) {
//    pos_km2  = pos - 2;
    pos_km1  = pos - 1;
    pos_kp1  = pos + 1;
//    pos_kp2  = pos + 2;
//    pos_jm2  = pos - d_yline_2;
    pos_jm1  = pos - d_yline_1;
    pos_jp1  = pos + d_yline_1;
//    pos_jp2  = pos + d_yline_2;
    pos_im2  = pos - d_slice_2;
    pos_im1  = pos - d_slice_1;
//    pos_ip1  = pos + d_slice_1;
//    pos_jk1  = pos - d_yline_1 - 1;
//    pos_ik1  = pos + d_slice_1 - 1;
//    pos_ijk  = pos + d_slice_1 - d_yline_1;
//    pos_ijk1 = pos + d_slice_1 - d_yline_1 - 1;

//    u1_ip2   = u1_ip1;
    u1_ip1   = f_u1;
    f_u1     = u1_im1;
    u1_im1   = u1[pos_im1];
//    v1_ip1   = f_v1;
    f_v1     = v1_im1;
    v1_im1   = v1_im2;
    v1_im2   = v1[pos_im2];
    w1_ip1   = f_w1;
    f_w1     = w1_im1;
    w1_im1   = w1_im2;
    w1_im2   = w1[pos_im2];

    u1[pos_kp1] = f_u1 - (f_w1 - w1_im1);
    v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);

    g_i  = d_nxt * rankx + i - 4 * LOOP - 1;

    if( g_i < NX )
      vs1 = u1_ip1 - (w1_ip1 - f_w1);
    else
      vs1 = 0.0;

    g_i   = d_nyt * ranky + j - 4 * LOOP - 1;
    if( g_i > 1 )
      vs2 = v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
    else
      vs2 = 0.0;

    w1[pos_kp1] = w1[pos_km1] - lam_mu[i*(d_nyt+4+8*LOOP)+j] * ((vs1         - u1[pos_kp1]) + (u1_ip1 - f_u1)
                                                             + (v1[pos_kp1] - vs2)         + (f_v1   - v1[pos_jm1]));

    pos         = pos_im1;
  }
}

extern "C"
void update_yldfac_buffer_x_H( float *yldfac, float *buf_L, float *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2,
                               int rank_L,    int rank_R,   int meshtp ) {
  if( rank_L == -1 && rank_R == -1 )
    return;

  /*dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
  dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nyt+8*LOOP+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y, 1);*/
  dim3 block( 1, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
  dim3 grid( 1, (nyt + 8 * LOOP + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z );
  //cudaPrintfInit();
  CUCHK( cudaFuncSetCacheConfig( update_yldfac_buffer_x, cudaFuncCachePreferL1 ) );
  update_yldfac_buffer_x<<<grid, block, 0, St1>>>( yldfac, buf_L, rank_L, Left, meshtp );
  CUCHK( cudaGetLastError() );
  update_yldfac_buffer_x<<<grid, block, 0, St2>>>( yldfac, buf_R, rank_R, Right, meshtp );
  CUCHK( cudaGetLastError() );
  //cudaPrintfDisplay(stdout, 1);
  //cudaPrintfEnd();

  return;
}

/* buffer exchanged for the swap area */
__global__ void update_yldfac_buffer_x( float *yldfac, float *buf, int rank, int flag, int meshtp ) {
  register int i, j, k, pos, bpos;
  register int b_slice_1, b_yline_1;
  register int xs, xe, zs;
  register int nxt, nyt, nzt, slice_1, yline_1;

  if( meshtp == 0 ) {
    nyt     = d_nyt;
    nxt     = d_nxt;
    nzt     = d_nzt;
    slice_1 = d_slice_1;
    yline_1 = d_yline_1;
  }

  b_slice_1  = (nyt + 8 * LOOP) * nzt;
  b_yline_1  = nzt;

  j     = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
  k     = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z + ALIGN;
  zs    = ALIGN;

  if( flag == Left ) {
    xs  = 2 + 4 * LOOP;
    xe  = xs + 4 * LOOP;
  } else if( flag == Right ) {
    xs  = nxt + 2;
    xe  = xs + 4 * LOOP;
  }

  if( rank != -1 ) {
    for( i = xs; i < xe; i++ ) {
      pos       = i * slice_1 + j * yline_1 + k;

      bpos      = (i - xs) * b_slice_1 + j * b_yline_1 + (k - zs);
      buf[bpos] = yldfac[pos];

      /*if (((flag == Right) && (i==103)) && ((j==46) && (k==132))){
         cuPrintf("swap send: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
      }*/
    }
  }

  return;
}

extern "C"
void update_yldfac_data_x_H( float *yldfac, float *buf_L, float *buf_R, int nyt, int nzt, cudaStream_t St1, cudaStream_t St2,
                             int rank_L,    int rank_R,   int meshtp ) {
  if( rank_L == -1 && rank_R == -1 )
    return;

  dim3 block( 1, BLOCK_SIZE_Y, BLOCK_SIZE_Z );
  dim3 grid( 1, (nyt + 8 * LOOP + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z );
  //cudaPrintfInit();
  CUCHK( cudaFuncSetCacheConfig( update_yldfac_buffer_x, cudaFuncCachePreferL1 ) );
  update_yldfac_data_x<<<grid, block, 0, St1>>>( yldfac, buf_L, rank_L, Left, meshtp );
  CUCHK( cudaGetLastError() );
  update_yldfac_data_x<<<grid, block, 0, St2>>>( yldfac, buf_R, rank_R, Right, meshtp );
  CUCHK( cudaGetLastError() );
  //cudaPrintfDisplay(stdout, 1);
  //cudaPrintfEnd();

  return;
}

/* copy exchanged buffer data back to swap zone*/
__global__ void update_yldfac_data_x( float *yldfac, float *buf, int rank, int flag, int meshtp ) {
  register int i, j, k, pos, bpos;
  register int b_slice_1, b_yline_1;
  register int xs, xe, zs;
  register int nxt, nyt, nzt, slice_1, yline_1;

  if( meshtp == 0 ) {
    nyt = d_nyt;
    nxt = d_nxt;
    nzt = d_nzt;
    slice_1 = d_slice_1;
    yline_1 = d_yline_1;
  }

  b_slice_1  = (nyt + 8 * LOOP) * nzt;
  b_yline_1  = nzt;

  j   = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
  k   = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z + ALIGN;
  zs  = ALIGN;

  if( flag == Left ) {
    xs  = 2;
    xe  = 2 + 4 * LOOP;
  } else if( flag == Right ) {
    xs  = nxt + 2 + 4 * LOOP;
    xe  = xs + 4 * LOOP;
  }

  if( rank != -1 ) {
    for( i = xs; i < xe; i++ ) {
      pos         = i * slice_1 + j * yline_1 + k;

      bpos        = (i - xs) * b_slice_1 + j * b_yline_1 + (k - zs);
      yldfac[pos] = buf[bpos];

      /*if (((flag == Left) && (i==3)) && ((j==46) && (k==132))){
         cuPrintf("swap recv: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
      }*/
    }
  }

  return;
}

extern "C"
void update_yldfac_buffer_y_H( float *yldfac,     float *buf_F,     float *buf_B, int nxt,    int nzt,
                               cudaStream_t St1,  cudaStream_t St2, int rank_F,   int rank_B, int meshtp ) {
  if( rank_F == -1 && rank_B == -1 )
    return;

  dim3 block( BLOCK_SIZE_X, 1, BLOCK_SIZE_Z );
  dim3 grid( (nxt + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, 1, (nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z );
  //cudaPrintfInit();
  CUCHK( cudaFuncSetCacheConfig( update_yldfac_buffer_y, cudaFuncCachePreferL1 ) );
  update_yldfac_buffer_y<<<grid, block, 0, St1>>>( yldfac, buf_F, rank_F, Front, meshtp );
  CUCHK( cudaGetLastError() );
  update_yldfac_buffer_y<<<grid, block, 0, St2>>>( yldfac, buf_B, rank_B, Back, meshtp );
  CUCHK( cudaGetLastError() );
  /*cudaPrintfDisplay(stdout, 1);
  cudaPrintfEnd();*/

  return;
}

/* buffer exchanged for the swap area along Y*/
__global__ void update_yldfac_buffer_y( float *yldfac, float *buf, int rank, int flag, int meshtp ) {
  register int i, j, k, pos, bpos;
  register int b_slice_1, b_yline_1;
  register int ys, ye, xs, zs;
  register int nyt, nzt, slice_1, yline_1;

  if( meshtp == 0 ) {
    nyt     = d_nyt;
    nzt     = d_nzt;
    slice_1 = d_slice_1;
    yline_1 = d_yline_1;
  }

  b_slice_1 = 4 * LOOP * nzt;
  b_yline_1 = nzt;

  i   = blockIdx.x * BLOCK_SIZE_X + threadIdx.x + 2 + 4 * LOOP;
  k   = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z + ALIGN;

  xs  = 2 + 4 * LOOP;
  zs  = ALIGN;

  if( flag == Front ) {
    ys  = 2 + 4 * LOOP;
    ye  = ys + 4 * LOOP;
  } else if( flag == Back ) {
    ys  = nyt + 2;
    ye  = ys + 4 * LOOP;
  }

  if( rank != -1 ) {
    for( j = ys; j < ye; j++ ) {
      pos       = i * slice_1 + j * yline_1 + k;

      bpos      = (i - xs) * b_slice_1 + (j - ys) * b_yline_1 + (k - zs);
      buf[bpos] = yldfac[pos];

      /*if (((flag == Back) && (i==103)) && ((j==33) && (k==132))){
         cuPrintf("swap send: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
      }*/
    }
  }

  return;
}

extern "C"
void update_yldfac_data_y_H( float *yldfac,     float *buf_F,     float *buf_B, int nxt,    int nzt,
                             cudaStream_t St1,  cudaStream_t St2, int rank_F,   int rank_B, int meshtp ) {
  if( rank_F == -1 && rank_B == -1)
    return;

  dim3 block( BLOCK_SIZE_X, 1, BLOCK_SIZE_Z );
  dim3 grid( (nxt + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, 1,(nzt + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z );
  //cudaPrintfInit();
  CUCHK( cudaFuncSetCacheConfig( update_yldfac_buffer_y, cudaFuncCachePreferL1 ) );
  update_yldfac_data_y<<<grid, block, 0, St1>>>( yldfac, buf_F, rank_F, Front, meshtp );
  CUCHK( cudaGetLastError() );
  update_yldfac_data_y<<<grid, block, 0, St2>>>( yldfac, buf_B, rank_B, Back, meshtp );
  CUCHK( cudaGetLastError() );
  /*cudaPrintfDisplay(stdout, 1);
  cudaPrintfEnd();*/

  return;
}

/* copy exchanged buffer data back to swap zone*/
__global__ void update_yldfac_data_y( float *yldfac, float *buf, int rank, int flag, int meshtp ) {
  register int i, j, k, pos, bpos;
  register int b_slice_1, b_yline_1;
  register int ys, ye, xs, zs;
  register int nyt, nzt, slice_1, yline_1;
//  register int nxt;

  if( meshtp == 0 ) {
    nyt     = d_nyt;
//    nxt     = d_nxt;
    nzt     = d_nzt;
    slice_1 = d_slice_1;
    yline_1 = d_yline_1;
  }

  b_slice_1 = 4 * LOOP * nzt;
  b_yline_1 = nzt;

  i   = blockIdx.x * BLOCK_SIZE_X + threadIdx.x + 2 + 4 * LOOP;
  k   = blockIdx.z * BLOCK_SIZE_Z + threadIdx.z + ALIGN;

  xs  = 2 + 4 * LOOP;
  zs  = ALIGN;

  if( flag == Front ) {
    ys  = 2;
    ye  = ys + 4 * LOOP;
  } else if( flag == Back ) {
    ys  = nyt + 2 + 4 * LOOP;
    ye  = ys + 4 * LOOP;
  }

  if( rank != -1 ) {
    for( j = ys; j < ye; j++ ) {
      pos         = i * slice_1 + j * yline_1 + k;

      bpos        = (i - xs) * b_slice_1 + (j - ys) * b_yline_1 + (k - zs);
      yldfac[pos] = buf[bpos];

      /*if (((flag == Front) && (i==103)) && ((j==3) && (k==132))){
         cuPrintf("swap recv: buf[%d] = %.16g, yldfac[%d] = %.16g\n", bpos, buf[bpos], pos, yldfac[pos]);
      }*/
    }
  }

  return;
}
