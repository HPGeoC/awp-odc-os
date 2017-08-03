/**
 @brief CUDA kernel functions for finite difference stencils.
 
 @section LICENSE
 Copyright (c) 2013-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include "kernel.h"
#include "pmcl3d_cons.h"

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

texture<float, 1, cudaReadModeElementType> p_vx1;
texture<float, 1, cudaReadModeElementType> p_vx2;
texture<int, 1, cudaReadModeElementType> p_ww;
texture<float, 1, cudaReadModeElementType> p_wwo;

extern "C"
void SetDeviceConstValue(float DH, float DT, int nxt, int nyt, int nzt)
{
    float h_c1, h_c2, h_dth, h_dt1, h_dh1;
    int   slice_1,  slice_2,  yline_1,  yline_2;
    h_c1  = 9.0/8.0;
    h_c2  = -1.0/24.0;
    h_dth = DT/DH;
    h_dt1 = 1.0/DT;
    h_dh1 = 1.0/DH;
    slice_1  = (nyt+4+8*LOOP)*(nzt+2*ALIGN);
    slice_2  = (nyt+4+8*LOOP)*(nzt+2*ALIGN)*2;
    yline_1  = nzt+2*ALIGN;
    yline_2  = (nzt+2*ALIGN)*2;
  
    cudaMemcpyToSymbol(d_c1,      &h_c1,    sizeof(float));
    cudaMemcpyToSymbol(d_c2,      &h_c2,    sizeof(float));
    cudaMemcpyToSymbol(d_dth,     &h_dth,   sizeof(float));
    cudaMemcpyToSymbol(d_dt1,     &h_dt1,   sizeof(float));
    cudaMemcpyToSymbol(d_dh1,     &h_dh1,   sizeof(float));
    cudaMemcpyToSymbol(d_DT,      &DT,      sizeof(float));
    cudaMemcpyToSymbol(d_DH,      &DH,      sizeof(float));
    cudaMemcpyToSymbol(d_nxt,     &nxt,     sizeof(int));
    cudaMemcpyToSymbol(d_nyt,     &nyt,     sizeof(int));
    cudaMemcpyToSymbol(d_nzt,     &nzt,     sizeof(int));
    cudaMemcpyToSymbol(d_slice_1, &slice_1, sizeof(int));
    cudaMemcpyToSymbol(d_slice_2, &slice_2, sizeof(int));
    cudaMemcpyToSymbol(d_yline_1, &yline_1, sizeof(int));
    cudaMemcpyToSymbol(d_yline_2, &yline_2, sizeof(int));
    return;
}

extern "C"
void BindArrayToTexture(float* vx1, float* vx2,int* ww, float* wwo, int memsize)
{
   cudaBindTexture(0, p_vx1,  vx1,  memsize);
   cudaBindTexture(0, p_vx2,  vx2,  memsize);
   cudaBindTexture(0, p_ww,   ww,   memsize);
   cudaBindTexture(0, p_wwo,   wwo,   memsize);
   cudaThreadSynchronize ();
   return;
}

extern "C"
void UnBindArrayFromTexture()
{
   cudaUnbindTexture(p_vx1);
   cudaUnbindTexture(p_vx2);
   cudaUnbindTexture(p_ww);
   cudaUnbindTexture(p_wwo);
   return;
}

extern "C"
void dvelcx_H(float* u1,    float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy,      float* xz, float* yz,
             float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, int nyt,   int nzt,   cudaStream_t St, int s_i,   int e_i)
{
    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nyt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dvelcx, cudaFuncCachePreferL1);
    dvelcx<<<grid, block, 0, St>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_i, e_i);
    return;
}

extern "C"
void dvelcy_H(float* u1,       float* v1,    float* w1,    float* xx,  float* yy, float* zz, float* xy,   float* xz,   float* yz,
              float* dcrjx,    float* dcrjy, float* dcrjz, float* d_1, int nxt,   int nzt,   float* s_u1, float* s_v1, float* s_w1,  
              cudaStream_t St, int s_j,      int e_j,      int rank)
{
    if(rank==-1) return;
    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nxt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dvelcy, cudaFuncCachePreferL1);
    dvelcy<<<grid, block, 0, St>>>(u1, v1, w1, xx, yy, zz, xy, xz, yz, dcrjx, dcrjy, dcrjz, d_1, s_u1, s_v1, s_w1, s_j, e_j);
    return;
}

extern "C"
void update_bound_y_H(float* u1,   float* v1, float* w1, float* f_u1,      float* f_v1,      float* f_w1,  float* b_u1, float* b_v1, 
                      float* b_w1, int nxt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_f,  int rank_b)
{
     if(rank_f==-1 && rank_b==-1) return;
     dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
     dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (nxt+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
     cudaFuncSetCacheConfig(update_boundary_y, cudaFuncCachePreferL1);
     update_boundary_y<<<grid, block, 0, St1>>>(u1, v1, w1, f_u1, f_v1, f_w1, rank_f, Front);
     update_boundary_y<<<grid, block, 0, St2>>>(u1, v1, w1, b_u1, b_v1, b_w1, rank_b, Back);
     return;
}

extern "C"
void dstrqc_H(float* xx,       float* yy,     float* zz,    float* xy,    float* xz, float* yz,
              float* r1,       float* r2,     float* r3,    float* r4,    float* r5, float* r6,
              float* u1,       float* v1,     float* w1,    float* lam,   float* mu, float* qp,float* coeff,
              float* qs,       float* dcrjx,  float* dcrjy, float* dcrjz, int nyt,   int nzt, 
              cudaStream_t St, float* lam_mu, int NX,       int rankx,    int ranky, int  s_i,  
              int e_i,         int s_j,       int e_j)
{
    dim3 block (BLOCK_SIZE_Z, BLOCK_SIZE_Y, 1);
    dim3 grid ((nzt+BLOCK_SIZE_Z-1)/BLOCK_SIZE_Z, (e_j-s_j+1+BLOCK_SIZE_Y-1)/BLOCK_SIZE_Y,1);
    cudaFuncSetCacheConfig(dstrqc, cudaFuncCachePreferL1);
    dstrqc<<<grid, block, 0, St>>>(xx,    yy,    zz,  xy,  xz, yz, r1, r2,    r3,    r4,    r5,     r6, 
                                   u1,    v1,    w1,  lam, mu, qp,coeff, qs, dcrjx, dcrjy, dcrjz, lam_mu, NX, 
                                   rankx, ranky, s_i, e_i, s_j);
    return;
}

extern "C"
void addsrc_H(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,  cudaStream_t St,
              float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
              float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz)
{
    dim3 grid, block;
    if(npsrc < 256)
    {
       block.x = npsrc;
       grid.x = 1;
    }
    else
    {
       block.x = 256;
       grid.x  = int((npsrc+255)/256);
    }
    cudaError_t cerr;
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addsrc before kernel: %s\n",cudaGetErrorString(cerr));
    addsrc_cu<<<grid, block, 0, St>>>(i,  READ_STEP, dim, psrc, npsrc, axx, ayy, azz, axz, ayz, axy,
                                      xx, yy,        zz,  xy,   yz,  xz);
    cerr=cudaGetLastError();
    if(cerr!=cudaSuccess) printf("CUDA ERROR: addsrc after kernel: %s\n",cudaGetErrorString(cerr));
    return;
}


__global__ void dvelcx(float* u1,    float* v1,    float* w1,    float* xx, float* yy, float* zz, float* xy, float* xz, float* yz, 
                      float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, int s_i,   int e_i)
{
    register int   i, j, k, pos,     pos_im1, pos_im2;
    register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
    register int   pos_jm2, pos_jm1, pos_jp1, pos_jp2;
    register int   pos_ip1, pos_jk1, pos_ik1, pos_ijk;
    register float f_xx,    xx_im1,  xx_ip1,  xx_im2;
    register float f_xy,    xy_ip1,  xy_ip2,  xy_im1;
    register float f_xz,    xz_ip1,  xz_ip2,  xz_im1;
    register float f_d1,    f_d2,    f_d3,    f_dcrj, f_dcrjy, f_dcrjz, f_yz;

    // At node (i,j,k)
    k    = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+ALIGN;
    j    = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+4*LOOP;
    i    = e_i;
    
    // One dimensional index of node (i,j,k)
    pos  = i*d_slice_1+j*d_yline_1+k;
    
    /*
     Indexing Notes:
     [pos] -> (i,j,k)
     [pos+d_slice_1] -> (i+1, j, k)
     [pos+d_slice_2] -> (i+2, j, k)
     etc.
     */

    // Stress tensor components
    f_xx    = xx[pos+d_slice_1];    // xx(i+1,j,k)
    xx_im1  = xx[pos];              // xx(i,j,k)
    xx_im2  = xx[pos-d_slice_1];    // xx(i-1,j,k)
    xy_ip1  = xy[pos+d_slice_2];    // xy(i+2,j,k)
    f_xy    = xy[pos+d_slice_1];    // xy(i+1,j,k)
    xy_im1  = xy[pos];              // xy(i,j,k)
    xz_ip1  = xz[pos+d_slice_2];    // xz(i+2,j,k)
    f_xz    = xz[pos+d_slice_1];    // xz(i+1,j,k)
    xz_im1  = xz[pos];              // xz(i,j,k)
    
    
    f_dcrjz = dcrjz[k];             // absorbing boundary condition (Cerjan)
    f_dcrjy = dcrjy[j];             // absorbing boundary condition (Cerjan)
    for(i=e_i;i>=s_i;i--)   
    {
        pos_km2  = pos-2;   // (i,j,k-2)
        pos_km1  = pos-1;   // (i,j,k-1)
        pos_kp1  = pos+1;   // (i,j,k+1)
        pos_kp2  = pos+2;   // (i,j,k+2)
        pos_jm2  = pos-d_yline_2;   // (i,j-2,k)
        pos_jm1  = pos-d_yline_1;   // (i,j-1,k)
        pos_jp1  = pos+d_yline_1;   // (i,j+1,k)
        pos_jp2  = pos+d_yline_2;   // (i,j+2,k)
        pos_im1  = pos-d_slice_1;   // (i-1,j,k)
        pos_im2  = pos-d_slice_2;   // (i-2,j,k)
        pos_ip1  = pos+d_slice_1;   // (i+1,j,k)
        pos_jk1  = pos-d_yline_1-1; // (i,j-1,k-1)
        pos_ik1  = pos+d_slice_1-1; // (i+1,j,k-1)
        pos_ijk  = pos+d_slice_1-d_yline_1; //(i+1,j-1,k)

        xx_ip1   = f_xx;        // xx(i+1,j,k)
        f_xx     = xx_im1;      // xx(i,j,k)
        xx_im1   = xx_im2;      // xx(i-1,j,k)
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

        f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;
        f_d1     = 0.25*(d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
        f_d2     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
        f_d3     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);

        f_d1     = d_dth/f_d1;
        f_d2     = d_dth/f_d2;
	f_d3     = d_dth/f_d3;

    	u1[pos]  = (u1[pos] + f_d1*( d_c1*(f_xx        - xx_im1)      + d_c2*(xx_ip1      - xx_im2) 
                                   + d_c1*(f_xy        - xy[pos_jm1]) + d_c2*(xy[pos_jp1] - xy[pos_jm2])
                                   + d_c1*(f_xz        - xz[pos_km1]) + d_c2*(xz[pos_kp1] - xz[pos_km2]) ))*f_dcrj; 
        v1[pos]  = (v1[pos] + f_d2*( d_c1*(xy_ip1      - f_xy)        + d_c2*(xy_ip2      - xy_im1)
                                   + d_c1*(yy[pos_jp1] - yy[pos])     + d_c2*(yy[pos_jp2] - yy[pos_jm1])
                                   + d_c1*(f_yz        - yz[pos_km1]) + d_c2*(yz[pos_kp1] - yz[pos_km2]) ))*f_dcrj;

        w1[pos]  = (w1[pos] + f_d3*( d_c1*(xz_ip1      - f_xz)        + d_c2*(xz_ip2      - xz_im1)
                                   + d_c1*(f_yz        - yz[pos_jm1]) + d_c2*(yz[pos_jp1] - yz[pos_jm2])
                                   + d_c1*(zz[pos_kp1] - zz[pos])     + d_c2*(zz[pos_kp2] - zz[pos_km1]) ))*f_dcrj;
        pos      = pos_im1;
    }

    return;
}


__global__ void dvelcy(float* u1,    float* v1,    float* w1,    float* xx,  float* yy,   float* zz,   float* xy, float* xz, float* yz,
                       float* dcrjx, float* dcrjy, float* dcrjz, float* d_1, float* s_u1, float* s_v1, float* s_w1, int s_j,   int e_j)
{
    register int   i, j, k, pos,     j2,      pos2, pos_jm1, pos_jm2;
    register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
    register int   pos_im2, pos_im1, pos_ip1, pos_ip2;
    register int   pos_jk1, pos_ik1, pos_ijk;
    register float f_xy,    xy_jp1,  xy_jm1,  xy_jm2;
    register float f_yy,    yy_jp2,  yy_jp1,  yy_jm1;
    register float f_yz,    yz_jp1,  yz_jm1,  yz_jm2;
    register float f_d1,    f_d2,    f_d3,    f_dcrj, f_dcrjx, f_dcrjz, f_xz;

    k     = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+ALIGN;
    i     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+4*LOOP;
    j     = e_j;
    j2    = 4*LOOP-1;
    pos   = i*d_slice_1+j*d_yline_1+k;
    pos2  = i*4*LOOP*d_yline_1+j2*d_yline_1+k; 

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
    for(j=e_j; j>=s_j; j--)
    {
        pos_km2  = pos-2;
        pos_km1  = pos-1;
        pos_kp1  = pos+1;
        pos_kp2  = pos+2;
        pos_jm2  = pos-d_yline_2;
        pos_jm1  = pos-d_yline_1;
        pos_im1  = pos-d_slice_1;
        pos_im2  = pos-d_slice_2;
        pos_ip1  = pos+d_slice_1;
        pos_ip2  = pos+d_slice_2;
        pos_jk1  = pos-d_yline_1-1;
        pos_ik1  = pos+d_slice_1-1;
        pos_ijk  = pos+d_slice_1-d_yline_1;

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

        f_dcrj   = f_dcrjx*dcrjy[j]*f_dcrjz;
        f_d1     = 0.25*(d_1[pos] + d_1[pos_jm1] + d_1[pos_km1] + d_1[pos_jk1]);
        f_d2     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_km1] + d_1[pos_ik1]);
        f_d3     = 0.25*(d_1[pos] + d_1[pos_ip1] + d_1[pos_jm1] + d_1[pos_ijk]);

        f_d1     = d_dth/f_d1;
        f_d2     = d_dth/f_d2;
        f_d3     = d_dth/f_d3;

        s_u1[pos2] = (u1[pos] + f_d1*( d_c1*(xx[pos]     - xx[pos_im1]) + d_c2*(xx[pos_ip1] - xx[pos_im2])
                                     + d_c1*(f_xy        - xy_jm1)      + d_c2*(xy_jp1      - xy_jm2)
                                     + d_c1*(f_xz        - xz[pos_km1]) + d_c2*(xz[pos_kp1] - xz[pos_km2]) ))*f_dcrj;
        s_v1[pos2] = (v1[pos] + f_d2*( d_c1*(xy[pos_ip1] - f_xy)        + d_c2*(xy[pos_ip2] - xy[pos_im1])
                                     + d_c1*(yy_jp1      - f_yy)        + d_c2*(yy_jp2      - yy_jm1)
                                     + d_c1*(f_yz        - yz[pos_km1]) + d_c2*(yz[pos_kp1] - yz[pos_km2]) ))*f_dcrj;
        s_w1[pos2] = (w1[pos] + f_d3*( d_c1*(xz[pos_ip1] - f_xz)        + d_c2*(xz[pos_ip2] - xz[pos_im1])
                                     + d_c1*(f_yz        - yz_jm1)      + d_c2*(yz_jp1      - yz_jm2)
                                     + d_c1*(zz[pos_kp1] - zz[pos])     + d_c2*(zz[pos_kp2] - zz[pos_km1]) ))*f_dcrj;

        pos        = pos_jm1;
        pos2       = pos2 - d_yline_1;
    }
    return;
}

__global__ void update_boundary_y(float* u1, float* v1, float* w1, float* s_u1, float* s_v1, float* s_w1, int rank, int flag)
{
    register int i, j, k, pos, posj;
    k     = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+ALIGN;
    i     = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+2+4*LOOP;

    if(flag==Front && rank!=-1){
	j     = 2;
    	pos   = i*d_slice_1+j*d_yline_1+k;
        posj  = i*4*LOOP*d_yline_1+k;
	for(j=2;j<2+4*LOOP;j++){
		u1[pos] = s_u1[posj];
		v1[pos] = s_v1[posj];
		w1[pos] = s_w1[posj];
		pos	= pos  + d_yline_1;
  		posj	= posj + d_yline_1;	
	}
    }

    if(flag==Back && rank!=-1){
    	j     = d_nyt+4*LOOP+2;
    	pos   = i*d_slice_1+j*d_yline_1+k;
        posj  = i*4*LOOP*d_yline_1+k;
	for(j=d_nyt+4*LOOP+2;j<d_nyt+8*LOOP+2;j++){
	        u1[pos] = s_u1[posj];
                v1[pos] = s_v1[posj];
                w1[pos] = s_w1[posj];
                pos     = pos  + d_yline_1;
                posj    = posj + d_yline_1;
	}
    }
    return;
}

__global__ void dstrqc(float* xx, float* yy,    float* zz,    float* xy,    float* xz,     float* yz,
                       float* r1, float* r2,    float* r3,    float* r4,    float* r5,     float* r6,
                       float* u1, float* v1,    float* w1,    float* lam,   float* mu,     float* qp,float* coeff,
                       float* qs, float* dcrjx, float* dcrjy, float* dcrjz, float* lam_mu, int NX,    
                       int rankx, int ranky,    int s_i,      int e_i,      int s_j)
{
    register int   i,  j,  k,  g_i;
    register int   pos,     pos_ip1, pos_im2, pos_im1;
    register int   pos_km2, pos_km1, pos_kp1, pos_kp2;
    register int   pos_jm2, pos_jm1, pos_jp1, pos_jp2;
    register int   pos_ik1, pos_jk1, pos_ijk, pos_ijk1,f_ww;
    register float vs1, vs2, vs3, a1, tmp, vx1,f_wwo;
    register float xl,  xm,  xmu1, xmu2, xmu3;
    register float qpa, h,   h1,   h2,   h3;
    register float qpaw,hw,h1w,h2w,h3w;
    register float f_vx1, f_vx2,  f_dcrj, f_r,  f_dcrjy, f_dcrjz;
    register float f_rtmp;
    register float f_u1, u1_ip1, u1_ip2, u1_im1;
    register float f_v1, v1_im1, v1_ip1, v1_im2;
    register float f_w1, w1_im1, w1_im2, w1_ip1;
    
    k    = blockIdx.x*BLOCK_SIZE_Z+threadIdx.x+ALIGN;
    j    = blockIdx.y*BLOCK_SIZE_Y+threadIdx.y+s_j;
    i    = e_i;
    pos  = i*d_slice_1+j*d_yline_1+k;

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
    for(i=e_i;i>=s_i;i--)
    {
        
        // fvx_1 = tau_1
        f_vx1    = tex1Dfetch(p_vx1, pos);
        
        // fvx_2 = tau_2
        f_vx2    = tex1Dfetch(p_vx2, pos);
        
        // f_ww = weight index
        f_ww     = tex1Dfetch(p_ww, pos);
        
        // f_wwo = weights
	f_wwo     = tex1Dfetch(p_wwo, pos);
/*
        if(f_wwo!=f_wwo){ 
          xx[pos] = yy[pos] = zz[pos] = xy[pos] = xz[pos] = yz[pos] = 1.0;
          r1[pos] = r2[pos] = r3[pos] = r4[pos] = r5[pos] = r6[pos] = 1.0;
          return;
        }
*/
        f_dcrj   = dcrjx[i]*f_dcrjy*f_dcrjz;

        pos_km2  = pos-2;
        pos_km1  = pos-1;
        pos_kp1  = pos+1;
        pos_kp2  = pos+2;
        pos_jm2  = pos-d_yline_2;
        pos_jm1  = pos-d_yline_1;
        pos_jp1  = pos+d_yline_1;
        pos_jp2  = pos+d_yline_2;
        pos_im2  = pos-d_slice_2;
        pos_im1  = pos-d_slice_1;
        pos_ip1  = pos+d_slice_1;
        pos_jk1  = pos-d_yline_1-1;
        pos_ik1  = pos+d_slice_1-1;
        pos_ijk  = pos+d_slice_1-d_yline_1;
        pos_ijk1 = pos+d_slice_1-d_yline_1-1;

        // xl = avg of 1/lambda
        xl       = 8.0/(  lam[pos]      + lam[pos_ip1] + lam[pos_jm1] + lam[pos_ijk]
                        + lam[pos_km1]  + lam[pos_ik1] + lam[pos_jk1] + lam[pos_ijk1] );
        
        // xm = avg of 2/mu
        xm       = 16.0/( mu[pos]       + mu[pos_ip1]  + mu[pos_jm1]  + mu[pos_ijk]
                        + mu[pos_km1]   + mu[pos_ik1]  + mu[pos_jk1]  + mu[pos_ijk1] );
        
        // avg of 1/mu
        xmu1     = 2.0/(  mu[pos]       + mu[pos_km1] );
        
        // avg of 1/mu
        xmu2     = 2.0/(  mu[pos]       + mu[pos_jm1] );
        
        // avg of 1/mu
        xmu3     = 2.0/(  mu[pos]       + mu[pos_ip1] );
        
        // avg of 1/lambda + 2/mu
        xl       = xl  +  xm;
        
        //avg of 1/2 * Qp
        qpa      = 0.0625*( qp[pos]     + qp[pos_ip1] + qp[pos_jm1] + qp[pos_ijk]
                          + qp[pos_km1] + qp[pos_ik1] + qp[pos_jk1] + qp[pos_ijk1] );
//			  www=f_ww;
        if(1./(qpa*2.0)<=200.0)
        {
            //	printf("coeff[f_ww*2-2] %g\n",coeff[f_ww*2-2]);
            qpaw=coeff[f_ww*2-2]*(2.*qpa)*(2.*qpa)+coeff[f_ww*2-1]*(2.*qpa);
            //	        qpaw=coeff[www*2-2]*(2.*qpa)*(2.*qpa)+coeff[www*2-1]*(2.*qpa);
            //		  qpaw=qpaw/2.;
        }
        else {
            qpaw  = f_wwo*qpa;
        }
        //	           printf("qpaw %f\n",qpaw);
        //		printf("qpaw1 %g\n",qpaw);
        
        
        // qpaw = some multiple of coefficients and Qp (call it Ap)
        qpaw=qpaw/f_wwo;
        //	printf("qpaw2 %g\n",qpaw);
        
        
        // avg of 1/2 * Qs
        h        = 0.0625*( qs[pos]     + qs[pos_ip1] + qs[pos_jm1] + qs[pos_ijk]
                           + qs[pos_km1] + qs[pos_ik1] + qs[pos_jk1] + qs[pos_ijk1] );
        
        if(1./(h*2.0)<=200.0)
        {
            hw=coeff[f_ww*2-2]*(2.*h)*(2.*h)+coeff[f_ww*2-1]*(2.*h);
            //                  hw=hw/2.;
        }
        else {
            hw  = f_wwo*h;
        }
        
        // hw is some multiple of coefficients and Qs (call it As)
        hw=hw/f_wwo;
        
        // h1 = avg of 1/2 * Qs
        h1       = 0.250*(  qs[pos]     + qs[pos_km1] );
        if(1./(h1*2.0)<=200.0)
        {
            h1w=coeff[f_ww*2-2]*(2.*h1)*(2.*h1)+coeff[f_ww*2-1]*(2.*h1);
            //                  h1w=h1w/2.;
        }
        else {
            h1w  = f_wwo*h1;
        }
        
        // h1w is some multiple of coefficients and Qs
        h1w=h1w/f_wwo;
        
        // h2 = avg of 1/2 * Qs
        h2       = 0.250*(  qs[pos]     + qs[pos_jm1] );
        if(1./(h2*2.0)<=200.0)
        {
            h2w=coeff[f_ww*2-2]*(2.*h2)*(2.*h2)+coeff[f_ww*2-1]*(2.*h2);
            //                  h2w=h2w/2.;
        }
        else {
            h2w  = f_wwo*h2;
        }
        
        // h2w is some multiple of coefficients and Qs
        h2w=h2w/f_wwo;
        
        // h3 = avg of 1/2 * Qs
        h3       = 0.250*(  qs[pos]     + qs[pos_ip1] );
        if(1./(h3*2.0)<=200.0)
        {
            h3w=coeff[f_ww*2-2]*(2.*h3)*(2.*h3)+coeff[f_ww*2-1]*(2.*h3);
            //                  h3w=h3w/2.;
        }
        else {
            h3w  = f_wwo*h3;
        }
        
        // h3 is some multiple of coefficients and Qs
        h3w=h3w/f_wwo;
        
        // h = -(2/mu) * As * (1/dh)
        h        = -xm*hw*d_dh1;
        
        // h1 = -(1/mu) * As * (1/dh)
        h1       = -xmu1*h1w*d_dh1;
        
        // h2 = -(1/mu) * As * (1/dh)
        h2       = -xmu2*h2w*d_dh1;
        
        // h3 = -(1/mu) * As * (1/dh)
        h3       = -xmu3*h3w*d_dh1;


	//        h1       = -xmu1*hw1*d_dh1;
        //h2       = -xmu2*hw2*d_dh1;
        //h3       = -xmu3*hw3*d_dh1;

        //qpa =  -Ap*(1/lambda + 2/mu)*(1/dh)
        qpa      = -qpaw*xl*d_dh1;
	//        qpa      = -qpaw*xl*d_dh1;
        
        // xm = (2/mu)*(dt/dh)
        xm       = xm*d_dth;
        
        // xmu1 = (1/mu) *(dt/dh)
        xmu1     = xmu1*d_dth;
        
        // xmu2 = (1/mu) *(dt/dh)
        
        xmu2     = xmu2*d_dth;
        
        // xmu3 = (1/mu) *(dt/dh)
        xmu3     = xmu3*d_dth;
        
        // xl = (1/lambda + 2/mu ) * (dt/dh)
        xl       = xl*d_dth;
        
        //f_vx2    = f_vx2*f_vx1;
        
        // h = tau_1 * -(2/mu) * f(Qs) * (1/dh)
        h        = h*f_vx1;
        
        // h1 = tau_1 * -(1/mu) * f(Qs) * (1/dh)
        h1       = h1*f_vx1;
        
        // h2 = tau_1 * -(1/mu) * f(Qs) * (1/dh)
        h2       = h2*f_vx1;
        
        // h3 = tau_1 * -(1/mu) * f(Qs) * (1/dh)
        h3       = h3*f_vx1;

        // qpa = -Ap*(1/lambda + 2/mu)*(1/dh)*tau1
        qpa      = qpa*f_vx1;

        // xm = (2/mu)*(dt/dh) + tau_qs * -(2/mu) * f(Qs) * (dt/dh)
        xm       = xm+d_DT*h;
        xmu1     = xmu1+d_DT*h1;
        xmu2     = xmu2+d_DT*h2;
        xmu3     = xmu3+d_DT*h3;
        
        // vx1 = dt * (1 + tau_qp * tau_qs)
        vx1      = d_DT*(1+f_vx2*f_vx1);
        
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
        
        
        //
        if(k == d_nzt+ALIGN-1)
        {
            u1[pos_kp1] = f_u1 - (f_w1        - w1_im1);
            v1[pos_kp1] = f_v1 - (w1[pos_jp1] - f_w1);
            
            g_i  = d_nxt*rankx + i - 4*LOOP - 1;
            
            if(g_i<NX)
                vs1	= u1_ip1 - (w1_ip1    - f_w1);
            else
                vs1	= 0.0;
            
            g_i  = d_nyt*ranky + j - 4*LOOP - 1;
            if(g_i>1)
                vs2	= v1[pos_jm1] - (f_w1 - w1[pos_jm1]);
            else
                vs2	= 0.0;
            
            w1[pos_kp1]	= w1[pos_km1] - lam_mu[i*(d_nyt+4+8*LOOP) + j]*((vs1         - u1[pos_kp1]) + (u1_ip1 - f_u1)
                                                                        +     			                (v1[pos_kp1] - vs2)         + (f_v1   - v1[pos_jm1]) );
        }
        else if(k == d_nzt+ALIGN-2)
        {
            u1[pos_kp2] = u1[pos_kp1] - (w1[pos_kp1]   - w1[pos_im1+1]);
            v1[pos_kp2] = v1[pos_kp1] - (w1[pos_jp1+1] - w1[pos_kp1]);
        }
        
        // vs1 = Dx
        vs1      = d_c1*(u1_ip1 - f_u1)        + d_c2*(u1_ip2      - u1_im1);
        
        // vs2 = Dy
        vs2      = d_c1*(f_v1   - v1[pos_jm1]) + d_c2*(v1[pos_jp1] - v1[pos_jm2]);
        
        // vs3 = Dz
        vs3      = d_c1*(f_w1   - w1[pos_km1]) + d_c2*(w1[pos_kp1] - w1[pos_km2]);
        
        
        // xl = (1/lambda + 2/mu)*dt/dh
        // tmp = (Dx + Dy + Dz) * (1/lambda + 2/mu)
        tmp      = xl*(vs1+vs2+vs3);
        
        // qpa = -f(Qp)*(1/lambda + 2/mu)*(1/dh)
        // a1 = (Dx + Dy + Dz) * -f(Qp)*(1/lambda + 2/mu) * 1/dh
        a1       = qpa*(vs1+vs2+vs3);
        
        // tmp = (Dx + Dy + Dz) * (1/lambda + 2/mu) * [1 - f(Qp) * dt/dh]
        tmp      = tmp+d_DT*a1;
        
        // modified for q(f)
        //           f_wwo     = f_wwo*2.;
        //	     a1=a1*2.;
        //	     h=h*2.;
        //	     h1=h1*2.;
        //	     h2=h2*2.;
        //	     h3=h3*2.;
        
        
        // mem var for xx
        f_r      = r1[pos];
        
        // (Dx + Dy + Dz) * -f(Qp)*(1/lambda + 2/mu) * 1/dh + (tau_qs * (2/mu) * f(Qs) * (1/dh)) * (Dy + Dz)
        f_rtmp   = -h*(vs2+vs3) + a1;
        
        
        
        xx[pos]  = xx[pos]  + tmp - xm*(vs2+vs3) + vx1*f_r;
        r1[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
        //KBW          r1[pos]  = f_vx2*f_r + f_rtmp;
        //r1[pos]  = f_vx2*f_r - f_wwo*h*(vs2+vs3)        + f_wwo*a1;
        f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
        xx[pos]  = (xx[pos] + d_DT*f_rtmp)*f_dcrj;
        
        
        
        f_r      = r2[pos];
        f_rtmp   = -h*(vs1+vs3) + a1;
        yy[pos]  = yy[pos]  + tmp - xm*(vs1+vs3) + vx1*f_r;
        r2[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
        f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
        yy[pos]  = (yy[pos] + d_DT*f_rtmp)*f_dcrj;
        
        f_r      = r3[pos];
        f_rtmp   = -h*(vs1+vs2) + a1;
        zz[pos]  = zz[pos]  + tmp - xm*(vs1+vs2) + vx1*f_r;
        r3[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
        f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
        zz[pos]  = (zz[pos] + d_DT*f_rtmp)*f_dcrj;
        
        vs1      = d_c1*(u1[pos_jp1] - f_u1)   + d_c2*(u1[pos_jp2] - u1[pos_jm1]);
        vs2      = d_c1*(f_v1        - v1_im1) + d_c2*(v1_ip1      - v1_im2);
        
        
        f_r      = r4[pos];
        f_rtmp   = h1*(vs1+vs2);
        xy[pos]  = xy[pos]  + xmu1*(vs1+vs2) + vx1*f_r;
        r4[pos]  = f_vx2*f_r + f_wwo*f_rtmp;
        f_rtmp   = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
        xy[pos]  = (xy[pos] + d_DT*f_rtmp)*f_dcrj;
        
        if(k == d_nzt+ALIGN-1)
        {
            zz[pos+1] = -zz[pos];
            xz[pos]   = 0.0;
            yz[pos]   = 0.0;
        }
        else
        {
            // modified for q(f)
            vs1     = d_c1*(u1[pos_kp1] - f_u1)   + d_c2*(u1[pos_kp2] - u1[pos_km1]);
            vs2     = d_c1*(f_w1        - w1_im1) + d_c2*(w1_ip1      - w1_im2);
            f_r     = r5[pos];
            f_rtmp  = h2*(vs1+vs2);
            xz[pos] = xz[pos]  + xmu2*(vs1+vs2) + vx1*f_r;
            r5[pos] = f_vx2*f_r + f_wwo*f_rtmp;
            //          f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
            //kBW	    	    r5[pos] = f_vx2*f_r + f_rtmp;
            f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
            xz[pos] = (xz[pos] + d_DT*f_rtmp)*f_dcrj;
            
            
            vs1     = d_c1*(v1[pos_kp1] - f_v1) + d_c2*(v1[pos_kp2] - v1[pos_km1]);
            vs2     = d_c1*(w1[pos_jp1] - f_w1) + d_c2*(w1[pos_jp2] - w1[pos_jm1]);
            
            
            f_r     = r6[pos];
            f_rtmp  = h3*(vs1+vs2);
            yz[pos] = yz[pos]  + xmu3*(vs1+vs2) + vx1*f_r;
            r6[pos] = f_vx2*f_r + f_wwo*f_rtmp;
            f_rtmp  = f_rtmp*(f_wwo-1) + f_vx2*f_r*(1-f_vx1);
            yz[pos] = (yz[pos] + d_DT*f_rtmp)*f_dcrj;
            
            if(k == d_nzt+ALIGN-2)
            {
                zz[pos+3] = -zz[pos];
                xz[pos+2] = -xz[pos];
                yz[pos+2] = -yz[pos];
            }
            else if(k == d_nzt+ALIGN-3)
            {
                xz[pos+4] = -xz[pos];
                yz[pos+4] = -yz[pos];
            }
        }
        pos     = pos_im1;
    }
    return;
}


__global__ void addsrc_cu(int i,      int READ_STEP, int dim,    int* psrc,  int npsrc,
                          float* axx, float* ayy,    float* azz, float* axz, float* ayz, float* axy,
                          float* xx,  float* yy,     float* zz,  float* xy,  float* yz,  float* xz)
{
        register float vtst;
        register int idx, idy, idz, j, pos;
        j = blockIdx.x*blockDim.x+threadIdx.x;
        if(j >= npsrc) return;
        vtst = (float)d_DT/(d_DH*d_DH*d_DH);

        i   = i - 1;
        idx = psrc[j*dim]   + 1 + 4*LOOP;
        idy = psrc[j*dim+1] + 1 + 4*LOOP;
        idz = psrc[j*dim+2] + ALIGN - 1;
        pos = idx*d_slice_1 + idy*d_yline_1 + idz;

        xx[pos] = xx[pos] - vtst*axx[j*READ_STEP+i];
        yy[pos] = yy[pos] - vtst*ayy[j*READ_STEP+i];
        zz[pos] = zz[pos] - vtst*azz[j*READ_STEP+i];
        xz[pos] = xz[pos] - vtst*axz[j*READ_STEP+i];
        yz[pos] = yz[pos] - vtst*ayz[j*READ_STEP+i];
        xy[pos] = xy[pos] - vtst*axy[j*READ_STEP+i];

        return;
}
