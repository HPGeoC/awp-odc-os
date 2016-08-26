/**
   @file Mesh.cpp
   @brief Contains functions for reading input mesh data from files and setting up corresponding data structures.
 
   @section LICENSE
 
   Copyright (c) 2013-2016, Regents of the University of California
   All rights reserved.
 
   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
   1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
   2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
*/

#include "Mesh.hpp"
#include "data/common.hpp"
#include "Grid.hpp"

/**************************************************************************
 * Efecan updated on Oct 4, 2012
 *    MEDIARESTART=3 is added for partitioned large mesh reading
 *
 ***************************************************************************/
#include <cstdio>
#include <cmath>
#include <complex>

#include <time.h>
#include <sys/time.h>

#ifdef YASK
using namespace yask;
#endif

double w_time()
{
  struct timeval t;
  if(gettimeofday(&t, NULL))
  {
    return 0;
  }
  return (double) t.tv_sec + 0.000001 * (double) t.tv_usec;
}

static real anelastic_coeff(real q, int_pt weight_index, real weight, real *coeff) {
    if(1.0/q <= 200.0) {
        q = (coeff[weight_index*2-2]*q*q + coeff[weight_index*2-1]*q)/weight;
    } else {
        q *= 0.5;
    }
    return q;
}



void odc::data::Mesh::initialize(odc::io::OptionParser i_options, int_pt x, int_pt y, int_pt z,
                                 int_pt bdry_size, bool anelastic, Grid1D i_inputBuffer,
                                 int_pt i_globalX, int_pt i_globalY, int_pt i_globalZ
#ifdef YASK
                                 , Grid_XYZ* density_grid, Grid_XYZ* mu_grid, Grid_XYZ* lam_grid,
                                 Grid_XYZ* weights_grid, Grid_XYZ* tau2_grid, Grid_XYZ* an_ap_grid,
                                 Grid_XYZ* an_as_grid, Grid_XYZ* an_xy_grid, Grid_XYZ* an_xz_grid,
                                 Grid_XYZ* an_yz_grid
#endif
                                 )
{
  real taumax, taumin;
    
  Grid3D tau = Alloc3D(2, 2, 2);
  Grid3D tau1 = Alloc3D(2, 2, 2);
  Grid3D tau2 = Alloc3D(2, 2, 2);
  Grid3D weights = Alloc3D(2, 2, 2);

  int_pt totalX = x + 2 * bdry_size;
  int_pt totalY = y + 2 * bdry_size;
  int_pt totalZ = z + 2 * bdry_size;
    
    
  m_usingAnelastic = anelastic;
#ifndef YASK
  m_density = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
  m_mu = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
  m_lam = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
#endif
  m_lam_mu = odc::data::Alloc3D(totalX, totalY, 1, odc::constants::boundary);

  if(m_usingAnelastic)
  {
    m_qp = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
    m_qs = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
    m_tau1 = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
    m_tau2 = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
    m_weights = odc::data::Alloc3D(totalX, totalY, totalZ, odc::constants::boundary);
    m_weight_index = odc::data::Alloc3Dww(totalX, totalY, totalZ, odc::constants::boundary);
    m_coeff = Alloc1D(16);
    weights_sub(weights, m_coeff, i_options.m_ex, i_options.m_fac);
  }

    
  new_inimesh(i_options.m_mediaStart,
#ifdef YASK
              density_grid,
              mu_grid,
              lam_grid,
              bdry_size,
#else
              &m_density[bdry_size][bdry_size][bdry_size],
              &m_mu[bdry_size][bdry_size][bdry_size],
              &m_lam[bdry_size][bdry_size][bdry_size],
#endif
              
              &m_qp[bdry_size][bdry_size][bdry_size],
              &m_qs[bdry_size][bdry_size][bdry_size],
              (totalY + 2*odc::constants::boundary)*(totalZ + 2*odc::constants::boundary),
              totalZ + 2*odc::constants::boundary,
              1,
              &taumax,
              &taumin,
              tau,
              weights,
              m_coeff,
              i_options.m_nVar,
              i_options.m_fp,
              i_options.m_fac,
              i_options.m_q0,
              i_options.m_ex,
              x,
              y,
              z,
              i_options.m_nX,
              i_options.m_nY,
              i_options.m_nZ, 
              i_options.m_iDyna,
              i_options.m_nVe,
              i_options.m_soCalQ,
              i_inputBuffer,
              i_options.m_nX,
              i_options.m_nY,
              i_options.m_nZ
              );

  set_boundaries(
#ifdef YASK
      density_grid, mu_grid, lam_grid,
#endif
      m_density, m_mu, m_lam, m_qp, m_qs, m_usingAnelastic, bdry_size, x, y, z);

  // For free surface boundary condition calculations
  for(int i = bdry_size; i < bdry_size + x; i++) {
    for(int j = bdry_size; j < bdry_size + y ;j++) {
      real t_xl, t_xl2m;
#ifdef YASK
      t_xl             = 1.0/lam_grid->readElem(i,j,bdry_size+z-1,0);
      t_xl2m           = 2.0/mu_grid->readElem(i,j,bdry_size+z-1,0) + t_xl;      
#else
      t_xl             = 1.0/m_lam[i][j][bdry_size+z-1];
      t_xl2m           = 2.0/m_mu[i][j][bdry_size+z-1] + t_xl;
#endif
      m_lam_mu[i][j][0]  = t_xl/t_xl2m;
    }
  }
    
  if(anelastic)
  {
    // note that tau here will be the same for every patch
    for(int i=0; i<2; i++)
    {
      for(int j=0; j<2; j++)
      {
        for(int k=0; k<2; k++)
        {
          real tauu     = tau[i][j][k]; 
          tau2[i][j][k] = exp(-i_options.m_dT/tauu);
          tau1[i][j][k] = 0.5*(1.-tau2[i][j][k]);
        }
      }
    }

    new_init_texture(tau1, tau2, m_tau1, m_tau2, weights, m_weight_index, m_weights,
                     bdry_size, bdry_size+x, bdry_size, bdry_size+y, bdry_size, bdry_size+z,
                     i_globalX, i_globalY, i_globalZ,
                     i_options.m_nZ);
  }
    
  Delloc3D(tau);
  Delloc3D(tau1);
  Delloc3D(tau2);
  Delloc3D(weights);

#ifdef YASK
  
  for(int_pt tx=-1+bdry_size; tx<x+1+bdry_size; tx++)
  {
    for(int_pt ty=-1+bdry_size; ty<y+1+bdry_size; ty++)
    {
      for(int_pt tz=-1+bdry_size; tz<z+1+bdry_size; tz++)
      {
        real local_qp = 0.125*(m_qp[tx][ty][tz] + m_qp[tx+1][ty][tz] + m_qp[tx][ty-1][tz] + m_qp[tx+1][ty-1][tz] + m_qp[tx][ty][tz-1] +
                               m_qp[tx+1][ty][tz-1] + m_qp[tx][ty-1][tz-1] + m_qp[tx+1][ty-1][tz-1]);          
        real local_qs_diag = 0.125*(m_qs[tx][ty][tz] + m_qs[tx+1][ty][tz] + m_qs[tx][ty-1][tz] + m_qs[tx+1][ty-1][tz] + m_qs[tx][ty][tz-1] +
                                    m_qs[tx+1][ty][tz-1] + m_qs[tx][ty-1][tz-1] + m_qs[tx+1][ty-1][tz-1]);
        real local_qs_xy = 0.5*(m_qs[tx][ty][tz] + m_qs[tx][ty][tz-1]);
        real local_qs_xz = 0.5*(m_qs[tx][ty][tz] + m_qs[tx][ty-1][tz]);
        real local_qs_yz = 0.5*(m_qs[tx][ty][tz] + m_qs[tx+1][ty][tz]);

        an_ap_grid->writeElem(anelastic_coeff(local_qp, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff),
                              tx, ty, tz, 0);
        an_as_grid->writeElem(anelastic_coeff(local_qs_diag, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff),
                              tx, ty, tz, 0);
        an_xy_grid->writeElem(anelastic_coeff(local_qs_xy, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff),
                              tx, ty, tz, 0);
        an_xz_grid->writeElem(anelastic_coeff(local_qs_xz, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff),
                              tx, ty, tz, 0);
        an_yz_grid->writeElem(anelastic_coeff(local_qs_yz, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff),
                              tx, ty, tz, 0);

        
        weights_grid->writeElem(m_weights[tx][ty][tz], tx, ty, tz, 0);
        tau2_grid->writeElem(m_tau2[tx][ty][tz], tx, ty, tz, 0);
        
      }
    }
  }
  
  if(m_usingAnelastic)
  {
    odc::data::Delloc3D(m_qp, 2);
    odc::data::Delloc3D(m_qs, 2);
    odc::data::Delloc3D(m_tau1, 2);
    odc::data::Delloc3D(m_tau2, 2);
    odc::data::Delloc3D(m_weights, 2);
        
    odc::data::Delloc3Dww(m_weight_index, 2);
  }
#endif
  
}


odc::data::Mesh::Mesh(odc::io::OptionParser i_options, odc::data::SoA data)
{
    
  real taumax, taumin;
    
  Grid3D tau = Alloc3D(2, 2, 2);
  Grid3D tau1 = Alloc3D(2, 2, 2);
  Grid3D tau2 = Alloc3D(2, 2, 2);
  Grid3D weights = Alloc3D(2, 2, 2);
    
  real tauu;
    
  m_density = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
  m_mu = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
  m_lam = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
  m_lam_mu = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, 1, odc::constants::boundary);
    
  if (i_options.m_nVe == 1)
  {
    m_usingAnelastic = true;
    // If doing anelastic attenuation
    m_qp = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
    m_qs = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
    m_tau1 = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
    m_tau2 = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
    m_weights = odc::data::Alloc3D(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
    m_weight_index = odc::data::Alloc3Dww(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::constants::boundary);
    m_coeff = Alloc1D(16);
    weights_sub(weights, m_coeff, i_options.m_ex, i_options.m_fac);
  }
    
  inimesh(i_options.m_mediaStart, m_density, m_mu, m_lam, m_qp, m_qs, &taumax, &taumin, tau, weights, m_coeff, i_options.m_nVar, i_options.m_fp, i_options.m_fac, i_options.m_q0, i_options.m_ex, data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, 1, 1, data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, odc::parallel::Mpi::coords, MPI_COMM_WORLD, i_options.m_iDyna, i_options.m_nVe, i_options.m_soCalQ, i_options.m_inVel, m_vse, m_vpe, m_dde);
    
    
  // For free surface boundary condition calculations
  for(int i=0;i<data.m_numXGridPoints;i++)
  {
    for(int j=0;j<data.m_numYGridPoints;j++)
    {
      real t_xl, t_xl2m;
      t_xl             = 1.0/m_lam[i][j][data.m_numZGridPoints-1];
      t_xl2m           = 2.0/m_mu[i][j][data.m_numZGridPoints-1] + t_xl;
      m_lam_mu[i][j][0]  = t_xl/t_xl2m;
    }
  }
    
  if(i_options.m_nVe == 1)
  {
    // If doing anelastic attenuation
    for(int i=0; i<2; i++)
    {
      for(int j=0; j<2; j++)
      {
        for(int k=0; k<2; k++)
        {
          tauu          = tau[i][j][k];
          tau2[i][j][k] = exp(-i_options.m_dT/tauu);
          tau1[i][j][k] = 0.5*(1.-tau2[i][j][k]);
        }
      }
    }
        
    init_texture(data.m_numXGridPoints, data.m_numYGridPoints, data.m_numZGridPoints, tau1, tau2, m_tau1, m_tau2, weights, m_weight_index, m_weights, 0, data.m_numXGridPoints-1, 0, data.m_numYGridPoints-1);
  }

    
  Delloc3D(tau);
  Delloc3D(tau1);
  Delloc3D(tau2);
  Delloc3D(weights);
}


void odc::data::Mesh::finalize()
{
#ifndef YASK  
  odc::data::Delloc3D(m_density, 2);
  odc::data::Delloc3D(m_lam, 2);
  odc::data::Delloc3D(m_mu, 2);
#endif
  odc::data::Delloc3D(m_lam_mu, 2);
    
  if (m_usingAnelastic)
  {
#ifndef YASK    
    odc::data::Delloc3D(m_qp, 2);
    odc::data::Delloc3D(m_qs, 2);
    odc::data::Delloc3D(m_tau1, 2);
    odc::data::Delloc3D(m_tau2, 2);
    odc::data::Delloc3D(m_weights, 2);
        
    odc::data::Delloc3Dww(m_weight_index, 2);
#endif
    
    Delloc1D(m_coeff);
  }
    
}


void odc::data::Mesh::inimesh(int MEDIASTART, Grid3D d1, Grid3D mu, Grid3D lam, Grid3D qp, Grid3D qs, float *taumax, float *taumin,
                              Grid3D tau, Grid3D weights,Grid1D coeff,
                              int nvar, float FP,  float FAC, float Q0, float EX, int nxt, int nyt, int nzt, int PX, int PY, int NX, int NY,
                              int NZ, int *coords, MPI_Comm MCW, int IDYNA, int NVE, int SoCalQ, char *INVEL,
                              float *vse, float *vpe, float *dde)
{
  printf("start of inimesh\n");
  double stime = 0., etime = 0.;
    
  int merr;
  int rank;
  int_pt err;
  float vp,vs,dd,pi;
  int   rmtype[3], rptype[3], roffset[3];
  MPI_Datatype readtype;
  MPI_Status   filestatus;
  MPI_File     fh;
  char mpiErrStr[100];
  int mpiErrStrLen;
  int_pt num_pts = (int_pt) nxt * (int_pt) nyt * (int_pt) nzt;
    
    
  pi      = 4.*atan(1.);
  //  *taumax = 1./(2*pi*0.01)*10.0*FAC;
  *taumax = 1./(2*pi*0.01)*1.0*FAC;
  if(EX<0.65 && EX>=0.01) *taumin = 1./(2*pi*10.0)*0.2*FAC;
  else if(EX<0.85 && EX>=0.65)
  {
    *taumin = 1./(2*pi*12.0)*0.5*FAC;
    *taumax = 1./(2*pi*0.08)*2.0*FAC;
  }
  else if (EX<0.95 && EX>=0.85)
  {
    //(EX<0.95 && EX>=0.85) *taumin = 1./(2*pi*280.0)*0.1*FAC;
    //  else if(EX<0.01) *taumin = 1./(2*pi*40.0)*0.1*FAC;
    *taumin = 1./(2*pi*15.0)*0.8*FAC;
    *taumax = 1./(2*pi*0.1)*2.5*FAC;
  }
    
  else if(EX<0.01) *taumin = 1./(2*pi*10.0)*0.2*FAC;
    
    
  tausub(tau, *taumin, *taumax);
  if(!coords[0] && !coords[1])
    printf("tau: %e,%e; %e,%e; %e,%e; %e,%e\n",
           tau[0][0][0],tau[1][0][0],tau[0][1][0],tau[1][1][0],
           tau[0][0][1],tau[1][0][1],tau[0][1][1],tau[1][1][1]);
  MPI_Comm_rank(MCW,&rank);
  if(MEDIASTART==0)
  {
    //*taumax = 1./(2*pi*0.01)*10.0*FAC;
    //*taumin = 1./(2*pi*400.0)*0.1*FAC;
    if(IDYNA==1)
    {
      vp=6000.0;
      vs=3464.0;
      dd=2670.0;
    }
    else
    {
      vp=4800.0;
      vs=2800.0;
      dd=2500.0;
    }
        
    for(int_pt i=0;i<nxt;i++)
    {
      for(int_pt j=0;j<nyt;j++)
      {
        for(int_pt k=0;k<nzt;k++)
        {
          lam[i][j][k]=1./(dd*(vp*vp - 2.*vs*vs));
          mu[i][j][k]=1./(dd*vs*vs);
          d1[i][j][k]=dd;
        }
      }
    }
  }
  else
  {
    Grid3D tmpvp=NULL, tmpvs=NULL, tmpdd=NULL;
    Grid3D tmppq=NULL, tmpsq=NULL;
    int var_offset;

    stime = w_time();
    printf("inimesh: Allocating....\n");
    tmpvp = Alloc3D(nxt, nyt, nzt);
    tmpvs = Alloc3D(nxt, nyt, nzt);
    tmpdd = Alloc3D(nxt, nyt, nzt);
    printf("inimesh: done allocating, took %lf\n", w_time()-stime);

    stime = w_time();
    printf("inimesh: Initializing....\n");
#pragma omp parallel for collapse(3)
    for(int_pt i=0;i<nxt;i++)
    {
      for(int_pt j=0;j<nyt;j++)
      {
        for(int_pt k=0;k<nzt;k++)
        {
          tmpvp[i][j][k]=0.0f;
          tmpvs[i][j][k]=0.0f;
          tmpdd[i][j][k]=0.0f;
        }
      }
    }
    printf("inimesh: done initializing, took %lf\n", w_time()-stime);
        
        
    if(NVE==1)
    {
      tmppq = Alloc3D(nxt, nyt, nzt);
      tmpsq = Alloc3D(nxt, nyt, nzt);
      stime = w_time();
      printf("inimesh: Initializing 2....\n");
            
#pragma omp parallel for collapse(3)
      for(int_pt i=0;i<nxt;i++)
      {
        for(int_pt j=0;j<nyt;j++)
        {
          for(int_pt k=0;k<nzt;k++)
          {
            tmppq[i][j][k]=0.0f;
            tmpsq[i][j][k]=0.0f;
          }
        }
      }
      printf("inimesh: done initializing 2, took %lf\n", w_time()-stime);
            
    }
        
    if(nvar==8)
    {
      var_offset=3;
    }
    else if(nvar==5)
    {
      var_offset=0;
    }
    else
    {
      var_offset=0;
    }
        
    if(MEDIASTART>=1 && MEDIASTART<=3)
    {
      char filename[200];
      if(MEDIASTART<3) sprintf(filename,"%s",INVEL);
      else if(MEDIASTART==3)
      {
        sprintf(filename,"input_rst/mediapart/media%07d.bin",rank);
        if(rank%100==0) printf("Rank=%d, reading file=%s\n",rank,filename);
      }
      Grid1D tmpta = Alloc1D(nvar*num_pts);
      if(MEDIASTART==3 || (PX==1 && PY==1))
      {
        FILE   *file;
        file = fopen(filename,"rb");
        if(!file)
        {
          printf("can't open file %s", filename);
          return;
        }
        if(!fread(tmpta,sizeof(float),nvar*num_pts,file))
        {
          printf("can't read file %s", filename);
          return;
        }
        //printf("%d) 0-0-0,1-10-3=%f, %f\n",rank,tmpta[0],tmpta[1+10*nxt+3*nxt*nyt]);
      }
      else
      {
        printf("MPI not implemented in CPU code, quitting.");
        abort();
        /*printf("%d) Media file will be read using MPI-IO\n", rank);
          rmtype[0]  = NZ;
          rmtype[1]  = NY;
          rmtype[2]  = NX*nvar;
          rptype[0]  = nzt;
          rptype[1]  = nyt;
          rptype[2]  = nxt*nvar;
          roffset[0] = 0;
          roffset[1] = nyt*coords[1];
          roffset[2] = nxt*coords[0]*nvar;
          err = MPI_Type_create_subarray(3, rmtype, rptype, roffset, MPI_ORDER_C, MPI_FLOAT, &readtype);
          if(err != MPI_SUCCESS){
          MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
          printf("%d) ERROR! MPI-IO mesh reading create subarray: %s\n",rank,mpiErrStr);
          }
          err = MPI_Type_commit(&readtype);
          if(err != MPI_SUCCESS){
          MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
          printf("%d) ERROR! MPI-IO mesh reading commit: %s\n",rank,mpiErrStr);
          }
          err = MPI_File_open(MCW,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
          if(err != MPI_SUCCESS){
          MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
          printf("%d) ERROR! MPI-IO mesh reading file open: %s\n",rank,mpiErrStr);
          }
          err = MPI_File_set_view(fh, 0, MPI_FLOAT, readtype, "native", MPI_INFO_NULL);
          if(err != MPI_SUCCESS){
          MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
          printf("%d) ERROR! MPI-IO mesh reading file set view: %s\n",rank,mpiErrStr);
          }
          err = MPI_File_read_all(fh, tmpta, nvar*nxt*nyt*nzt, MPI_FLOAT, &filestatus);
          if(err != MPI_SUCCESS){
          MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
          printf("%d) ERROR! MPI-IO mesh reading file read: %s\n",rank,mpiErrStr);
          }
          err = MPI_File_close(&fh);
          if(err != MPI_SUCCESS){
          MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
          printf("%d) ERROR! MPI-IO mesh reading file close: %s\n",rank,mpiErrStr);
          }
          if(!rank) printf("Media file is read using MPI-IO\n");*/
      }
            
      stime = w_time();
      printf("inimesh: Starting comp 1....\n");

#pragma omp parallel for collapse(3)
      for(int_pt k=0;k<nzt;k++)
      {
        for(int_pt j=0;j<nyt;j++)
        {
          for(int_pt i=0;i<nxt;i++)
          {
            tmpvp[i][j][k]=tmpta[(k*nyt*(int_pt) nxt+j*nxt+i)*nvar+var_offset];
            tmpvs[i][j][k]=tmpta[(k*nyt*nxt+j*(int_pt) nxt+i)*nvar+var_offset+1];
            tmpdd[i][j][k]=tmpta[(k*nyt*nxt+j*(int_pt) nxt+i)*nvar+var_offset+2];
                        
            if(nvar>3)
            {
              tmppq[i][j][k]=tmpta[(k*nyt*(int_pt)nxt+j*nxt+i)*nvar+var_offset+3];
              tmpsq[i][j][k]=tmpta[(k*nyt*(int_pt)nxt+j*nxt+i)*nvar+var_offset+4];
            }
            if(tmpvp[i][j][k]!=tmpvp[i][j][k] ||
               tmpvs[i][j][k]!=tmpvs[i][j][k] ||
               tmpdd[i][j][k]!=tmpdd[i][j][k]){
              printf("%d) tmpvp,vs,dd is NAN!\n",rank);
              MPI_Abort(MPI_COMM_WORLD,1);
            }
          }
        }
      }
      printf("inimesh: done comp 1, took %lf\n", w_time()-stime);
            
      //printf("%d) vp,vs,dd[0^3]=%f,%f,%f\n",rank,tmpvp[0][0][0],
      //   tmpvs[0][0][0], tmpdd[0][0][0]);
      Delloc1D(tmpta);
    }
    /*
      if(nvar==3 && NVE==1)
      {
      for(i=0;i<nxt;i++)
      for(j=0;j<nyt;j++){
      for(k=0;k<nzt;k++){
      tmpsq[i][j][k]=0.05*tmpvs[i][j][k];
      tmppq[i][j][k]=2.0*tmpsq[i][j][k];
      //tmpsq[i][j][k] = 50.0;
      //tmppq[i][j][k] = 50.0;
      }
      }
      }
    */
    float w0=0.0f, ww1=0.0f, w2=0.0f, tmp1=0.0f, tmp2=0.0f;
    if(NVE==1)
    {
      w0=2*pi*FP;
      //ww1=2*pi*FL;
      //w2=2*pi*FH;
      //*taumax=1./ww1;
      //*taumin=1./w2;
      //tmp1=2./pi*(log((*taumax)/(*taumin)));
      //tmp2=2./pi*log(w0*(*taumin));
      if(!rank) printf("w0 = %g\n",w0);
    }
        
    vse[0] = 1.0e10;
    vpe[0] = 1.0e10;
    dde[0] = 1.0e10;
    vse[1] = -1.0e10;
    vpe[1] = -1.0e10;
    dde[1] = -1.0e10;
    float facex = (float)pow(FAC,EX);
        

#pragma omp parallel for collapse(2)
    for(int_pt i=0;i<nxt;i++)
    {
      for(int_pt j=0;j<nyt;j++)
      {
        float weights_los[2][2][2];
        float weights_lop[2][2][2];
        float val[2];
        float mu1, denom;
        float qpinv=0.0f, qsinv=0.0f, vpvs=0.0f;
        int ii,jj,kk,iii,jjj,kkk,num;
        std::complex<double> value(0.0, 0.0);
        std::complex<double> sqrtm1(0.0, 1.0);

        for(int_pt k=0;k<nzt;k++)
        {
          //printf("iteration: %d,%d,%d\n",i,j,k);
          //tmpvs[i][j][k] = tmpvs[i][j][k]*(1+ ( log(w2/w0) )/(pi*tmpsq[i][j][k]) );
          //tmpvp[i][j][k] = tmpvp[i][j][k]*(1+ ( log(w2/w0) )/(pi*tmppq[i][j][k]) );
          //	    tmpsq[i][j][k] = 20;
          //	    tmppq[i][j][k] = 20;
          if(tmpvs[i][j][k]<200.0)
          {
            tmpvs[i][j][k]=200.0;
            tmpvp[i][j][k]=600.0;
            // tmpsq[i][j][k] = 20;
            // tmppq[i][j][k] = 20;
          }
          tmpsq[i][j][k] = 0.1  * tmpvs[i][j][k];
          tmppq[i][j][k] = 2.0   * tmpsq[i][j][k];
                    
          if(tmppq[i][j][k]>200.0)
          {
            // QF - VP
            val[0] = 0.0;
            val[1] = 0.0;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++){
                  denom = ((w0*w0*tau[ii][jj][kk]*tau[ii][jj][kk]+1.0)*tmppq[i][j][k]*facex);
                  val[0] += weights[ii][jj][kk]/denom;
                  val[1] += -weights[ii][jj][kk]*w0*tau[ii][jj][kk]/denom;
                }
            mu1 = tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k]/(1.0-val[0]);
          }
          else
          {
            //		 if(rank==0) printf("coeff[num]1,2 = %g %g\n",coeff[0],coeff[1]);
            num=0;
            for (iii=0;iii<2;iii++)
              for(jjj=0;jjj<2;jjj++)
                for(kkk=0;kkk<2;kkk++){
                  weights_lop[iii][jjj][kkk]=coeff[num]/(tmppq[i][j][k]*tmppq[i][j][k])+coeff[num+1]/(tmppq[i][j][k]);
                  num=num+2;
                }
            //		 if(rank==0) printf("weights_lop %g\n",weights_lop[0][0][0]);
                        
            value=0.0+0.0*sqrtm1;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  value=value+1.0/(1.0-((double)weights_lop[ii][jj][kk])/(1.0+sqrtm1*((double)w0*tau[ii][jj][kk])));
                }
            value=1./value;
            //		 if(rank==0) printf("creal(value) %f\n",creal(value));
            // if(rank==0) printf("sqrtm1 %f\n",cimag(sqrtm1));
                        
            mu1=tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k]/(8.*std::real(value));
            //	 if(rank==0) printf("mu1 %g\n",mu1);
          }
                    
                    
          tmpvp[i][j][k] = sqrt(mu1/tmpdd[i][j][k]);
                    
                    
                    
                    
          // QF - VS
          if(tmpsq[i][j][k]>200.0)
          {
            val[0] = 0.0;
            val[1] = 0.0;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  denom = ((w0*w0*tau[ii][jj][kk]*tau[ii][jj][kk]+1.0)*tmpsq[i][j][k]*facex);
                  val[0] += weights[ii][jj][kk]/denom;
                  val[1] += -weights[ii][jj][kk]*w0*tau[ii][jj][kk]/denom;
                }
            mu1 = tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]/(1.0-val[0]);
          }
          else
          {
            num=0;
            for (iii=0;iii<2;iii++)
              for(jjj=0;jjj<2;jjj++)
                for(kkk=0;kkk<2;kkk++)
                {
                  weights_los[iii][jjj][kkk]=coeff[num]/(tmpsq[i][j][k]*tmpsq[i][j][k])+coeff[num+1]/(tmpsq[i][j][k]);
                  num=num+2;
                }
            value=0.0+0.0*sqrtm1;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  value=value+1.0/(1.0-((double)weights_los[ii][jj][kk])/(1.0+sqrtm1*((double)w0*tau[ii][jj][kk])));
                                    
                }
            value=1./value;
            mu1=tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]/(8.*std::real(value));
          }
                    
          tmpvs[i][j][k] = sqrt(mu1/tmpdd[i][j][k]);
                    
                    
          // QF - end
          if (SoCalQ==1)
          {
            vpvs=tmpvp[i][j][k]/tmpvs[i][j][k];
            if (vpvs<1.45)  tmpvs[i][j][k]=tmpvp[i][j][k]/1.45;
          }
          if(tmpvp[i][j][k]>7600.0)
          {
            tmpvs[i][j][k]=4387.0;
            tmpvp[i][j][k]=7600.0;
          }
          if(tmpvs[i][j][k]<200.0)
          {
            tmpvs[i][j][k]=200.0;
            tmpvp[i][j][k]=600.0;
          }
          if(tmpdd[i][j][k]<1700.0) tmpdd[i][j][k]=1700.0;
          //printf("tmpvp,tmpvs,tmpdd: %e,%e,%e\n",tmpvp[i][j][k],tmpvs[i][j][k],tmpdd[i][j][k]);
          mu[i][j][(nzt-1) - k]  = 1./(tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]);
          lam[i][j][(nzt-1) - k] = 1./(tmpdd[i][j][k]*(tmpvp[i][j][k]*tmpvp[i][j][k]
                                                       -2.*tmpvs[i][j][k]*tmpvs[i][j][k]));
          d1[i][j][(nzt-1) - k]  = tmpdd[i][j][k];
          if(NVE==1)
          {
            if(tmppq[i][j][k]<=0.0)
            {
              qpinv=0.0;
              qsinv=0.0;
            }
            else
            {
              qpinv=1./tmppq[i][j][k];
              qsinv=1./tmpsq[i][j][k];
            }
            //tmppq[i][j][k]=tmp1*qpinv/(1.0-tmp2*qpinv);
            //tmpsq[i][j][k]=tmp1*qsinv/(1.0-tmp2*qsinv);
            tmppq[i][j][k] = qpinv/facex;
            tmpsq[i][j][k] = qsinv/facex;
            qp[i][j][(nzt-1) - k] = tmppq[i][j][k];
            qs[i][j][(nzt-1) - k] = tmpsq[i][j][k];
          }
          //printf("tmppq,tmpsq: %e,%e\n",tmppq[i][j][k],tmpsq[i][j][k]);
                    
#pragma omp critical
          {
            if(tmpvs[i][j][k]<vse[0]) vse[0] = tmpvs[i][j][k];
            if(tmpvs[i][j][k]>vse[1]) vse[1] = tmpvs[i][j][k];
            if(tmpvp[i][j][k]<vpe[0]) vpe[0] = tmpvp[i][j][k];
            if(tmpvp[i][j][k]>vpe[1]) vpe[1] = tmpvp[i][j][k];
            if(tmpdd[i][j][k]<dde[0]) dde[0] = tmpdd[i][j][k];
            if(tmpdd[i][j][k]>dde[1]) dde[1] = tmpdd[i][j][k];
          }
        }
      }
    }
    Delloc3D(tmpvp);
    Delloc3D(tmpvs);
    Delloc3D(tmpdd);
    if(NVE==1)
    {
      Delloc3D(tmppq);
      Delloc3D(tmpsq);
    }

    // TODO:
    // (gwilkins) Looks like some sort of assignment statements for ghost regions?
    // We don't hard code those so this shouldn't be necessary
    //   (rjtobin)  I think they are necessary
        
    //5 Planes (except upper XY-plane)
        
    for(int_pt j=0;j<nyt;j++)
      for(int_pt k=0;k<nzt;k++)
      {
        lam[-1][j][k]     = lam[0][j][k];
        lam[nxt][j][k] = lam[nxt-1][j][k];
        mu[-1][j][k]      = mu[0][j][k];
        mu[nxt][j][k]  = mu[nxt-1][j][k];
        d1[-1][j][k]      = d1[0][j][k];
        d1[nxt][j][k]  = d1[nxt-1][j][k];
      }
        
    for(int_pt i=0;i<nxt;i++)
      for(int_pt k=0;k<nzt;k++)
      {
        lam[i][-1][k]     = lam[i][0][k];
        lam[i][nyt][k] = lam[i][nyt-1][k];
        mu[i][-1][k]      = mu[i][0][k];
        mu[i][nyt][k]  = mu[i][nyt-1][k];
        d1[i][-1][k]      = d1[i][0][k];
        d1[i][nyt][k]  = d1[i][nyt-1][k];
      }
        
    for(int_pt i=0;i<nxt;i++)
      for(int_pt j=0;j<nyt;j++)
      {
        lam[i][j][-1]   = lam[i][j][0];
        mu[i][j][-1]    = mu[i][j][0];
        d1[i][j][-1]    = d1[i][j][0];
      }
        
    //12 border lines
    for(int_pt i=0;i<nxt;i++)
    {
      lam[i][-1][-1]          = lam[i][0][0];
      mu[i][-1][-1]           = mu[i][0][0];
      d1[i][-1][-1]           = d1[i][0][0];
      lam[i][nyt][-1]         = lam[i][nyt-1][0];
      mu[i][nyt][-1]          = mu[i][nyt-1][0];
      d1[i][nyt][-1]          = d1[i][nyt-1][0];
      lam[i][-1][nzt]         = lam[i][0][nzt-1];
      mu[i][-1][nzt]          = mu[i][0][nzt-1];
      d1[i][-1][nzt]          = d1[i][0][nzt-1];
      lam[i][nyt][nzt]        = lam[i][nyt-1][nzt-1];
      mu[i][nyt][nzt]         = mu[i][nyt-1][nzt-1];
      d1[i][nyt][nzt]         = d1[i][nyt-1][nzt-1];
    }
        
    for(int_pt j=0;j<nyt;j++)
    {
      lam[-1][j][-1]          = lam[0][j][0];
      mu[-1][j][-1]           = mu[0][j][0];
      d1[-1][j][-1]           = d1[0][j][0];
      lam[nxt][j][-1]         = lam[nxt-1][j][0];
      mu[nxt][j][-1]          = mu[nxt-1][j][0];
      d1[nxt][j][-1]          = d1[nxt-1][j][0];
      lam[-1][j][nzt]         = lam[0][j][nzt-1];
      mu[-1][j][nzt]          = mu[0][j][nzt-1];
      d1[-1][j][nzt]          = d1[0][j][nzt-1];
      lam[nxt][j][nzt]        = lam[nxt-1][j][nzt-1];
      mu[nxt][j][nzt]         = mu[nxt-1][j][nzt-1];
      d1[nxt][j][nzt]         = d1[nxt-1][j][nzt-1];
    }
        
    for(int_pt k=0;k<nzt;k++)
    {
      lam[-1][-1][k]          = lam[0][0][k];
      mu[-1][-1][k]           = mu[0][0][k];
      d1[-1][-1][k]           = d1[0][0][k];
      lam[nxt][-1][k]         = lam[nxt-1][0][k];
      mu[nxt][-1][k]          = mu[nxt-1][0][k];
      d1[nxt][-1][k]          = d1[nxt-1][0][k];
      lam[-1][nyt][k]         = lam[0][nyt-1][k];
      mu[-1][nyt][k]          = mu[0][nyt-1][k];
      d1[-1][nyt][k]          = d1[0][nyt-1][k];
      lam[nxt][nyt][k]        = lam[nxt-1][nyt-1][k];
      mu[nxt][nyt][k]         = mu[nxt-1][nyt-1][k];
      d1[nxt][nyt][k]         = d1[nxt-1][nyt-1][k];
    }
        
    //8 Corners
    lam[-1][-1][-1]             = lam[0][0][0];
    mu[-1][-1][-1]              = mu[0][0][0];
    d1[-1][-1][-1]              = d1[0][0][0];
    lam[nxt][-1][-1]            = lam[nxt-1][0][0];
    mu[nxt][-1][-1]             = mu[nxt-1][0][0];
    d1[nxt][-1][-1]             = d1[nxt-1][0][0];
    lam[-1][nyt][-1]            = lam[0][nyt-1][0];
    mu[-1][nyt][-1]             = mu[0][nyt-1][0];
    d1[-1][nyt][-1]             = d1[0][nyt-1][0];
    lam[-1][-1][nzt]            = lam[0][0][nzt-1];
    mu[-1][-1][nzt]             = mu[0][0][nzt-1];
    d1[-1][-1][nzt]             = d1[0][0][nzt-1];
    lam[nxt][-1][nzt]           = lam[nxt-1][0][nzt-1];
    mu[nxt][-1][nzt]            = mu[nxt-1][0][nzt-1];
    d1[nxt][-1][nzt]            = d1[nxt-1][0][nzt-1];
    lam[nxt][nyt][-1]           = lam[nxt-1][nyt-1][0];
    mu[nxt][nyt][-1]            = mu[nxt-1][nyt-1][0];
    d1[nxt][nyt][-1]            = d1[nxt-1][nyt-1][0];
    lam[-1][nyt][nzt]           = lam[0][nyt-1][nzt-1];
    mu[-1][nyt][nzt]            = mu[0][nyt-1][nzt-1];
    d1[-1][nyt][nzt]            = d1[0][nyt-1][nzt-1];
    lam[nxt][nyt][nzt]          = lam[nxt-1][nyt-1][nzt-1];
    mu[nxt][nyt][nzt]           = mu[nxt-1][nyt-1][nzt-1];
    d1[nxt][nyt][nzt]           = d1[nxt-1][nyt-1][nzt-1];
        
        
        
    for(int_pt i=0;i<nxt;i++)
      for(int_pt j=0;j<nyt;j++)
      {
        int_pt k = nzt; 
        d1[i][j][k]   = d1[i][j][k-1];
        mu[i][j][k]   = mu[i][j][k-1];
        lam[i][j][k]  = lam[i][j][k-1];
        if(NVE==1)
        {
          qp[i][j][k] = qp[i][j][k-1];
          qs[i][j][k] = qs[i][j][k-1];
        }
      }
         
        
    if(!coords[0] && !coords[1])
      printf("Before MPI_Allreduce for vpe, vse, dde\n");
    float tmpvse[2],tmpvpe[2],tmpdde[2];
        
    merr = MPI_Allreduce(vse,tmpvse,2,MPI_FLOAT,MPI_MAX,MCW);
    merr = MPI_Allreduce(vpe,tmpvpe,2,MPI_FLOAT,MPI_MAX,MCW);
    merr = MPI_Allreduce(dde,tmpdde,2,MPI_FLOAT,MPI_MAX,MCW);
    vse[1] = tmpvse[1];
    vpe[1] = tmpvpe[1];
    dde[1] = tmpdde[1];
    merr = MPI_Allreduce(vse,tmpvse,2,MPI_FLOAT,MPI_MIN,MCW);
    merr = MPI_Allreduce(vpe,tmpvpe,2,MPI_FLOAT,MPI_MIN,MCW);
    merr = MPI_Allreduce(dde,tmpdde,2,MPI_FLOAT,MPI_MIN,MCW);
    vse[0] = tmpvse[0];
    vpe[0] = tmpvpe[0];
    dde[0] = tmpdde[0];
  }
  printf("end of inimesh\n");
  return;
}// end function inimesh

void odc::data::Mesh::new_inimesh(int MEDIASTART,
#ifdef YASK
                                  Grid_XYZ* density_grid,
                                  Grid_XYZ* mu_grid,
                                  Grid_XYZ* lam_grid,
                                  int_pt bdry_width,
#else                                  
                                  real *d1,
                                  real *mu,
                                  real *lam,
#endif
                                  real *qp,
                                  real *qs,
                                  int_pt i_strideX,
                                  int_pt i_strideY,
                                  int_pt i_strideZ,                                  
                                  float *taumax,
                                  float *taumin,
                                  Grid3D tau,
                                  Grid3D weights,
                                  Grid1D coeff,
                                  int nvar,
                                  float FP,
                                  float FAC,
                                  float Q0,
                                  float EX,
                                  int nxt,
                                  int nyt,
                                  int nzt,
                                  int NX,
                                  int NY,
                                  int NZ,
                                  int IDYNA,
                                  int NVE,
                                  int SoCalQ,
                                  real *i_inputBuffer,
                                  int_pt i_inputSizeX,
                                  int_pt i_inputSizeY,
                                  int_pt i_inputSizeZ
                                  )
{
  printf("start of inimesh\n");
  double stime = 0., etime = 0.;
    
  int merr;
  int_pt err;
  float vp,vs,dd,pi;
  int   rmtype[3], rptype[3], roffset[3];
  int_pt num_pts = (int_pt) nxt * (int_pt) nyt * (int_pt) nzt;
    
    
  pi      = 4.*atan(1.);
  *taumax = 1./(2*pi*0.01)*1.0*FAC;
  if(EX<0.65 && EX>=0.01) *taumin = 1./(2*pi*10.0)*0.2*FAC;
  else if(EX<0.85 && EX>=0.65)
  {
    *taumin = 1./(2*pi*12.0)*0.5*FAC;
    *taumax = 1./(2*pi*0.08)*2.0*FAC;
  }
  else if (EX<0.95 && EX>=0.85)
  {
    *taumin = 1./(2*pi*15.0)*0.8*FAC;
    *taumax = 1./(2*pi*0.1)*2.5*FAC;
  }
    
  else if(EX<0.01) *taumin = 1./(2*pi*10.0)*0.2*FAC;
    
    
  tausub(tau, *taumin, *taumax);
  printf("tau: %e,%e; %e,%e; %e,%e; %e,%e\n",
         tau[0][0][0],tau[1][0][0],tau[0][1][0],tau[1][1][0],
         tau[0][0][1],tau[1][0][1],tau[0][1][1],tau[1][1][1]);

  if(MEDIASTART != 0 && odc::parallel::Mpi::m_size != 1)
  {
    printf("Error: Only MEDIASTART=0 is currently supported by MPI.\n");
    printf("\tReverting to MEDIASTART=0");
    MEDIASTART=0;
  }
  
  if(MEDIASTART==0)
  {
    if(IDYNA==1)
    {
      vp=6000.0;
      vs=3464.0;
      dd=2670.0;
    }
    else
    {
      vp=1800.0; // was 4800.
      vs=1600.0; // was 2800. 
      dd=2500.0;
    }

    m_vse[0] = vs;
    m_vse[1] = vs;
    m_vpe[0] = vp;
    m_vpe[1] = vp;
    m_dde[0] = dd;
    m_dde[1] = dd;
    
    for(int_pt i=-1;i<nxt+1;i++)
      for(int_pt j=-1;j<nyt+1;j++)
        for(int_pt k=-1;k<nzt+1;k++)
        {
          //TODO(Josh): optimize this
          int_pt offset = i * i_strideX + j * i_strideY + k * i_strideZ;

	  if(NVE==1)
	  {
	    qp[offset] = 0.00416667;
	    qs[offset] = 0.00833333;
          }
          
#ifdef YASK
          lam_grid->writeElem(1./(dd*(vp*vp - 2.*vs*vs)), i+bdry_width, j+bdry_width, k+bdry_width, 0);
          mu_grid->writeElem(1./(dd*vs*vs), i+bdry_width, j+bdry_width, k+bdry_width, 0);
          density_grid->writeElem(dd, i+bdry_width, j+bdry_width, k+bdry_width, 0);
#else
          lam[offset]=1./(dd*(vp*vp - 2.*vs*vs)); 
          mu[offset]=1./(dd*vs*vs); 
          d1[offset]=dd;           
#endif
        }
  }
  else
  {
    Grid3D tmpvp=NULL, tmpvs=NULL, tmpdd=NULL;
    Grid3D tmppq=NULL, tmpsq=NULL;
    int var_offset;

    stime = w_time();
    printf("inimesh: Allocating....\n");
    tmpvp = Alloc3D(nxt, nyt, nzt);
    tmpvs = Alloc3D(nxt, nyt, nzt);
    tmpdd = Alloc3D(nxt, nyt, nzt);
    printf("inimesh: done allocating, took %lf\n", w_time()-stime);

    stime = w_time();
    printf("inimesh: Initializing....\n");
#pragma omp parallel for collapse(3)
    for(int_pt i=0;i<nxt;i++)
    {
      for(int_pt j=0;j<nyt;j++)
      {
        for(int_pt k=0;k<nzt;k++)
        {
          tmpvp[i][j][k]=0.0f;
          tmpvs[i][j][k]=0.0f;
          tmpdd[i][j][k]=0.0f;
        }
      }
    }
    printf("inimesh: done initializing, took %lf\n", w_time()-stime);
        
        
    if(NVE==1)
    {
      tmppq = Alloc3D(nxt, nyt, nzt);
      tmpsq = Alloc3D(nxt, nyt, nzt);
      stime = w_time();
      printf("inimesh: Initializing 2....\n");
            
#pragma omp parallel for collapse(3)
      for(int_pt i=0;i<nxt;i++)
      {
        for(int_pt j=0;j<nyt;j++)
        {
          for(int_pt k=0;k<nzt;k++)
          {
            tmppq[i][j][k]=0.0f;
            tmpsq[i][j][k]=0.0f;
          }
        }
      }
      printf("inimesh: done initializing 2, took %lf\n", w_time()-stime);
            
    }
        
    if(nvar==8)
    {
      var_offset=3;
    }
    else if(nvar==5)
    {
      var_offset=0;
    }
    else
    {
      var_offset=0;
    }
        
    if(MEDIASTART>=1 && MEDIASTART<=3)
    {
      stime = w_time();
      printf("inimesh: Starting comp 1....\n");

#pragma omp parallel for collapse(3)
      for(int_pt k=0;k<nzt;k++)
      {
        for(int_pt j=0;j<nyt;j++)
        {
          for(int_pt i=0;i<nxt;i++)
          {
            int_pt l_readOffset = (k*i_inputSizeY*i_inputSizeX+j*i_inputSizeX+i)*nvar;
            tmpvp[i][j][k]=i_inputBuffer[l_readOffset+var_offset];
            tmpvs[i][j][k]=i_inputBuffer[l_readOffset+var_offset+1];
            tmpdd[i][j][k]=i_inputBuffer[l_readOffset+var_offset+2];
                        
            if(nvar>3)
            {
              tmppq[i][j][k]=i_inputBuffer[l_readOffset+var_offset+3];
              tmpsq[i][j][k]=i_inputBuffer[l_readOffset+var_offset+4];
            }
            if(tmpvp[i][j][k]!=tmpvp[i][j][k] ||
               tmpvs[i][j][k]!=tmpvs[i][j][k] ||
               tmpdd[i][j][k]!=tmpdd[i][j][k]){
              printf("tmpvp,vs,dd is NAN!\n");
              MPI_Abort(MPI_COMM_WORLD,1);
            }
          }
        }
      }
            
    }

    float w0=0.0f, ww1=0.0f, w2=0.0f, tmp1=0.0f, tmp2=0.0f;
    if(NVE==1)
    {
      w0=2*pi*FP;
      printf("w0 = %g\n",w0);
    }
        
    float facex = (float)pow(FAC,EX);



        
#pragma omp parallel for collapse(2)
    for(int_pt i=0;i<nxt;i++)
    {
      for(int_pt j=0;j<nyt;j++)
      {
        float weights_los[2][2][2];
        float weights_lop[2][2][2];
        float val[2];
        float mu1, denom;
        float qpinv=0.0f, qsinv=0.0f, vpvs=0.0f;
        int ii,jj,kk,iii,jjj,kkk,num;
        std::complex<double> value(0.0, 0.0);
        std::complex<double> sqrtm1(0.0, 1.0);

        for(int_pt k=0;k<nzt;k++)
        {
          if(tmpvs[i][j][k]<200.0)
          {
            tmpvs[i][j][k]=200.0;
            tmpvp[i][j][k]=600.0;
          }
          tmpsq[i][j][k] = 0.1  * tmpvs[i][j][k];
          tmppq[i][j][k] = 2.0   * tmpsq[i][j][k];
                    
          if(tmppq[i][j][k]>200.0)
          {
            // QF - VP
            val[0] = 0.0;
            val[1] = 0.0;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  denom = ((w0*w0*tau[ii][jj][kk]*tau[ii][jj][kk]+1.0)*tmppq[i][j][k]*facex);
                  val[0] += weights[ii][jj][kk]/denom;
                  val[1] += -weights[ii][jj][kk]*w0*tau[ii][jj][kk]/denom;
                }
            mu1 = tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k]/(1.0-val[0]);
          }
          else
          {
            num=0;
            for (iii=0;iii<2;iii++)
              for(jjj=0;jjj<2;jjj++)
                for(kkk=0;kkk<2;kkk++)
                {
                  weights_lop[iii][jjj][kkk]=coeff[num]/(tmppq[i][j][k]*tmppq[i][j][k])+coeff[num+1]/(tmppq[i][j][k]);
                  num=num+2;
                }
                        
            value=0.0+0.0*sqrtm1;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  value=value+1.0/(1.0-((double)weights_lop[ii][jj][kk])/(1.0+sqrtm1*((double)w0*tau[ii][jj][kk])));
                }
            value=1./value;
                        
            mu1=tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k]/(8.*std::real(value));
          }
                    
                    
          tmpvp[i][j][k] = sqrt(mu1/tmpdd[i][j][k]);
                    
                    
                    
                    
          // QF - VS
          if(tmpsq[i][j][k]>200.0)
          {
            val[0] = 0.0;
            val[1] = 0.0;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  denom = ((w0*w0*tau[ii][jj][kk]*tau[ii][jj][kk]+1.0)*tmpsq[i][j][k]*facex);
                  val[0] += weights[ii][jj][kk]/denom;
                  val[1] += -weights[ii][jj][kk]*w0*tau[ii][jj][kk]/denom;
                }
            mu1 = tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]/(1.0-val[0]);
          }
          else
          {
            num=0;
            for (iii=0;iii<2;iii++)
              for(jjj=0;jjj<2;jjj++)
                for(kkk=0;kkk<2;kkk++)
                {
                  weights_los[iii][jjj][kkk]=coeff[num]/(tmpsq[i][j][k]*tmpsq[i][j][k])+coeff[num+1]/(tmpsq[i][j][k]);
                  num=num+2;
                }
            value=0.0+0.0*sqrtm1;
            for(ii=0;ii<2;ii++)
              for(jj=0;jj<2;jj++)
                for(kk=0;kk<2;kk++)
                {
                  value=value+1.0/(1.0-((double)weights_los[ii][jj][kk])/(1.0+sqrtm1*((double)w0*tau[ii][jj][kk])));
                }
            value=1./value;
            mu1=tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]/(8.*std::real(value));
          }
                    
          tmpvs[i][j][k] = sqrt(mu1/tmpdd[i][j][k]);
                    
                    
          // QF - end
          if (SoCalQ==1)
          {
            vpvs=tmpvp[i][j][k]/tmpvs[i][j][k];
            if (vpvs<1.45)  tmpvs[i][j][k]=tmpvp[i][j][k]/1.45;
          }
          if(tmpvp[i][j][k]>7600.0)
          {
            tmpvs[i][j][k]=4387.0;
            tmpvp[i][j][k]=7600.0;
          }
          if(tmpvs[i][j][k]<200.0)
          {
            tmpvs[i][j][k]=200.0;
            tmpvp[i][j][k]=600.0;
          }

          int_pt offset = i * i_strideX + j * i_strideY + (nzt - 1 - k) * i_strideZ; 
                    
          if(tmpdd[i][j][k]<1700.0) tmpdd[i][j][k]=1700.0;
          
#ifdef YASK
          mu_grid->writeElem(1./(tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]), i+bdry_width, j+bdry_width, nzt-1-k+bdry_width,0);
          lam_grid->writeElem(1./(tmpdd[i][j][k]*(tmpvp[i][j][k]*tmpvp[i][j][k]
                              -2.*tmpvs[i][j][k]*tmpvs[i][j][k]))
                              , i+bdry_width, j+bdry_width, nzt-1-k+bdry_width,0);
          density_grid->writeElem(tmpdd[i][j][k], i+bdry_width, j+bdry_width, nzt-1-k+bdry_width,0);
#else
          mu[offset]  = 1./(tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]);
          lam[offset] = 1./(tmpdd[i][j][k]*(tmpvp[i][j][k]*tmpvp[i][j][k]
                                            -2.*tmpvs[i][j][k]*tmpvs[i][j][k]));
          d1[offset]  = tmpdd[i][j][k];          
#endif          
          
          if(NVE==1)
          {
            if(tmppq[i][j][k]<=0.0)
            {
              qpinv=0.0;
              qsinv=0.0;
            }
            else
            {
              qpinv=1./tmppq[i][j][k];
              qsinv=1./tmpsq[i][j][k];
            }
            tmppq[i][j][k] = qpinv/facex;
            tmpsq[i][j][k] = qsinv/facex;
            qp[offset] = tmppq[i][j][k]; 
            qs[offset] = tmpsq[i][j][k]; 
          }
        }
      }
    }

    // initialize these to large values, we will be setting them to max / min of various arrays
    double max_vse = -1.0e10;
    double max_vpe = -1.0e10;
    double max_dde = -1.0e10;
    double min_vse = 1.0e10;
    double min_vpe = 1.0e10;
    double min_dde = 1.0e10;


        
#pragma omp parallel for reduction(max: max_vpe, max_vse, max_dde), reduction(min: min_vpe, min_vse, min_dde) collapse(2)
    for(int_pt i=0;i<nxt;i++)
    {
      for(int_pt j=0;j<nyt;j++)
      {
        for(int_pt k=0; k<nzt; k++)
        {
          if(tmpvs[i][j][k]<min_vse) min_vse = tmpvs[i][j][k];
          if(tmpvs[i][j][k]>max_vse) max_vse = tmpvs[i][j][k];
          if(tmpvp[i][j][k]<min_vpe) min_vpe = tmpvp[i][j][k];
          if(tmpvp[i][j][k]>max_vpe) max_vpe = tmpvp[i][j][k];
          if(tmpdd[i][j][k]<min_dde) min_dde = tmpdd[i][j][k];
          if(tmpdd[i][j][k]>max_dde) max_dde = tmpdd[i][j][k];                    
        }
      }
    }

    m_vse[0] = min_vse;
    m_vse[1] = max_vse;
    m_vpe[0] = min_vpe;
    m_vpe[1] = max_vpe;
    m_dde[0] = min_dde;
    m_dde[1] = max_dde;

    // TODO(Mpi): these values above need to be reduced over MPI nodes too (not here though)
        
    Delloc3D(tmpvp);
    Delloc3D(tmpvs);
    Delloc3D(tmpdd);
    if(NVE==1)
    {
      Delloc3D(tmppq);
      Delloc3D(tmpsq);
    }
  }
}

/**
 
   @param taumin  -
   @param taumax  -
 
   @param[out] tau     -
*/
void odc::data::Mesh::tausub( Grid3D tau, float taumin,float taumax)
{
  int idx, idy, idz;
  float tautem[2][2][2];
  float tmp;
    
  //(gwilkins) Why use this access pattern?
  tautem[0][0][0]=1.0;
  tautem[1][0][0]=6.0;
  tautem[0][1][0]=7.0;
  tautem[1][1][0]=4.0;
  tautem[0][0][1]=8.0;
  tautem[1][0][1]=3.0;
  tautem[0][1][1]=2.0;
  tautem[1][1][1]=5.0;
    
  for(idx=0;idx<2;idx++)
    for(idy=0;idy<2;idy++)
      for(idz=0;idz<2;idz++)
      {
        tmp = tautem[idx][idy][idz];
        tmp = (tmp-0.5)/8.0;
        tmp = 2.0*tmp - 1.0;
                
        tau[idx][idy][idz] = exp(0.5*(log(taumax*taumin) + log(taumax/taumin)*tmp));
      }
    
  return;
} // end function tausub





/**
 
   @param nxt         -
   @param nyt         -
   @param nzt         -
   @param tau1        -
   @param tau2        -
   @param vx1         -
   @param vx2         -
   @param weights     -
   @param ww          -
   @param wwo         -
   @param xls         -
   @param xre         -
   @param yls         -
   @param yre         -
*/
void odc::data::Mesh::init_texture(int nxt,  int nyt,  int nzt,  Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2,
                                   Grid3D weights, Grid3Dww ww,Grid3D wwo,
                                   int xls,  int xre,  int yls,  int yre)
{
  int i, j, k, itx, ity, itz;
  itx = 0;
  ity = 0;
  itz = (nzt-1)%2;
  for(i=xls;i<=xre;i++)
  {
    itx = 1 - itx;
    for(j=yls;j<=yre;j++)
    {
      ity = 1 - ity;
      for(k=0;k<nzt;k++)
      {
        itz           = 1 - itz;
        vx1[i][j][k]  = tau1[itx][ity][itz];
        vx2[i][j][k]  = tau2[itx][ity][itz];
        wwo[i][j][k]   = 8.0*weights[itx][ity][itz];
        if (itx<0.5 && ity<0.5 && itz<0.5) ww[i][j][k]   = 1;
        else if(itx<0.5 && ity<0.5 && itz>0.5) ww[i][j][k]   = 2;
        else if(itx<0.5 && ity>0.5 && itz<0.5) ww[i][j][k]   = 3;
        else if(itx<0.5 && ity>0.5 && itz>0.5) ww[i][j][k]   = 4;
        else if(itx>0.5 && ity<0.5 && itz<0.5) ww[i][j][k]   = 5;
        else if(itx>0.5 && ity<0.5 && itz>0.5) ww[i][j][k]   = 6;
        else if(itx>0.5 && ity>0.5 && itz<0.5) ww[i][j][k]   = 7;
        else if(itx>0.5 && ity>0.5 && itz>0.5) ww[i][j][k]   = 8;
        //		 printf("%g %g\n",ww[i][j][k],ww[i][j][k]);
                
      }
    }
  }
  return;
}

// TODO(Josh): clean this
void odc::data::Mesh::new_init_texture(Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2, Grid3D weights, Grid3Dww ww,Grid3D wwo,
                                       int_pt startX,  int_pt endX,  int_pt startY,  int_pt endY, int_pt startZ, int_pt endZ,
                                       int_pt globalStartX, int_pt globalStartY, int_pt globalStartZ, int_pt sizeZ)
{
  int i, j, k, itx, ity, itz;

  itx = (abs(globalStartX + startX)) % 2;
    
  ity = (abs(globalStartY + startY)) % 2;

  itz = (abs(globalStartZ + startZ + sizeZ - 1)) % 2; 

  for(i=startX;i<endX;i++)
  {
    itx = 1 - itx;
    for(j=startY;j<endY;j++)
    {
      ity = 1 - ity;
      for(k=startZ;k<endZ;k++)
      {
        itz           = 1 - itz;
        vx1[i][j][k]  = tau1[itx][ity][itz];
        vx2[i][j][k]  = tau2[itx][ity][itz];
        wwo[i][j][k]   = 8.0*weights[itx][ity][itz];
        if (itx<0.5 && ity<0.5 && itz<0.5) ww[i][j][k]   = 1;
        else if(itx<0.5 && ity<0.5 && itz>0.5) ww[i][j][k]   = 2;
        else if(itx<0.5 && ity>0.5 && itz<0.5) ww[i][j][k]   = 3;
        else if(itx<0.5 && ity>0.5 && itz>0.5) ww[i][j][k]   = 4;
        else if(itx>0.5 && ity<0.5 && itz<0.5) ww[i][j][k]   = 5;
        else if(itx>0.5 && ity<0.5 && itz>0.5) ww[i][j][k]   = 6;
        else if(itx>0.5 && ity>0.5 && itz<0.5) ww[i][j][k]   = 7;
        else if(itx>0.5 && ity>0.5 && itz>0.5) ww[i][j][k]   = 8;
        //		 printf("%g %g\n",ww[i][j][k],ww[i][j][k]);
                
      }
    }
  }
  return;
}





/**
 
   @param weights
   @param coeff
   @param ex
   @param fac
 
 
*/
void odc::data::Mesh::weights_sub(Grid3D weights,Grid1D coeff, float ex, float fac)
{
    
  int i,j,k;
    
  if(ex<0.15 && ex>=0.01)
  {
        
    weights[0][0][0] =0.3273;
    weights[1][0][0] =1.0434;
    weights[0][1][0] =0.044;
    weights[1][1][0] =0.9393;
    weights[0][0][1] =1.7268;
    weights[1][0][1] =0.369;
    weights[0][1][1] =0.8478;
    weights[1][1][1] =0.4474;
        
    coeff[0] = 7.3781;
    coeff[1]= 4.1655;
    coeff[2]= -83.1627;
    coeff[3]=13.1326;
    coeff[4]=69.0839;
    coeff[5]=0.4981;
    coeff[6]= -37.6966;
    coeff[7]=5.5263;
    coeff[8]=-51.4056;
    coeff[9]=8.1934;
    coeff[10]=13.1865;
    coeff[11]=3.4775;
    coeff[12]=-36.1049;
    coeff[13]=7.2107;
    coeff[14]=12.3809;
    coeff[15]=3.6117;
        
  }
  else if(ex<0.25 && ex>=0.15)
  {
        
    weights[0][0][0] =0.001;
    weights[1][0][0] =1.0349;
    weights[0][1][0] =0.0497;
    weights[1][1][0] =1.0407;
    weights[0][0][1] =1.7245;
    weights[1][0][1] =0.2005;
    weights[0][1][1] =0.804;
    weights[1][1][1] =0.4452;
        
    coeff[0] = 31.8902;
    coeff[1]= 1.6126;
    coeff[2]= -83.2611;
    coeff[3]=13.0749;
    coeff[4]=65.485;
    coeff[5]=0.5118;
    coeff[6]= -42.02;
    coeff[7]=5.0875;
    coeff[8]=-49.2656;
    coeff[9]=8.1552;
    coeff[10]=25.7345;
    coeff[11]=2.2801;
    coeff[12]=-40.8942;
    coeff[13]=7.9311;
    coeff[14]=7.0206;
    coeff[15]=3.4692;
        
        
  }
  else if(ex<0.35 && ex>=0.25)
  {
    weights[0][0][0] =0.001;
    weights[1][0][0] =1.0135;
    weights[0][1][0] =0.0621;
    weights[1][1][0] =1.1003;
    weights[0][0][1] =1.7198;
    weights[1][0][1] =0.0918;
    weights[0][1][1] =0.6143;
    weights[1][1][1] =0.4659;
        
    coeff[0] = 43.775;
    coeff[1]= -0.1091;
    coeff[2]= -83.1088;
    coeff[3]=13.0161;
    coeff[4]=60.9008;
    coeff[5]=0.592;
    coeff[6]= -43.4857;
    coeff[7]=4.5869;
    coeff[8]=-45.3315;
    coeff[9]=8.0252;
    coeff[10]=34.3571;
    coeff[11]=1.199;
    coeff[12]=-41.4422;
    coeff[13]=8.399;
    coeff[14]=-2.8772;
    coeff[15]=3.5323;
        
  }
  else if(ex<0.45 && ex>=0.35)
  {
    weights[0][0][0] =0.001;
    weights[1][0][0] =0.9782;
    weights[0][1][0] =0.082;
    weights[1][1][0] =1.1275;
    weights[0][0][1] =1.7122;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.4639;
    weights[1][1][1] =0.509;
        
    coeff[0] = 41.6858;
    coeff[1]= -0.7344;
    coeff[2]= -164.2252;
    coeff[3]=14.9961;
    coeff[4]=103.0301;
    coeff[5]=-0.4199;
    coeff[6]= -41.1157;
    coeff[7]=3.8266;
    coeff[8]=-73.0432;
    coeff[9]=8.5857;
    coeff[10]=38.0868;
    coeff[11]=0.3937;
    coeff[12]=-43.2133;
    coeff[13]=8.6747;
    coeff[14]=5.6362;
    coeff[15]=3.3287;
  }
  else if(ex<0.55 && ex>=0.45)
  {
    weights[0][0][0] =0.2073;
    weights[1][0][0] =0.912;
    weights[0][1][0] =0.1186;
    weights[1][1][0] =1.081;
    weights[0][0][1] =1.6984;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.1872;
    weights[1][1][1] =0.6016;
        
    coeff[0] = 20.0539;
    coeff[1]= -0.4354;
    coeff[2]= -81.6068;
    coeff[3]=12.8573;
    coeff[4]=45.9948;
    coeff[5]=1.1528;
    coeff[6]= -23.07;
    coeff[7]=2.6719;
    coeff[8]=-27.8961;
    coeff[9]=7.1927;
    coeff[10]=31.4788;
    coeff[11]=-0.0434;
    coeff[12]=-25.1661;
    coeff[13]=8.245;
    coeff[14]=-45.2178;
    coeff[15]=4.8476;
        
  }
  else if(ex<0.65 && ex>=0.55)
  {    
    weights[0][0][0] = 0.3112;
    weights[1][0][0] = 0.8339;
    weights[0][1][0] = 0.1616;
    weights[1][1][0] = 1.0117;
    weights[0][0][1] = 1.6821;
    weights[1][0][1] = 0.0001;
    weights[0][1][1] = 0.0001;
    weights[1][1][1] = 0.7123;
        
    coeff[0] = 8.0848;
    coeff[1]= -0.1968;
    coeff[2]= -79.9715;
    coeff[3]=12.7318;
    coeff[4]=35.7155;
    coeff[5]=1.68;
    coeff[6]= -13.0365;
    coeff[7]=1.8101;
    coeff[8]=-13.2235;
    coeff[9]=6.3697;
    coeff[10]=25.4548;
    coeff[11]=-0.3947;
    coeff[12]=-10.4478;
    coeff[13]=7.657;
    coeff[14]=-75.9179;
    coeff[15]=6.1791;
  }
  else if(ex<0.75 && ex>=0.65)
  {
    weights[0][0][0] =0.1219;
    weights[1][0][0] =0.001;
    weights[0][1][0] =0.5084;
    weights[1][1][0] =0.2999;
    weights[0][0][1] =1.2197;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.001;
    weights[1][1][1] =1.3635;
        
    coeff[0] = 1.9975;
    coeff[1]= 0.418;
    coeff[2]= -76.6932;
    coeff[3]=11.3479;
    coeff[4]=40.7406;
    coeff[5]=1.9511;
    coeff[6]= -2.7761;
    coeff[7]=0.5987;
    coeff[8]=0;
    coeff[9]=0;
    coeff[10]=0;
    coeff[11]=0;
    coeff[12]=41.317;
    coeff[13]=2.1874;
    coeff[14]=-88.8095;
    coeff[15]=11.0003;
  }
  else if(ex<0.85 && ex>=0.75)
  {
    weights[0][0][0] = 0.0462 ;
    weights[1][0][0] = 0.001;
    weights[0][1][0] = 0.4157;
    weights[1][1][0] = 0.1585;
    weights[0][0][1] = 1.3005;
    weights[1][0][1] = 0.001;
    weights[0][1][1] = 0.001;
    weights[1][1][1] = 1.4986;
        
    coeff[0] = 5.1672;
    coeff[1]= 0.2129;
    coeff[2]= -46.506;
    coeff[3]=11.7213;
    coeff[4]=-5.8873;
    coeff[5]=1.4279;
    coeff[6]= -8.2448;
    coeff[7]=0.3455;
    coeff[8]=15.0254;
    coeff[9]=-0.283;
    coeff[10]=0;
    coeff[11]=0;
    coeff[12]=58.975;
    coeff[13]=0.8131;
    coeff[14]=-108.6828;
    coeff[15]=12.4362;
        
  }
  else if(ex<0.95 && ex>=0.85)
  {
    weights[0][0][0] =0.001;
    weights[1][0][0] =0.001;
    weights[0][1][0] =0.1342;
    weights[1][1][0] =0.1935;
    weights[0][0][1] =1.5755;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.001;
    weights[1][1][1] =1.5297;
        
    coeff[0] = -0.8151;
    coeff[1]= 0.1621;
    coeff[2]= -61.9333;
    coeff[3]=12.5014;
    coeff[4]=0.0358;
    coeff[5]=-0.0006;
    coeff[6]= 0;
    coeff[7]=0;
    coeff[8]=22.0291;
    coeff[9]=-0.4022;
    coeff[10]=0;
    coeff[11]=0;
    coeff[12]=56.0043;
    coeff[13]=0.7978;
    coeff[14]=-116.9175;
    coeff[15]=13.0244;
  }
  else if(ex<0.01)
  {
    weights[0][0][0] = 0.8867;
    weights[1][0][0] = 1.0440  ;
    weights[0][1][0] =0.0423  ;
    weights[1][1][0] =0.8110 ;
    weights[0][0][1] =1.7275   ;
    weights[1][0][1] =0.5615 ;
    weights[0][1][1] =0.8323 ;
    weights[1][1][1] =0.4641 ;
        
        
    coeff[0] = -27.5089;
    coeff[1]= 7.4177;
    coeff[2]=-82.8803;
    coeff[3]=13.1952;
    coeff[4]=72.0312;
    coeff[5]=0.5298;
    coeff[6]=-34.1779;
    coeff[7]=6.0293;
    coeff[8]=-52.2607;
    coeff[9]=8.1754;
    coeff[10]=-1.6270;
    coeff[11]=4.6858;
    coeff[12]=-27.7770;
    coeff[13]=6.2852;
    coeff[14]=14.6295;
    coeff[15]=3.8839;
  }

  // The following is done in inimesh in the CPU code.
  double facex = pow(fac,ex);
  for(i=0;i<2;i++)
    for(j=0;j<2;j++)
      for(k=0;k<2;k++)
        weights[i][j][k] = weights[i][j][k]/facex;
    
  return;
} // end function weights_sub


// TODO(Josh): rewrite this in a sane way
void odc::data::Mesh::set_boundaries(
#ifdef YASK
    Grid_XYZ* gd1,Grid_XYZ* gmu,Grid_XYZ* glam,
#endif
//#else
    Grid3D d1, Grid3D mu, Grid3D lam,
//#endif
    Grid3D qp, Grid3D qs, bool anelastic,
    int_pt bdry_width, int_pt nx, int_pt ny, int_pt nz)

{
  int_pt h = bdry_width;
  for(int_pt j=0;j<ny;j++)
  {
    for(int_pt k=0;k<nz;k++)
    {
#ifdef YASK
      glam->writeElem(glam->readElem(h,h+j,h+k,0), h-1,h+j,h+k,0);
      glam->writeElem(glam->readElem(h+nx-1,h+j,h+k,0), h+nx,h+j,h+k,0);
      
      gmu->writeElem(gmu->readElem(h,h+j,h+k,0), h-1,h+j,h+k,0);
      gmu->writeElem(gmu->readElem(h+nx-1,h+j,h+k,0), h+nx,h+j,h+k,0);

      gd1->writeElem(gd1->readElem(h,h+j,h+k,0),h-1,h+j,h+k,0);
      gd1->writeElem(gd1->readElem(h+nx-1,h+j,h+k,0),h+nx,h+j,h+k,0);
#else
      lam[h-1][h+j][h+k]     = lam[h][h+j][h+k];
      lam[h+nx][h+j][h+k]    = lam[h+nx-1][h+j][h+k];
      mu[h-1][h+j][h+k]      = mu[h][h+j][h+k];
      mu[h+nx][h+j][h+k]     = mu[h+nx-1][h+j][h+k];
      d1[h-1][h+j][h+k]      = d1[h][h+j][h+k];
      d1[h+nx][h+j][h+k]     = d1[h+nx-1][h+j][h+k];
#endif
    }
  }   

  for(int_pt i=0;i<nx;i++)
  {
    for(int_pt k=0;k<nz;k++)
    {
#ifdef YASK
      glam->writeElem(glam->readElem(h+i,h,h+k,0), h+i,h-1,h+k,0);
      glam->writeElem(glam->readElem(h+i-1,h+ny-1,h+k,0), h+i,h+ny,h+k,0);
      
      gmu->writeElem(gmu->readElem(h+i,h,h+k,0), h+i,h-1,h+k,0);
      gmu->writeElem(gmu->readElem(h+i,h+ny-1,h+k,0), h+i,h+ny,h+k,0);

      gd1->writeElem(gd1->readElem(h+i,h,h+k,0),h+i,h-1,h+k,0);
      gd1->writeElem(gd1->readElem(h+i,h+ny-1,h+k,0),h+i,h+ny,h+k,0);
#else
      lam[h+i][h-1][h+k]     = lam[h+i][h][h+k];
      lam[h+i][h+ny][h+k]    = lam[h+i][h+ny-1][h+k];
      mu[h+i][h-1][h+k]      = mu[h+i][h][h+k];
      mu[h+i][h+ny][h+k]     = mu[h+i][h+ny-1][h+k];
      d1[h+i][h-1][h+k]      = d1[h+i][h][h+k];
      d1[h+i][h+ny][h+k]     = d1[h+i][h+ny-1][h+k];
#endif
    }
  }

  for(int_pt i=0;i<nx;i++)
  {
    for(int_pt j=0;j<ny;j++)
    {
#ifdef YASK
      glam->writeElem(glam->readElem(h+i,h+j,h,0), h+i,h+j,h-1,0);
      gmu->writeElem(gmu->readElem(h+i,h+j,h,0), h+i,h+j,h-1,0);
      gd1->writeElem(gd1->readElem(h+i,h+j,h,0),h+i,h+j,h-1,0);
#else      
      lam[h+i][h+j][h-1]   = lam[h+i][h+j][h];
      mu[h+i][h+j][h-1]    = mu[h+i][h+j][h];
      d1[h+i][h+j][h-1]    = d1[h+i][h+j][h];
#endif
    }
  }
  
  //12 border lines
  for(int_pt i=0;i<nx;i++)
  {
#ifdef YASK
    glam->writeElem(glam->readElem(h+i,h,h,0), h+i,h-1,h-1,0);
    gmu->writeElem(gmu->readElem(h+i,h,h,0), h+i,h-1,h-1,0);
    gd1->writeElem(gd1->readElem(h+i,h,h,0),h+i,h-1,h-1,0);
    glam->writeElem(glam->readElem(h+i,h+ny-1,h,0), h+i,h+ny,h-1,0);
    gmu->writeElem(gmu->readElem(h+i,h+ny-1,h,0), h+i,h+ny,h-1,0);
    gd1->writeElem(gd1->readElem(h+i,h+ny-1,h,0),h+i,h+ny,h-1,0);
    glam->writeElem(glam->readElem(h+i,h,h+nz-1,0), h+i,h-1,h+nz,0);
    gmu->writeElem(gmu->readElem(h+i,h,h+nz-1,0), h+i,h-1,h+nz,0);
    gd1->writeElem(gd1->readElem(h+i,h,h+nz-1,0),h+i,h-1,h+nz,0);
    glam->writeElem(glam->readElem(h+i,h+ny-1,h+nz-1,0), h+i,h+ny,h+nz,0);
    gmu->writeElem(gmu->readElem(h+i,h+ny-1,h+nz-1,0),h+i,h+ny,h+nz,0);
    gd1->writeElem(gd1->readElem(h+i,h+ny-1,h+nz-1,0),h+i,h+ny,h+nz,0);
#else
    
    lam[h+i][h-1][h-1]          = lam[h+i][h][h];
    mu[h+i][h-1][h-1]           = mu[h+i][h][h];
    d1[h+i][h-1][h-1]           = d1[h+i][h][h];
    lam[h+i][h+ny][h-1]         = lam[h+i][h+ny-1][h];
    mu[h+i][h+ny][h-1]          = mu[h+i][h+ny-1][h];
    d1[h+i][h+ny][h-1]          = d1[h+i][h+ny-1][h];
    lam[h+i][h-1][h+nz]         = lam[h+i][h][h+nz-1];
    mu[h+i][h-1][h+nz]          = mu[h+i][h][h+nz-1];
    d1[h+i][h-1][h+nz]          = d1[h+i][h][h+nz-1];
    lam[h+i][h+ny][h+nz]        = lam[h+i][h+ny-1][h+nz-1];
    mu[h+i][h+ny][h+nz]         = mu[h+i][h+ny-1][h+nz-1];
    d1[h+i][h+ny][h+nz]         = d1[h+i][h+ny-1][h+nz-1];
#endif
  } 
        
  for(int_pt j=0;j<ny;j++)
  {
#ifdef YASK
    glam->writeElem(glam->readElem(h,h+j,h,0), h-1,h+j,h-1,0);
    gmu->writeElem(gmu->readElem(h,h+j,h,0), h-1,h+j,h-1,0);
    gd1->writeElem(gd1->readElem(h,h+j,h,0), h-1,h+j,h-1,0);
    glam->writeElem(glam->readElem(h+nx-1,h+j,h,0), h+nx,h+j,h-1,0);
    gmu->writeElem(gmu->readElem(h+nx-1,h+j,h,0), h+nx,h+j,h-1,0);
    gd1->writeElem(gd1->readElem(h+nx-1,h+j,h,0), h+nx,h+j,h-1,0);
    glam->writeElem(glam->readElem(h,h+j,h+nz-1,0), h-1,h+j,h+nz,0);
    gmu->writeElem(gmu->readElem(h,h+j,h+nz-1,0), h-1,h+j,h+nz,0);
    gd1->writeElem(gd1->readElem(h,h+j,h+nz-1,0), h-1,h+j,h+nz,0);
    glam->writeElem(glam->readElem(h+nx-1,h+j,h+nz-1,0), h+nx,h+j,h+nz,0);
    gmu->writeElem(gmu->readElem(h+nx-1,h+j,h+nz-1,0), h+nx,h+j,h+nz,0);
    gd1->writeElem(gd1->readElem(h+nx-1,h+j,h+nz-1,0), h+nx,h+j,h+nz,0);
#else          
    lam[h-1][h+j][h-1]          = lam[h][h+j][h];
    mu[h-1][h+j][h-1]           = mu[h][h+j][h];
    d1[h-1][h+j][h-1]           = d1[h][h+j][h];
    lam[h+nx][h+j][h-1]         = lam[h+nx-1][h+j][h];
    mu[h+nx][h+j][h-1]          = mu[h+nx-1][h+j][h];
    d1[h+nx][h+j][h-1]          = d1[h+nx-1][h+j][h];
    lam[h-1][h+j][h+nz]         = lam[h][h+j][h+nz-1];
    mu[h-1][h+j][h+nz]          = mu[h][h+j][h+nz-1];
    d1[h-1][h+j][h+nz]          = d1[h][h+j][h+nz-1];
    lam[h+nx][h+j][h+nz]        = lam[h+nx-1][h+j][h+nz-1];
    mu[h+nx][h+j][h+nz]         = mu[h+nx-1][h+j][h+nz-1];
    d1[h+nx][h+j][h+nz]         = d1[h+nx-1][h+j][h+nz-1];
#endif
  }
        
  for(int_pt k=0;k<nz;k++)
  {
#ifdef YASK
    glam->writeElem(glam->readElem(h,h,h+k,0), h-1,h-1,h+k,0);
    gmu->writeElem(gmu->readElem(h,h,h+k,0), h-1,h-1,h+k,0);
    gd1->writeElem(gd1->readElem(h,h,h+k,0), h-1,h-1,h+k,0);
    glam->writeElem(glam->readElem(h+nx-1,h,h+k,0), h+nx,h-1,h+k,0);
    gmu->writeElem(gmu->readElem(h+nx-1,h,h+k,0), h+nx,h-1,h+k,0);
    gd1->writeElem(gd1->readElem(h+nx-1,h,h+k,0), h+nx,h-1,h+k,0);
    glam->writeElem(glam->readElem(h,h+ny-1,h+k,0), h-1,h+ny,h+k,0);
    gmu->writeElem(gmu->readElem(h,h+ny-1,h+k,0), h-1,h+ny,h+k,0);
    gd1->writeElem(gd1->readElem(h,h+ny-1,h+k,0), h-1,h+ny,h+k,0);
    glam->writeElem(glam->readElem(h+nx-1,h+ny-1,h+k,0), h+nx,h+ny,h+k,0);
    gmu->writeElem(gmu->readElem(h+nx-1,h+ny-1,h+k,0), h+nx,h+ny,h+k,0);
    gd1->writeElem(gd1->readElem(h+nx-1,h+ny-1,h+k,0), h+nx,h+ny,h+k,0);
#else      
    
    lam[h-1][h-1][h+k]          = lam[h][h][h+k];
    mu[h-1][h-1][h+k]           = mu[h][h][h+k];
    d1[h-1][h-1][h+k]           = d1[h][h][h+k];
    lam[h+nx][h-1][h+k]         = lam[h+nx-1][h][h+k];
    mu[h+nx][h-1][h+k]          = mu[h+nx-1][h][h+k];
    d1[h+nx][h-1][h+k]          = d1[h+nx-1][h][h+k];
    lam[h-1][h+ny][h+k]         = lam[h][h+ny-1][h+k];
    mu[h-1][h+ny][h+k]          = mu[h][h+ny-1][h+k];
    d1[h-1][h+ny][h+k]          = d1[h][h+ny-1][h+k];
    lam[h+nx][h+ny][h+k]        = lam[h+nx-1][h+ny-1][h+k];
    mu[h+nx][h+ny][h+k]         = mu[h+nx-1][h+ny-1][h+k];
    d1[h+nx][h+ny][h+k]         = d1[h+nx-1][h+ny-1][h+k];
#endif
  }
        
  //8 Corners
#ifdef YASK
  glam->writeElem(glam->readElem(h,h,h ,0), h-1,h-1,h-1,0);
  gmu->writeElem( gmu->readElem( h,h,h ,0), h-1,h-1,h-1,0);
  gd1->writeElem( gd1->readElem( h,h,h ,0), h-1,h-1,h-1,0);

  glam->writeElem(glam->readElem(h+nx-1,h,h,0), h+nx,h-1,h-1,0);
  gmu->writeElem( gmu->readElem( h+nx-1,h,h,0), h+nx,h-1,h-1,0);
  gd1->writeElem( gd1->readElem( h+nx-1,h,h,0), h+nx,h-1,h-1,0);

  glam->writeElem(glam->readElem(h,h+ny-1,h,0), h-1,h+ny,h-1,0);
  gmu->writeElem( gmu->readElem( h,h+ny-1,h,0), h-1,h+ny,h-1,0);
  gd1->writeElem( gd1->readElem( h,h+ny-1,h,0), h-1,h+ny,h-1,0);

  glam->writeElem(glam->readElem(h,h,h+nz-1,0), h-1,h-1,h+nz,0);
  gmu->writeElem( gmu->readElem( h,h,h+nz-1,0), h-1,h-1,h+nz,0);
  gd1->writeElem( gd1->readElem( h,h,h+nz-1,0), h-1,h-1,h+nz,0);

  glam->writeElem(glam->readElem(h+nx-1,h,h+nz-1,0), h+nx,h-1,h+nz,0);
  gmu->writeElem( gmu->readElem( h+nx-1,h,h+nz-1,0), h+nx,h-1,h+nz,0);
  gd1->writeElem( gd1->readElem( h+nx-1,h,h+nz-1,0), h+nx,h-1,h+nz,0);

  glam->writeElem(glam->readElem(h+nx-1,h+ny-1,h,0), h+nx,h+ny,h-1,0);
  gmu->writeElem( gmu->readElem( h+nx-1,h+ny-1,h,0), h+nx,h+ny,h-1,0);
  gd1->writeElem( gd1->readElem( h+nx-1,h+ny-1,h,0), h+nx,h+ny,h-1,0);

  glam->writeElem(glam->readElem(h,h+ny-1,h+nz-1,0), h-1,h+ny,h+nz,0);
  gmu->writeElem( gmu->readElem( h,h+ny-1,h+nz-1,0), h-1,h+ny,h+nz,0);
  gd1->writeElem( gd1->readElem( h,h+ny-1,h+nz-1,0), h-1,h+ny,h+nz,0);

  glam->writeElem(glam->readElem(h+nx-1,h+ny-1,h+nz-1,0), h+nx,h+ny,h+nz,0);
  gmu->writeElem( gmu->readElem( h+nx-1,h+ny-1,h+nz-1,0), h+nx,h+ny,h+nz,0);
  gd1->writeElem( gd1->readElem( h+nx-1,h+ny-1,h+nz-1,0), h+nx,h+ny,h+nz,0);
#else        
  lam[h-1][h-1][h-1]             = lam[h][h][h];
  mu[h-1][h-1][h-1]              = mu[h][h][h];
  d1[h-1][h-1][h-1]              = d1[h][h][h];
  lam[h+nx][h-1][h-1]            = lam[h+nx-1][h][h];
  mu[h+nx][h-1][h-1]             = mu[h+nx-1][h][h]; 
  d1[h+nx][h-1][h-1]             = d1[h+nx-1][h][h];
  lam[h-1][h+ny][h-1]            = lam[h][h+ny-1][h];
  mu[h-1][h+ny][h-1]             = mu[h][h+ny-1][h];
  d1[h-1][h+ny][h-1]             = d1[h][h+ny-1][h];
  lam[h-1][h-1][h+nz]            = lam[h][h][h+nz-1]; 
  mu[h-1][h-1][h+nz]             = mu[h][h][h+nz-1];
  d1[h-1][h-1][h+nz]             = d1[h][h][h+nz-1];
  lam[h+nx][h-1][h+nz]           = lam[h+nx-1][h][h+nz-1];
  mu[h+nx][h-1][h+nz]            = mu[h+nx-1][h][h+nz-1];
  d1[h+nx][h-1][h+nz]            = d1[h+nx-1][h][h+nz-1];
  lam[h+nx][h+ny][h-1]           = lam[h+nx-1][h+ny-1][h];
  mu[h+nx][h+ny][h-1]            = mu[h+nx-1][h+ny-1][h];
  d1[h+nx][h+ny][h-1]            = d1[h+nx-1][h+ny-1][h];
  lam[h-1][h+ny][h+nz]           = lam[h][h+ny-1][h+nz-1];
  mu[h-1][h+ny][h+nz]            = mu[h][h+ny-1][h+nz-1];
  d1[h-1][h+ny][h+nz]            = d1[h][h+ny-1][h+nz-1];
  lam[h+nx][h+ny][h+nz]          = lam[h+nx-1][h+ny-1][h+nz-1];
  mu[h+nx][h+ny][h+nz]           = mu[h+nx-1][h+ny-1][h+nz-1];
  d1[h+nx][h+ny][h+nz]           = d1[h+nx-1][h+ny-1][h+nz-1];
#endif        
        
        
  for(int_pt i=0;i<nx;i++)
  {
    for(int_pt j=0;j<ny;j++)
    {
      int_pt k = nz;
#ifdef YASK
      gd1->writeElem(gd1->readElem(h+i,h+j,h+k-1,0), h+i,h+j,h+k,0);
      gmu->writeElem(gmu->readElem(h+i,h+j,h+k-1,0), h+i,h+j,h+k,0);
      glam->writeElem(glam->readElem(h+i,h+j,h+k-1,0),h+i,h+j,h+k,0);
#else            
      d1[h+i][h+j][h+k]   = d1[h+i][h+j][h+k-1];
      mu[h+i][h+j][h+k]   = mu[h+i][h+j][h+k-1];
      lam[h+i][h+j][h+k]  = lam[h+i][h+j][h+k-1];
#endif
      if(anelastic)
      {
        qp[h+i][h+j][h+k] = qp[h+i][h+j][h+k-1];
        qs[h+i][h+j][h+k] = qs[h+i][h+j][h+k-1];
      }
    }
  }
}
