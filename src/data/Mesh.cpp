/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 @author Rajdeep Konwar (rkonwar AT ucsd.edu)

 @file Mesh.cpp
 @brief Contains functions for reading input mesh data from files and setting up corresponding data structures.

 @section LICENSE

 Copyright (c) 2013-2017, Regents of the University of California
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
*/

#include "Mesh.hpp"
#include "data/common.hpp"
#include "Grid.hpp"

#include <cstdio>
#include <cmath>
#include <complex>

#include <time.h>
#include <sys/time.h>

#ifdef YASK
using namespace yask;
#endif

static real anelastic_coeff( real q, int_pt weight_index, real weight, real *coeff ) {
  if( 1.0 / q <= 200.0 ) {
    q = (coeff[weight_index*2-2] * q * q + coeff[weight_index*2-1] * q) / weight;
  } else {
    q *= 0.5;
  }

  return q;
}

void odc::data::Mesh::initialize( odc::io::OptionParser& i_options,
                                  int_pt x,         int_pt y,         int_pt z,
                                  int_pt bdry_size, bool anelastic,   Grid1D i_inputBuffer,
                                  int_pt i_globalX, int_pt i_globalY, int_pt i_globalZ
#ifdef YASK
                                  , Grid_XYZ* density_grid, Grid_XYZ* mu_grid, Grid_XYZ* lam_grid,
                                  Grid_XYZ* weights_grid, Grid_XYZ* tau2_grid, Grid_XYZ* an_ap_grid,
                                  Grid_XYZ* an_as_grid, Grid_XYZ* an_xy_grid, Grid_XYZ* an_xz_grid,
                                  Grid_XYZ* an_yz_grid
#endif
                                ) {
  real taumax = 0.0, taumin = 0.0;

  Grid3D tau        = Alloc3D( 2, 2, 2 );
  Grid3D tau1       = Alloc3D( 2, 2, 2 );
  Grid3D tau2       = Alloc3D( 2, 2, 2 );
  Grid3D weights    = Alloc3D( 2, 2, 2 );

  int_pt totalX     = x + 2 * bdry_size;
  int_pt totalY     = y + 2 * bdry_size;
  int_pt totalZ     = z + 2 * bdry_size;

  m_usingAnelastic  = anelastic;
#ifndef YASK
  m_density         = odc::data::Alloc3D( totalX, totalY, totalZ, odc::constants::boundary );
  m_mu              = odc::data::Alloc3D( totalX, totalY, totalZ, odc::constants::boundary );
  m_lam             = odc::data::Alloc3D( totalX, totalY, totalZ, odc::constants::boundary );
#endif
  m_lam_mu          = odc::data::Alloc3D( totalX, totalY, 1, odc::constants::boundary, true );

  if( m_usingAnelastic ) {
    m_qp            = odc::data::Alloc3D(   totalX, totalY, totalZ, odc::constants::boundary );
    m_qs            = odc::data::Alloc3D(   totalX, totalY, totalZ, odc::constants::boundary );
    m_tau1          = odc::data::Alloc3D(   totalX, totalY, totalZ, odc::constants::boundary );
    m_tau2          = odc::data::Alloc3D(   totalX, totalY, totalZ, odc::constants::boundary );
    m_weights       = odc::data::Alloc3D(   totalX, totalY, totalZ, odc::constants::boundary );
    m_weight_index  = odc::data::Alloc3Dww( totalX, totalY, totalZ, odc::constants::boundary );
    m_coeff         = Alloc1D( 16 );
    weights_sub( weights, m_coeff, i_options.m_ex, i_options.m_fac );
  }

  inimesh( i_options.m_mediaStart,
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
           (totalY + 2 * odc::constants::boundary) * (totalZ + 2 * odc::constants::boundary),
           totalZ + 2 * odc::constants::boundary,
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
                  m_density, m_mu, m_lam, m_qp, m_qs, m_usingAnelastic, bdry_size, x, y, z );

  //! For free surface boundary condition calculations
  for( int i = bdry_size; i < bdry_size + x; i++ ) {
    for( int j = bdry_size; j < bdry_size + y ;j++ ) {
      real t_xl, t_xl2m;
#ifdef YASK
      t_xl              = 1.0 / lam_grid->readElem( i, j, bdry_size + z - 1, 0 );
      t_xl2m            = 2.0 / mu_grid->readElem( i, j, bdry_size + z - 1, 0 ) + t_xl;
#else
      t_xl              = 1.0 / m_lam[i][j][bdry_size+z-1];
      t_xl2m            = 2.0 / m_mu[i][j][bdry_size+z-1] + t_xl;
#endif
      m_lam_mu[i][j][0] = t_xl / t_xl2m;
    }
  }

  if( anelastic ) {
    // note that tau here will be the same for every patch
    for( int i = 0; i < 2; i++ ) {
      for( int j = 0; j < 2; j++ ) {
        for( int k = 0; k < 2; k++ ) {
          real tauu     = tau[i][j][k];
          tau2[i][j][k] = exp( -i_options.m_dT / tauu );
          tau1[i][j][k] = 0.5 * (1.0 - tau2[i][j][k]);
        }
      }
    }

    init_texture( tau1, tau2, m_tau1, m_tau2, weights, m_weight_index, m_weights,
                  bdry_size, bdry_size + x, bdry_size, bdry_size + y, bdry_size, bdry_size + z,
                  i_globalX, i_globalY, i_globalZ, i_options.m_nZ );
  }

  Delloc3D( tau );
  Delloc3D( tau1 );
  Delloc3D( tau2 );
  Delloc3D( weights );

#ifdef YASK
  for( int_pt tx = -1 + bdry_size; tx < x + 1 + bdry_size; tx++ ) {
    for( int_pt ty = -1 + bdry_size; ty < y + 1 + bdry_size; ty++ ) {
      for( int_pt tz = -1 + bdry_size; tz < z + 1 + bdry_size; tz++ ) {
        real local_qp       = 0.125 * ( m_qp[tx][ty][tz] + m_qp[tx+1][ty][tz] + m_qp[tx][ty-1][tz] + m_qp[tx+1][ty-1][tz] + m_qp[tx][ty][tz-1] +
                                        m_qp[tx+1][ty][tz-1] + m_qp[tx][ty-1][tz-1] + m_qp[tx+1][ty-1][tz-1]);
        real local_qs_diag  = 0.125 * ( m_qs[tx][ty][tz] + m_qs[tx+1][ty][tz] + m_qs[tx][ty-1][tz] + m_qs[tx+1][ty-1][tz] + m_qs[tx][ty][tz-1] +
                                        m_qs[tx+1][ty][tz-1] + m_qs[tx][ty-1][tz-1] + m_qs[tx+1][ty-1][tz-1]);
        real local_qs_xy    = 0.5 * (m_qs[tx][ty][tz] + m_qs[tx][ty][tz-1]);
        real local_qs_xz    = 0.5 * (m_qs[tx][ty][tz] + m_qs[tx][ty-1][tz]);
        real local_qs_yz    = 0.5 * (m_qs[tx][ty][tz] + m_qs[tx+1][ty][tz]);

        an_ap_grid->writeElem( anelastic_coeff( local_qp, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff ), tx, ty, tz, 0 );
        an_as_grid->writeElem( anelastic_coeff( local_qs_diag, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff ), tx, ty, tz, 0 );
        an_xy_grid->writeElem( anelastic_coeff( local_qs_xy, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff ), tx, ty, tz, 0 );
        an_xz_grid->writeElem( anelastic_coeff( local_qs_xz, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff ), tx, ty, tz, 0 );
        an_yz_grid->writeElem( anelastic_coeff( local_qs_yz, m_weight_index[tx][ty][tz], m_weights[tx][ty][tz], m_coeff ), tx, ty, tz, 0 );

        weights_grid->writeElem( m_weights[tx][ty][tz], tx, ty, tz, 0 );
        tau2_grid->writeElem( m_tau2[tx][ty][tz], tx, ty, tz, 0 );
      }
    }
  }

  if( m_usingAnelastic ) {
    odc::data::Delloc3D( m_qp, 2 );
    odc::data::Delloc3D( m_qs, 2 );
    odc::data::Delloc3D( m_tau1, 2 );
    odc::data::Delloc3D( m_tau2, 2 );
    odc::data::Delloc3D( m_weights, 2 );
    odc::data::Delloc3Dww( m_weight_index, 2 );
  }
#endif
}

void odc::data::Mesh::inimesh( int       MEDIASTART,
#ifdef YASK
                               Grid_XYZ* density_grid,
                               Grid_XYZ* mu_grid,
                               Grid_XYZ* lam_grid,
                               int_pt    bdry_width,
#else
                               real*     d1,
                               real*     mu,
                               real*     lam,
#endif
                               real*     qp,
                               real*     qs,
                               int_pt    i_strideX,
                               int_pt    i_strideY,
                               int_pt    i_strideZ,
                               float*    taumax,
                               float*    taumin,
                               Grid3D    tau,
                               Grid3D    weights,
                               Grid1D    coeff,
                               int       nvar,
                               float     FP,
                               float     FAC,
                               float     Q0,
                               float     EX,
                               int       nxt,
                               int       nyt,
                               int       nzt,
                               int       NX,
                               int       NY,
                               int       NZ,
                               int       IDYNA,
                               int       NVE,
                               int       SoCalQ,
                               Grid1D    i_inputBuffer,
                               int_pt    i_inputSizeX,
                               int_pt    i_inputSizeY,
                               int_pt    i_inputSizeZ
                             ) {
  int_pt  i, j, k, offset, l_readOffset;
  real    vp, vs, dd, pi;
  real    max_vse  = -1.0e10;
  real    max_vpe  = -1.0e10;
  real    max_dde  = -1.0e10;
  real    min_vse  = 1.0e10;
  real    min_vpe  = 1.0e10;
  real    min_dde  = 1.0e10;

  pi        = 4.0 * atan( 1.0 );
  *taumax   = 1.0 / (2.0 * pi * 0.01) * 1.0 * FAC;
  if( EX < 0.65 && EX >= 0.01 ) {
    *taumin = 1.0 / (2.0 * pi * 10.0) * 0.2 * FAC;
  } else if( EX < 0.85 && EX >= 0.65 ) {
    *taumin = 1.0 / (2.0 * pi * 12.0) * 0.5 * FAC;
    *taumax = 1.0 / (2.0 * pi * 0.08) * 2.0 * FAC;
  } else if( EX < 0.95 && EX >= 0.85 ) {
    *taumin = 1.0 / (2.0 * pi * 15.0) * 0.8 * FAC;
    *taumax = 1.0 / (2.0 * pi * 0.10) * 2.5 * FAC;
  } else if( EX < 0.01 ) {
    *taumin = 1.0 / (2.0 * pi * 10.0) * 0.2 * FAC;
  }

  tausub( tau, *taumin, *taumax );

  if( MEDIASTART != 0 && MEDIASTART != 4 && odc::parallel::Mpi::m_size != 1 ) {
    std::cerr << "Error: Only MEDIASTART equal to 0 and 4 are currently supported by MPI." << std::endl;
    std::cerr << "\tReverting to MEDIASTART=0" << std::endl;
    MEDIASTART = 0;
  }

  //! MEDIASTART 0 corresponds to a homogeneous mesh with hardcoded material parameters.
  if( MEDIASTART == 0 ) {
    if( IDYNA == 1 ) {
      vp  = 6000.0;
      vs  = 3464.0;
      dd  = 2670.0;
    } else {
      vp  = 1800.0;   //! was 4800
      vs  = 1600.0;   //! was 2800
      dd  = 2500.0;
    }

    max_vpe = min_vpe = vp;
    max_vse = min_vse = vs;
    max_dde = min_dde = dd;

    for( i = -1; i < nxt + 1; i++ ) {
      for( j = -1; j < nyt + 1; j++ ) {
        for( k = -1; k < nzt + 1; k++ ) {
          //TODO(Josh): optimize this
          offset  = i * i_strideX + j * i_strideY + k * i_strideZ;

          if( NVE == 1 ) {
            qp[offset] = 0.00416667;
            qs[offset] = 0.00833333;
          }

#ifdef YASK
          lam_grid->writeElem( 1.0 / (dd * (vp * vp - 2. * vs * vs)), i + bdry_width, j + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem( 1.0 / (dd * vs * vs), i + bdry_width, j + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( dd, i + bdry_width, j + bdry_width, k + bdry_width, 0 );
#else
          lam[offset] = 1.0 / (dd * (vp * vp - 2.0 * vs * vs));
          mu[offset]  = 1.0 / (dd * vs * vs);
          d1[offset]  = dd;
#endif
        }
      }
    }
  }

  //! (Raj): Generates grid from user-provided mesh file (*.bin.*)
  else if( MEDIASTART == 4 ) {
    for( k = 0; k < nzt; k++ ) {
      for( j = 0; j < nyt; j++ ) {
        for( i = 0; i < nxt; i++ ) {
          l_readOffset  = (k * nyt * nxt + j * nxt + i) * nvar;
          vp            = i_inputBuffer[l_readOffset+0];
          vs            = i_inputBuffer[l_readOffset+1];
          dd            = i_inputBuffer[l_readOffset+2];

          if( vs < min_vse )
            min_vse = vs;
          if( vs > max_vse )
            max_vse = vs;
          if( vp < min_vpe )
            min_vpe = vp;
          if( vp > max_vpe )
            max_vpe = vp;
          if( dd < min_dde )
            min_dde = dd;
          if( dd > max_dde )
            max_dde = dd;

          offset        = i * i_strideX + j * i_strideY + k * i_strideZ;
#ifdef YASK
          lam_grid->writeElem( 1.0 / (dd * (vp * vp - 2.0 * vs * vs)), i + bdry_width, j + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem( 1.0 / (dd * vs * vs), i + bdry_width, j + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( dd, i + bdry_width, j + bdry_width, k + bdry_width, 0 );
#else
          lam[offset]   = 1.0 / (dd * (vp * vp - 2.0 * vs * vs));
          mu[offset]    = 1.0 / (dd * vs * vs);
          d1[offset]    = dd;
#endif

          //! If viscoelastic (by default 1)
          if( NVE == 1 ) {
            qp[offset]  = i_inputBuffer[l_readOffset+3];
            qs[offset]  = i_inputBuffer[l_readOffset+4];
          }
        }
      }
    }

    /**
      (Raj): Important note on coordinate-system followed internally :
        x-direction : towards right
        y-direction : into the paper/screen
        z-direction : upwards
        origin      : lower-left corner towards screen/paper
    */

#ifdef AWP_USE_MPI
    int         leftNeighbor  = odc::parallel::Mpi::m_neighborRanks[0][1][1];
    int         rightNeighbor = odc::parallel::Mpi::m_neighborRanks[2][1][1];
    int         botNeighbor   = odc::parallel::Mpi::m_neighborRanks[1][0][1];
    int         topNeighbor   = odc::parallel::Mpi::m_neighborRanks[1][2][1];
    int_pt      arrSize1      = 6 * (int_pt) nzt * (int_pt) nyt;
    int_pt      arrSize2      = 6 * (int_pt) nzt * (int_pt) nxt;

    real        sendBuff1[6][nzt][nyt], recvBuff1[6][nzt][nyt];
    real        sendBuff2[6][nzt][nxt], recvBuff2[6][nzt][nxt];

    MPI_Request reqs[2];
    MPI_Status  stats[2];

    //! If right neighbor present
    if( rightNeighbor != -1 ) {
      for( k = 0; k < nzt; k++ ) {
        for( j = 0; j < nyt; j++ ) {
#ifdef YASK
          //! Reads grid-properties (lambda, mu and density) for grid-points -1 (x-dir) from right-face (yz-plane)
          sendBuff1[0][k][j]  = lam_grid->readElem(     nxt - 2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[1][k][j]  = mu_grid->readElem(      nxt - 2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[2][k][j]  = density_grid->readElem( nxt - 2 + bdry_width, j + bdry_width, k + bdry_width, 0 );

          //! Reads grid-properties (lambda, mu and density) for grid-points -2 (x-dir) from right-face in (yz-plane)
          sendBuff1[3][k][j]  = lam_grid->readElem(     nxt - 3 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[4][k][j]  = mu_grid->readElem(      nxt - 3 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[5][k][j]  = density_grid->readElem( nxt - 3 + bdry_width, j + bdry_width, k + bdry_width, 0 );
#else
          //! Reads grid-properties (lambda, mu and density) for grid-points -1 (x-dir) from right-face (yz-plane)
          offset              = (nxt - 2) * i_strideX + j * i_strideY + k * i_strideZ;
          sendBuff1[0][k][j]  = lam[offset];
          sendBuff1[1][k][j]  = mu[offset];
          sendBuff1[2][k][j]  = d1[offset];

          //! Reads grid-properties (lambda, mu and density) for grid-points -2 (x-dir) from right-face (yz-plane)
          offset              = (nxt - 3) * i_strideX + j * i_strideY + k * i_strideZ;
          sendBuff1[3][k][j]  = lam[offset];
          sendBuff1[4][k][j]  = mu[offset];
          sendBuff1[5][k][j]  = d1[offset];
#endif
        }
      }

      //! Receives buffer over channel 1 and sends buffer over channel 2 b/w self & right neighbor
      MPI_Irecv( &recvBuff1[0][0][0], arrSize1, AWP_MPI_REAL, rightNeighbor, 1, MPI_COMM_WORLD, &reqs[0] );
      MPI_Isend( &sendBuff1[0][0][0], arrSize1, AWP_MPI_REAL, rightNeighbor, 2, MPI_COMM_WORLD, &reqs[1] );

      //! Wait until above communication has finished (crucial)
      MPI_Waitall( 2, reqs, stats );

      for( k = 0; k < nzt; k++ ) {
        for( j = 0; j < nyt; j++ ) {
#ifdef YASK
          //! Writes grid-properties (lambda, mu and density) for grid-points +1 (x-dir) from right-face (yz-plane)
          lam_grid->writeElem(     recvBuff1[0][k][j], nxt + 0 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff1[1][k][j], nxt + 0 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff1[2][k][j], nxt + 0 + bdry_width, j + bdry_width, k + bdry_width, 0 );

          //! Writes grid-properties (lambda, mu and density) for grid-points +2 (x-dir) from right-face (yz-plane)
          lam_grid->writeElem(     recvBuff1[3][k][j], nxt + 1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff1[4][k][j], nxt + 1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff1[5][k][j], nxt + 1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
#else
          //! Writes grid-properties (lambda, mu and density) for grid-points +1 (x-dir) from right-face (yz-plane)
          offset      = (nxt + 0) * i_strideX + j * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff1[0][k][j];
          mu[offset]  = recvBuff1[1][k][j];
          d1[offset]  = recvBuff1[2][k][j];

          //! Writes grid-properties (lambda, mu and density) for grid-points +2 (x-dir) from right-face (yz-plane)
          offset      = (nxt + 1) * i_strideX + j * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff1[3][k][j];
          mu[offset]  = recvBuff1[4][k][j];
          d1[offset]  = recvBuff1[5][k][j];
#endif
        }
      }
    }

    //! If left neighbor present
    if( leftNeighbor != -1 ) {
      for( k = 0; k < nzt; k++ ) {
        for( j = 0; j < nyt; j++ ) {
#ifdef YASK
          //! Reads grid-properties (lambda, mu and density) for grid-points +1 (x-dir) from left-face (yz-plane)
          sendBuff1[0][k][j]  = lam_grid->readElem(     1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[1][k][j]  = mu_grid->readElem(      1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[2][k][j]  = density_grid->readElem( 1 + bdry_width, j + bdry_width, k + bdry_width, 0 );

          //! Reads grid-properties (lambda, mu and density) for grid-points +2 (x-dir) from left-face (yz-plane)
          sendBuff1[3][k][j]  = lam_grid->readElem(     2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[4][k][j]  = mu_grid->readElem(      2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          sendBuff1[5][k][j]  = density_grid->readElem( 2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
#else
          //! Reads grid-properties (lambda, mu and density) for grid-points +1 (x-dir) from left-face (yz-plane)
          offset              = 1 * i_strideX + j * i_strideY + k * i_strideZ;
          sendBuff1[0][k][j]  = lam[offset];
          sendBuff1[1][k][j]  = mu[offset];
          sendBuff1[2][k][j]  = d1[offset];

          //! Reads grid-properties (lambda, mu and density) for grid-points +2 (x-dir) from left-face (yz-plane)
          offset              = 2 * i_strideX + j * i_strideY + k * i_strideZ;
          sendBuff1[3][k][j]  = lam[offset];
          sendBuff1[4][k][j]  = mu[offset];
          sendBuff1[5][k][j]  = d1[offset];
#endif
        }
      }

      //! Receives buffer over channel 2 and sends buffer over channel 1 b/w self & left neighbor
      MPI_Irecv( &recvBuff1[0][0][0], arrSize1, AWP_MPI_REAL, leftNeighbor, 2, MPI_COMM_WORLD, &reqs[0] );
      MPI_Isend( &sendBuff1[0][0][0], arrSize1, AWP_MPI_REAL, leftNeighbor, 1, MPI_COMM_WORLD, &reqs[1] );

      //! Wait until above communication has finished (crucial)
      MPI_Waitall( 2, reqs, stats );

      for( k = 0; k < nzt; k++ ) {
        for( j = 0; j < nyt; j++ ) {
#ifdef YASK
          //! Writes grid-properties (lambda, mu and density) for grid-points -1 (x-dir) from left-face (yz-plane)
          lam_grid->writeElem(     recvBuff1[0][k][j], -1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff1[1][k][j], -1 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff1[2][k][j], -1 + bdry_width, j + bdry_width, k + bdry_width, 0 );

          //! Writes grid-properties (lambda, mu and density) for grid-points -2 (x-dir) from left-face (yz-plane)
          lam_grid->writeElem(     recvBuff1[3][k][j], -2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff1[4][k][j], -2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff1[5][k][j], -2 + bdry_width, j + bdry_width, k + bdry_width, 0 );
#else
          //! Writes grid-properties (lambda, mu and density) for grid-points -1 (x-dir) from left-face (yz-plane)
          offset      = -1 * i_strideX + j * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff1[0][k][j];
          mu[offset]  = recvBuff1[1][k][j];
          d1[offset]  = recvBuff1[2][k][j];

          //! Writes grid-properties (lambda, mu and density) for grid-points -2 (x-dir) from left-face (yz-plane)
          offset      = -2 * i_strideX + j * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff1[3][k][j];
          mu[offset]  = recvBuff1[4][k][j];
          d1[offset]  = recvBuff1[5][k][j];
#endif
        }
      }
    }

    //! If bottom neighbor present
    if( botNeighbor != -1 ) {
      for( k = 0; k < nzt; k++ ) {
        for( i = 0; i < nxt; i++ ) {
#ifdef YASK
          //! Reads grid-properties (lambda, mu and density) for grid-points +1 (y-dir) from bottom-face (xz-plane)
          sendBuff2[0][k][i]  = lam_grid->readElem(     i + bdry_width, 1 + bdry_width, k + bdry_width, 0 );
          sendBuff2[1][k][i]  = mu_grid->readElem(      i + bdry_width, 1 + bdry_width, k + bdry_width, 0 );
          sendBuff2[2][k][i]  = density_grid->readElem( i + bdry_width, 1 + bdry_width, k + bdry_width, 0 );

          //! Reads grid-properties (lambda, mu and density) for grid-points +2 (y-dir) from bottom-face (xz-plane)
          sendBuff2[3][k][i]  = lam_grid->readElem(     i + bdry_width, 2 + bdry_width, k + bdry_width, 0 );
          sendBuff2[4][k][i]  = mu_grid->readElem(      i + bdry_width, 2 + bdry_width, k + bdry_width, 0 );
          sendBuff2[5][k][i]  = density_grid->readElem( i + bdry_width, 2 + bdry_width, k + bdry_width, 0 );
#else
          //! Reads grid-properties (lambda, mu and density) for grid-points +1 (y-dir) from bottom-face (xz-plane)
          offset              = i * i_strideX + 1 * i_strideY + k * i_strideZ;
          sendBuff2[0][k][j]  = lam[offset];
          sendBuff2[1][k][j]  = mu[offset];
          sendBuff2[2][k][j]  = d1[offset];

          //! Reads grid-properties (lambda, mu and density) for grid-points +2 (y-dir) from bottom-face (xz-plane)
          offset              = i * i_strideX + 2 * i_strideY + k * i_strideZ;
          sendBuff2[3][k][j]  = lam[offset];
          sendBuff2[4][k][j]  = mu[offset];
          sendBuff2[5][k][j]  = d1[offset];
#endif
        }
      }

      //! Receives buffer over channel 3 and sends buffer over channel 4 b/w self & bottom neighbor
      MPI_Irecv( &recvBuff2[0][0][0], arrSize2, AWP_MPI_REAL, botNeighbor, 3, MPI_COMM_WORLD, &reqs[0] );
      MPI_Isend( &sendBuff2[0][0][0], arrSize2, AWP_MPI_REAL, botNeighbor, 4, MPI_COMM_WORLD, &reqs[1] );

      //! Wait until above communication has finished (crucial)
      MPI_Waitall( 2, reqs, stats );

      for( k = 0; k < nzt; k++ ) {
        for( i = 0; i < nxt; i++ ) {
#ifdef YASK
          //! Writes grid-properties (lambda, mu and density) for grid-points -1 from bottom-face in xy-plane
          lam_grid->writeElem(     recvBuff2[0][k][i], i + bdry_width, -1 + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff2[1][k][i], i + bdry_width, -1 + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff2[2][k][i], i + bdry_width, -1 + bdry_width, k + bdry_width, 0 );

          //! Writes grid-properties (lambda, mu and density) for grid-points -2 from bottom-face in xy-plane
          lam_grid->writeElem(     recvBuff2[3][k][i], i + bdry_width, -2 + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff2[4][k][i], i + bdry_width, -2 + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff2[5][k][i], i + bdry_width, -2 + bdry_width, k + bdry_width, 0 );
#else
          //! Writes grid-properties (lambda, mu and density) for grid-points -1 from bottom-face in xy-plane
          offset      = i * i_strideX - 1 * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff2[0][k][j];
          mu[offset]  = recvBuff2[1][k][j];
          d1[offset]  = recvBuff2[2][k][j];

          //! Writes grid-properties (lambda, mu and density) for grid-points -2 from bottom-face in xy-plane
          offset      = i * i_strideX - 2 * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff2[3][k][j];
          mu[offset]  = recvBuff2[4][k][j];
          d1[offset]  = recvBuff2[5][k][j];
#endif
        }
      }
    }

    //! If top neighbor present
    if( topNeighbor != -1 ) {
      for( k = 0; k < nzt; k++ ) {
        for( i = 0; i < nxt; i++ ) {
#ifdef YASK
          //! Reads grid-properties (lambda, mu and density) for grid-points -1 (y-dir) from top-face (xz-plane)
          sendBuff2[0][k][i]  = lam_grid->readElem(     i + bdry_width, nyt - 2 + bdry_width, k + bdry_width, 0 );
          sendBuff2[1][k][i]  = mu_grid->readElem(      i + bdry_width, nyt - 2 + bdry_width, k + bdry_width, 0 );
          sendBuff2[2][k][i]  = density_grid->readElem( i + bdry_width, nyt - 2 + bdry_width, k + bdry_width, 0 );

          //! Reads grid-properties (lambda, mu and density) for grid-points -2 (y-dir) from top-face (xz-plane)
          sendBuff2[3][k][i]  = lam_grid->readElem(     i + bdry_width, nyt - 3 + bdry_width, k + bdry_width, 0 );
          sendBuff2[4][k][i]  = mu_grid->readElem(      i + bdry_width, nyt - 3 + bdry_width, k + bdry_width, 0 );
          sendBuff2[5][k][i]  = density_grid->readElem( i + bdry_width, nyt - 3 + bdry_width, k + bdry_width, 0 );
#else
          //! Reads grid-properties (lambda, mu and density) for grid-points -1 (y-dir) from top-face (xz-plane)
          offset              = i * i_strideX + (nyt - 2) * i_strideY + k * i_strideZ;
          sendBuff2[0][k][j]  = lam[offset];
          sendBuff2[1][k][j]  = mu[offset];
          sendBuff2[2][k][j]  = d1[offset];

          //! Reads grid-properties (lambda, mu and density) for grid-points -2 (y-dir) from top-face (xz-plane)
          offset              = i * i_strideX + (nyt - 3) * i_strideY + k * i_strideZ;
          sendBuff2[3][k][j]  = lam[offset];
          sendBuff2[4][k][j]  = mu[offset];
          sendBuff2[5][k][j]  = d1[offset];
#endif
        }
      }

      //! Receives buffer over channel 4 and sends buffer over channel 3 b/w self & top neighbor
      MPI_Irecv( &recvBuff2[0][0][0], arrSize2, AWP_MPI_REAL, topNeighbor, 4, MPI_COMM_WORLD, &reqs[0] );
      MPI_Isend( &sendBuff2[0][0][0], arrSize2, AWP_MPI_REAL, topNeighbor, 3, MPI_COMM_WORLD, &reqs[1] );

      //! Wait until above communication has finished (crucial)
      MPI_Waitall( 2, reqs, stats );

      for( k = 0; k < nzt; k++ ) {
        for( i = 0; i < nxt; i++ ) {
#ifdef YASK
          //! Writes grid-properties (lambda, mu and density) for grid-points +1 (y-dir) from top-face (xz-plane)
          lam_grid->writeElem(     recvBuff2[0][k][i], i + bdry_width, nyt + 0 + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff2[1][k][i], i + bdry_width, nyt + 0 + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff2[2][k][i], i + bdry_width, nyt + 0 + bdry_width, k + bdry_width, 0 );

          //! Writes grid-properties (lambda, mu and density) for grid-points +2 (y-dir) from top-face (xz-plane)
          lam_grid->writeElem(     recvBuff2[3][k][i], i + bdry_width, nyt + 1 + bdry_width, k + bdry_width, 0 );
          mu_grid->writeElem(      recvBuff2[4][k][i], i + bdry_width, nyt + 1 + bdry_width, k + bdry_width, 0 );
          density_grid->writeElem( recvBuff2[5][k][i], i + bdry_width, nyt + 1 + bdry_width, k + bdry_width, 0 );
#else
          //! Writes grid-properties (lambda, mu and density) for grid-points +1 (y-dir) from top-face (xz-plane)
          offset      = i * i_strideX + (nyt + 0) * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff2[0][k][j];
          mu[offset]  = recvBuff2[1][k][j];
          d1[offset]  = recvBuff2[2][k][j];

          //! Writes grid-properties (lambda, mu and density) for grid-points +2 (y-dir) from top-face (xz-plane)
          offset      = i * i_strideX + (nyt + 1) * i_strideY + k * i_strideZ;
          lam[offset] = recvBuff2[3][k][j];
          mu[offset]  = recvBuff2[4][k][j];
          d1[offset]  = recvBuff2[5][k][j];
#endif
        }
      }
    }
#endif
  }

  else {
    int var_offset;
    Grid3D tmpvp = NULL, tmpvs = NULL, tmpdd = NULL;
    Grid3D tmppq = NULL, tmpsq = NULL;

    tmpvp = Alloc3D( nxt, nyt, nzt );
    tmpvs = Alloc3D( nxt, nyt, nzt );
    tmpdd = Alloc3D( nxt, nyt, nzt );

#pragma omp parallel for collapse( 3 )
    for( i = 0; i < nxt; i++ ) {
      for( j = 0; j < nyt; j++ ) {
        for( k = 0; k < nzt; k++ ) {
          tmpvp[i][j][k]  = 0.0f;
          tmpvs[i][j][k]  = 0.0f;
          tmpdd[i][j][k]  = 0.0f;
        }
      }
    }

    tmppq = Alloc3D( nxt, nyt, nzt );
    tmpsq = Alloc3D( nxt, nyt, nzt );

    if( NVE == 1 ) {
#pragma omp parallel for collapse( 3 )
      for( i = 0; i < nxt; i++ ) {
        for( j = 0; j < nyt; j++ ) {
          for( k = 0; k < nzt; k++ ) {
            tmppq[i][j][k]  = 0.0f;
            tmpsq[i][j][k]  = 0.0f;
          }
        }
      }
    }

    if( nvar == 8 )
      var_offset  = 3;
    else if( nvar == 5 )
      var_offset  = 0;
    else
      var_offset  = 0;

    if( MEDIASTART >= 1 && MEDIASTART <= 3 ) {
#pragma omp parallel for collapse( 3 )
      for( k = 0; k < nzt; k++ ) {
        for( j = 0; j < nyt; j++ ) {
          for( i = 0; i < nxt; i++ ) {
            l_readOffset      = (k * i_inputSizeY * i_inputSizeX + j * i_inputSizeX + i) * nvar;
            tmpvp[i][j][k]    = i_inputBuffer[l_readOffset+var_offset];
            tmpvs[i][j][k]    = i_inputBuffer[l_readOffset+var_offset+1];
            tmpdd[i][j][k]    = i_inputBuffer[l_readOffset+var_offset+2];

            if( nvar > 3 ) {
              tmppq[i][j][k]  = i_inputBuffer[l_readOffset+var_offset+3];
              tmpsq[i][j][k]  = i_inputBuffer[l_readOffset+var_offset+4];
            }

            if( tmpvp[i][j][k] != tmpvp[i][j][k] ||
                tmpvs[i][j][k] != tmpvs[i][j][k] ||
                tmpdd[i][j][k] != tmpdd[i][j][k] ) {
#ifdef AWP_USE_MPI
              MPI_Abort( MPI_COMM_WORLD, 1 );
#endif
            }
          }
        }
      }
    }

    float w0 = 0.0f;
    if( NVE == 1 ) {
      w0  = 2.0 * pi * FP;
    }

    float facex = (float) pow( FAC, EX );

#pragma omp parallel for collapse( 2 )
    for( i = 0; i < nxt; i++ ) {
      for( j = 0; j < nyt; j++ ) {
        float weights_los[2][2][2];
        float weights_lop[2][2][2];
        float val[2];
        float mu1, denom;
        float qpinv = 0.0f, qsinv = 0.0f, vpvs = 0.0f;
        int ii, jj, kk, iii, jjj, kkk, num;
        std::complex< double > value( 0.0, 0.0 );
        std::complex< double > sqrtm1( 0.0, 1.0 );

        for( k = 0; k < nzt; k++ ) {
          if( tmpvs[i][j][k] < 200.0 ) {
            tmpvs[i][j][k]  = 200.0;
            tmpvp[i][j][k]  = 600.0;
          }

          tmpsq[i][j][k]    = 0.1 * tmpvs[i][j][k];
          tmppq[i][j][k]    = 2.0 * tmpsq[i][j][k];

          if( tmppq[i][j][k] > 200.0 ) {
            // QF - VP
            val[0] = 0.0;
            val[1] = 0.0;
            for( ii = 0; ii < 2; ii++ ) {
              for( jj = 0; jj < 2; jj++ ) {
                for( kk = 0; kk < 2; kk++ ) {
                  denom   = ((w0 * w0 * tau[ii][jj][kk] * tau[ii][jj][kk] + 1.0) * tmppq[i][j][k] * facex);
                  val[0]  += weights[ii][jj][kk] / denom;
                  val[1]  += -weights[ii][jj][kk] * w0 * tau[ii][jj][kk] / denom;
                }
              }
            }

            mu1 = tmpdd[i][j][k] * tmpvp[i][j][k] * tmpvp[i][j][k] / (1.0 - val[0]);
          } else {
            num = 0;
            for( iii = 0; iii < 2; iii++ ) {
              for( jjj = 0; jjj < 2; jjj++ ) {
                for( kkk = 0; kkk < 2; kkk++ ) {
                  weights_lop[iii][jjj][kkk] = coeff[num] / (tmppq[i][j][k] * tmppq[i][j][k]) + coeff[num + 1] / (tmppq[i][j][k]);
                  num = num + 2;
                }
              }
            }

            value = 0.0 + 0.0 * sqrtm1;
            for( ii = 0; ii < 2; ii++ ) {
              for( jj = 0; jj < 2; jj++ ) {
                for( kk = 0; kk < 2; kk++ ) {
                  value = value + 1.0 / (1.0 - ((double) weights_lop[ii][jj][kk]) / (1.0 + sqrtm1 * ((double) w0 * tau[ii][jj][kk])));
                }
              }
            }

            value = 1.0 / value;
            mu1   = tmpdd[i][j][k] * tmpvp[i][j][k] * tmpvp[i][j][k] / (8.0 * std::real( value ));
          }

          tmpvp[i][j][k] = sqrt( mu1 / tmpdd[i][j][k] );

          // QF - VS
          if( tmpsq[i][j][k] > 200.0 ) {
            val[0] = 0.0;
            val[1] = 0.0;
            for( ii = 0; ii < 2; ii++ ) {
              for( jj = 0; jj < 2; jj++ ) {
                for( kk = 0; kk < 2; kk++ ) {
                  denom   = ((w0 * w0 * tau[ii][jj][kk] * tau[ii][jj][kk] + 1.0) * tmpsq[i][j][k] * facex);
                  val[0]  += weights[ii][jj][kk] / denom;
                  val[1]  += -weights[ii][jj][kk] * w0 * tau[ii][jj][kk] / denom;
                }
              }
            }
            mu1 = tmpdd[i][j][k] * tmpvs[i][j][k] * tmpvs[i][j][k] / (1.0 - val[0]);
          } else {
            num = 0;
            for( iii = 0; iii < 2; iii++ ) {
              for( jjj = 0; jjj < 2; jjj++ ) {
                for( kkk = 0; kkk < 2; kkk++ ) {
                  weights_los[iii][jjj][kkk] = coeff[num] / (tmpsq[i][j][k] * tmpsq[i][j][k]) + coeff[num + 1] / (tmpsq[i][j][k]);
                  num = num + 2;
                }
              }
            }

            value = 0.0 + 0.0 * sqrtm1;
            for( ii = 0; ii < 2; ii++ ) {
              for( jj = 0; jj < 2; jj++ ) {
                for( kk = 0; kk < 2; kk++ ) {
                  value = value + 1.0 / (1.0 - ((double) weights_los[ii][jj][kk]) / (1.0 + sqrtm1 * ((double) w0 * tau[ii][jj][kk])));
                }
              }
            }
            value = 1.0 / value;
            mu1   = tmpdd[i][j][k] * tmpvs[i][j][k] * tmpvs[i][j][k] / (8.0 * std::real(value));
          }

          tmpvs[i][j][k] = sqrt( mu1 / tmpdd[i][j][k] );

          // QF - end
          if( SoCalQ == 1 ) {
            vpvs  = tmpvp[i][j][k] / tmpvs[i][j][k];
            if( vpvs < 1.45 )
              tmpvs[i][j][k]  = tmpvp[i][j][k] / 1.45;
          }

          if( tmpvp[i][j][k] > 7600.0 ) {
            tmpvs[i][j][k]  = 4387.0;
            tmpvp[i][j][k]  = 7600.0;
          }

          if( tmpvs[i][j][k] < 200.0 ) {
            tmpvs[i][j][k]  = 200.0;
            tmpvp[i][j][k]  = 600.0;
          }

          offset  = i * i_strideX + j * i_strideY + (nzt - 1 - k) * i_strideZ;
          if( tmpdd[i][j][k] < 1700.0 )
            tmpdd[i][j][k]  = 1700.0;

#ifdef YASK
          mu_grid->writeElem( 1.0 / (tmpdd[i][j][k] * tmpvs[i][j][k] * tmpvs[i][j][k]),
                              i + bdry_width, j + bdry_width, nzt - 1 - k + bdry_width, 0 );
          lam_grid->writeElem( 1.0 / (tmpdd[i][j][k] * (tmpvp[i][j][k] * tmpvp[i][j][k]
                               - 2.0 * tmpvs[i][j][k] * tmpvs[i][j][k])),
                               i + bdry_width, j + bdry_width, nzt - 1 - k + bdry_width, 0 );
          density_grid->writeElem( tmpdd[i][j][k], i + bdry_width, j + bdry_width, nzt - 1 - k + bdry_width, 0 );
#else
          mu[offset]  = 1.0 / (tmpdd[i][j][k] * tmpvs[i][j][k] * tmpvs[i][j][k]);
          lam[offset] = 1.0 / (tmpdd[i][j][k] * (tmpvp[i][j][k] * tmpvp[i][j][k]
                                          -2.0 * tmpvs[i][j][k] * tmpvs[i][j][k]));
          d1[offset]  = tmpdd[i][j][k];
#endif

          if( NVE == 1 ) {
            if( tmppq[i][j][k] <= 0.0 ) {
              qpinv = 0.0;
              qsinv = 0.0;
            } else {
              qpinv = 1.0 / tmppq[i][j][k];
              qsinv = 1.0 / tmpsq[i][j][k];
            }

            tmppq[i][j][k]  = qpinv / facex;
            tmpsq[i][j][k]  = qsinv / facex;
            qp[offset]      = tmppq[i][j][k];
            qs[offset]      = tmpsq[i][j][k];
          }
        }
      }
    }

#pragma omp parallel for reduction( max: max_vpe, max_vse, max_dde ), reduction( min: min_vpe, min_vse, min_dde ) collapse( 2 )
    for( i = 0; i < nxt; i++ ) {
      for( j = 0; j < nyt; j++ ) {
        for( k = 0; k < nzt; k++ ) {
          if( tmpvs[i][j][k] < min_vse )
            min_vse = tmpvs[i][j][k];
          if( tmpvs[i][j][k] > max_vse )
            max_vse = tmpvs[i][j][k];
          if( tmpvp[i][j][k] < min_vpe )
            min_vpe = tmpvp[i][j][k];
          if( tmpvp[i][j][k] > max_vpe )
            max_vpe = tmpvp[i][j][k];
          if( tmpdd[i][j][k] < min_dde )
            min_dde = tmpdd[i][j][k];
          if( tmpdd[i][j][k] > max_dde )
            max_dde = tmpdd[i][j][k];
        }
      }
    }

    Delloc3D( tmpvp );
    Delloc3D( tmpvs );
    Delloc3D( tmpdd );

    Delloc3D( tmppq );
    Delloc3D( tmpsq );
  }

#ifdef AWP_USE_MPI
  //! Getting max and min values of Vs, Vp and density across all MPIs
  MPI_Allreduce( &min_vse, &m_vse[0], 1, AWP_MPI_REAL, MPI_MIN, MPI_COMM_WORLD );
  MPI_Allreduce( &max_vse, &m_vse[1], 1, AWP_MPI_REAL, MPI_MAX, MPI_COMM_WORLD );
  MPI_Allreduce( &min_vpe, &m_vpe[0], 1, AWP_MPI_REAL, MPI_MIN, MPI_COMM_WORLD );
  MPI_Allreduce( &max_vpe, &m_vpe[1], 1, AWP_MPI_REAL, MPI_MAX, MPI_COMM_WORLD );
  MPI_Allreduce( &min_dde, &m_dde[0], 1, AWP_MPI_REAL, MPI_MIN, MPI_COMM_WORLD );
  MPI_Allreduce( &max_dde, &m_dde[1], 1, AWP_MPI_REAL, MPI_MAX, MPI_COMM_WORLD );
#else
  m_vse[0] = min_vse;
  m_vse[1] = max_vse;
  m_vpe[0] = min_vpe;
  m_vpe[1] = max_vpe;
  m_dde[0] = min_dde;
  m_dde[1] = max_dde;
#endif
}

/**

   @param taumin  -
   @param taumax  -
 
   @param[out] tau     -
*/
void odc::data::Mesh::tausub( Grid3D tau, float taumin,float taumax ) {
  int   idx, idy, idz;
  float tautem[2][2][2];
  float tmp;

  //(gwilkins) Why use this access pattern?
  tautem[0][0][0] = 1.0;
  tautem[1][0][0] = 6.0;
  tautem[0][1][0] = 7.0;
  tautem[1][1][0] = 4.0;
  tautem[0][0][1] = 8.0;
  tautem[1][0][1] = 3.0;
  tautem[0][1][1] = 2.0;
  tautem[1][1][1] = 5.0;

  for( idx = 0; idx < 2; idx++ ) {
    for( idy = 0; idy < 2; idy++ ) {
      for( idz = 0; idz < 2; idz++ ) {
        tmp = tautem[idx][idy][idz];
        tmp = (tmp - 0.5) / 8.0;
        tmp = 2.0 * tmp - 1.0;

        tau[idx][idy][idz] = exp(0.5 * (log(taumax * taumin) + log(taumax / taumin) * tmp));
      }
    }
  }
  return;
}

// TODO(Josh): clean this
void odc::data::Mesh::init_texture( Grid3D tau1, Grid3D tau2, Grid3D vx1, Grid3D vx2, Grid3D weights, Grid3Dww ww, Grid3D wwo,
                                    int_pt startX, int_pt endX, int_pt startY, int_pt endY, int_pt startZ, int_pt endZ,
                                    int_pt globalStartX, int_pt globalStartY, int_pt globalStartZ, int_pt sizeZ ) {
  int_pt i, j, k, itx, ity, itz;

  itx = (std::abs( (int_pt)(globalStartX + startX) )) % 2;
  ity = (std::abs( (int_pt)(globalStartY + startY) )) % 2;
  itz = (std::abs( (int_pt)(globalStartZ + startZ + sizeZ - 1) )) % 2;

  for( i = startX; i < endX; i++ ) {
    itx = 1 - itx;
    for( j = startY; j < endY; j++ ) {
      ity = 1 - ity;
      for( k = startZ; k < endZ; k++ ) {
        itz           = 1 - itz;
        vx1[i][j][k]  = tau1[itx][ity][itz];
        vx2[i][j][k]  = tau2[itx][ity][itz];
        wwo[i][j][k]  = 8.0 * weights[itx][ity][itz];
        if( itx < 0.5 && ity < 0.5 && itz < 0.5 )
          ww[i][j][k] = 1;
        else if( itx < 0.5 && ity < 0.5 && itz > 0.5 )
          ww[i][j][k] = 2;
        else if( itx < 0.5 && ity > 0.5 && itz < 0.5 )
          ww[i][j][k] = 3;
        else if( itx < 0.5 && ity > 0.5 && itz > 0.5 )
          ww[i][j][k] = 4;
        else if( itx > 0.5 && ity < 0.5 && itz < 0.5 )
          ww[i][j][k] = 5;
        else if( itx > 0.5 && ity < 0.5 && itz > 0.5 )
          ww[i][j][k] = 6;
        else if( itx > 0.5 && ity > 0.5 && itz < 0.5 )
          ww[i][j][k] = 7;
        else if( itx > 0.5 && ity > 0.5 && itz > 0.5 )
          ww[i][j][k] = 8;
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
void odc::data::Mesh::weights_sub( Grid3D weights, Grid1D coeff, float ex, float fac ) {
  int i, j, k;

  if( ex < 0.15 && ex >= 0.01 ) {
    weights[0][0][0]  = 0.3273;
    weights[1][0][0]  = 1.0434;
    weights[0][1][0]  = 0.044;
    weights[1][1][0]  = 0.9393;
    weights[0][0][1]  = 1.7268;
    weights[1][0][1]  = 0.369;
    weights[0][1][1]  = 0.8478;
    weights[1][1][1]  = 0.4474;
  
    coeff[0]          = 7.3781;
    coeff[1]          = 4.1655;
    coeff[2]          = -83.1627;
    coeff[3]          = 13.1326;
    coeff[4]          = 69.0839;
    coeff[5]          = 0.4981;
    coeff[6]          = -37.6966;
    coeff[7]          = 5.5263;
    coeff[8]          = -51.4056;
    coeff[9]          = 8.1934;
    coeff[10]         = 13.1865;
    coeff[11]         = 3.4775;
    coeff[12]         = -36.1049;
    coeff[13]         = 7.2107;
    coeff[14]         = 12.3809;
    coeff[15]         = 3.6117;
  } else if( ex < 0.25 && ex >= 0.15 ) {
    weights[0][0][0]  = 0.001;
    weights[1][0][0]  = 1.0349;
    weights[0][1][0]  = 0.0497;
    weights[1][1][0]  = 1.0407;
    weights[0][0][1]  = 1.7245;
    weights[1][0][1]  = 0.2005;
    weights[0][1][1]  = 0.804;
    weights[1][1][1]  = 0.4452;

    coeff[0]          = 31.8902;
    coeff[1]          = 1.6126;
    coeff[2]          = -83.2611;
    coeff[3]          = 13.0749;
    coeff[4]          = 65.485;
    coeff[5]          = 0.5118;
    coeff[6]          = -42.02;
    coeff[7]          = 5.0875;
    coeff[8]          = -49.2656;
    coeff[9]          = 8.1552;
    coeff[10]         = 25.7345;
    coeff[11]         = 2.2801;
    coeff[12]         = -40.8942;
    coeff[13]         = 7.9311;
    coeff[14]         = 7.0206;
    coeff[15]         = 3.4692;
  } else if( ex < 0.35 && ex >= 0.25 ) {
    weights[0][0][0]  = 0.001;
    weights[1][0][0]  = 1.0135;
    weights[0][1][0]  = 0.0621;
    weights[1][1][0]  = 1.1003;
    weights[0][0][1]  = 1.7198;
    weights[1][0][1]  = 0.0918;
    weights[0][1][1]  = 0.6143;
    weights[1][1][1]  = 0.4659;

    coeff[0]          = 43.775;
    coeff[1]          = -0.1091;
    coeff[2]          = -83.1088;
    coeff[3]          = 13.0161;
    coeff[4]          = 60.9008;
    coeff[5]          = 0.592;
    coeff[6]          = -43.4857;
    coeff[7]          = 4.5869;
    coeff[8]          = -45.3315;
    coeff[9]          = 8.0252;
    coeff[10]         = 34.3571;
    coeff[11]         = 1.199;
    coeff[12]         = -41.4422;
    coeff[13]         = 8.399;
    coeff[14]         = -2.8772;
    coeff[15]         = 3.5323;
  } else if( ex < 0.45 && ex >= 0.35 ) {
    weights[0][0][0]  = 0.001;
    weights[1][0][0]  = 0.9782;
    weights[0][1][0]  = 0.082;
    weights[1][1][0]  = 1.1275;
    weights[0][0][1]  = 1.7122;
    weights[1][0][1]  = 0.001;
    weights[0][1][1]  = 0.4639;
    weights[1][1][1]  = 0.509;

    coeff[0]          = 41.6858;
    coeff[1]          = -0.7344;
    coeff[2]          = -164.2252;
    coeff[3]          = 14.9961;
    coeff[4]          = 103.0301;
    coeff[5]          = -0.4199;
    coeff[6]          = -41.1157;
    coeff[7]          = 3.8266;
    coeff[8]          = -73.0432;
    coeff[9]          = 8.5857;
    coeff[10]         = 38.0868;
    coeff[11]         = 0.3937;
    coeff[12]         = -43.2133;
    coeff[13]         = 8.6747;
    coeff[14]         = 5.6362;
    coeff[15]         = 3.3287;
  } else if( ex < 0.55 && ex >= 0.45 ) {
    weights[0][0][0]  = 0.2073;
    weights[1][0][0]  = 0.912;
    weights[0][1][0]  = 0.1186;
    weights[1][1][0]  = 1.081;
    weights[0][0][1]  = 1.6984;
    weights[1][0][1]  = 0.001;
    weights[0][1][1]  = 0.1872;
    weights[1][1][1]  = 0.6016;

    coeff[0]          = 20.0539;
    coeff[1]          = -0.4354;
    coeff[2]          = -81.6068;
    coeff[3]          = 12.8573;
    coeff[4]          = 45.9948;
    coeff[5]          = 1.1528;
    coeff[6]          = -23.07;
    coeff[7]          = 2.6719;
    coeff[8]          = -27.8961;
    coeff[9]          = 7.1927;
    coeff[10]         = 31.4788;
    coeff[11]         = -0.0434;
    coeff[12]         = -25.1661;
    coeff[13]         = 8.245;
    coeff[14]         = -45.2178;
    coeff[15]         = 4.8476;
  } else if( ex < 0.65 && ex >= 0.55 ) {
    weights[0][0][0]  = 0.3112;
    weights[1][0][0]  = 0.8339;
    weights[0][1][0]  = 0.1616;
    weights[1][1][0]  = 1.0117;
    weights[0][0][1]  = 1.6821;
    weights[1][0][1]  = 0.0001;
    weights[0][1][1]  = 0.0001;
    weights[1][1][1]  = 0.7123;

    coeff[0]          = 8.0848;
    coeff[1]          = -0.1968;
    coeff[2]          = -79.9715;
    coeff[3]          = 12.7318;
    coeff[4]          = 35.7155;
    coeff[5]          = 1.68;
    coeff[6]          = -13.0365;
    coeff[7]          = 1.8101;
    coeff[8]          = -13.2235;
    coeff[9]          = 6.3697;
    coeff[10]         = 25.4548;
    coeff[11]         = -0.3947;
    coeff[12]         = -10.4478;
    coeff[13]         = 7.657;
    coeff[14]         = -75.9179;
    coeff[15]         = 6.1791;
  } else if( ex < 0.75 && ex >= 0.65 ) {
    weights[0][0][0]  = 0.1219;
    weights[1][0][0]  = 0.001;
    weights[0][1][0]  = 0.5084;
    weights[1][1][0]  = 0.2999;
    weights[0][0][1]  = 1.2197;
    weights[1][0][1]  = 0.001;
    weights[0][1][1]  = 0.001;
    weights[1][1][1]  = 1.3635;

    coeff[0]          = 1.9975;
    coeff[1]          = 0.418;
    coeff[2]          = -76.6932;
    coeff[3]          = 11.3479;
    coeff[4]          = 40.7406;
    coeff[5]          = 1.9511;
    coeff[6]          = -2.7761;
    coeff[7]          = 0.5987;
    coeff[8]          = 0.0;
    coeff[9]          = 0.0;
    coeff[10]         = 0.0;
    coeff[11]         = 0.0;
    coeff[12]         = 41.317;
    coeff[13]         = 2.1874;
    coeff[14]         = -88.8095;
    coeff[15]         = 11.0003;
  } else if( ex < 0.85 && ex >= 0.75 ) {
    weights[0][0][0]  = 0.0462 ;
    weights[1][0][0]  = 0.001;
    weights[0][1][0]  = 0.4157;
    weights[1][1][0]  = 0.1585;
    weights[0][0][1]  = 1.3005;
    weights[1][0][1]  = 0.001;
    weights[0][1][1]  = 0.001;
    weights[1][1][1]  = 1.4986;

    coeff[0]          = 5.1672;
    coeff[1]          = 0.2129;
    coeff[2]          = -46.506;
    coeff[3]          = 11.7213;
    coeff[4]          = -5.8873;
    coeff[5]          = 1.4279;
    coeff[6]          = -8.2448;
    coeff[7]          = 0.3455;
    coeff[8]          = 15.0254;
    coeff[9]          = -0.283;
    coeff[10]         = 0.0;
    coeff[11]         = 0.0;
    coeff[12]         = 58.975;
    coeff[13]         = 0.8131;
    coeff[14]         = -108.6828;
    coeff[15]         = 12.4362;
  } else if( ex < 0.95 && ex >= 0.85 ) {
    weights[0][0][0]  = 0.001;
    weights[1][0][0]  = 0.001;
    weights[0][1][0]  = 0.1342;
    weights[1][1][0]  = 0.1935;
    weights[0][0][1]  = 1.5755;
    weights[1][0][1]  = 0.001;
    weights[0][1][1]  = 0.001;
    weights[1][1][1]  = 1.5297;

    coeff[0]          = -0.8151;
    coeff[1]          = 0.1621;
    coeff[2]          = -61.9333;
    coeff[3]          = 12.5014;
    coeff[4]          = 0.0358;
    coeff[5]          = -0.0006;
    coeff[6]          = 0.0;
    coeff[7]          = 0.0;
    coeff[8]          = 22.0291;
    coeff[9]          = -0.4022;
    coeff[10]         = 0.0;
    coeff[11]         = 0.0;
    coeff[12]         = 56.0043;
    coeff[13]         = 0.7978;
    coeff[14]         = -116.9175;
    coeff[15]         = 13.0244;
  } else if( ex < 0.01 ) {
    weights[0][0][0]  = 0.8867;
    weights[1][0][0]  = 1.0440;
    weights[0][1][0]  = 0.0423;
    weights[1][1][0]  = 0.8110;
    weights[0][0][1]  = 1.7275;
    weights[1][0][1]  = 0.5615;
    weights[0][1][1]  = 0.8323;
    weights[1][1][1]  = 0.4641;

    coeff[0]          = -27.5089;
    coeff[1]          = 7.4177;
    coeff[2]          = -82.8803;
    coeff[3]          = 13.1952;
    coeff[4]          = 72.0312;
    coeff[5]          = 0.5298;
    coeff[6]          = -34.1779;
    coeff[7]          = 6.0293;
    coeff[8]          = -52.2607;
    coeff[9]          = 8.1754;
    coeff[10]         = -1.6270;
    coeff[11]         = 4.6858;
    coeff[12]         = -27.7770;
    coeff[13]         = 6.2852;
    coeff[14]         = 14.6295;
    coeff[15]         = 3.8839;
  }

  //! The following is done in inimesh in the CPU code.
  double facex = pow( fac,ex );
  for( i = 0; i < 2; i++ )
    for( j = 0; j < 2; j++ )
      for( k = 0; k < 2; k++ )
        weights[i][j][k] = weights[i][j][k] / facex;

  return;
}

// TODO(Josh): rewrite this in a sane way
void odc::data::Mesh::set_boundaries(
#ifdef YASK
                                      Grid_XYZ* gd1, Grid_XYZ* gmu, Grid_XYZ* glam,
#endif
                                      Grid3D d1, Grid3D mu, Grid3D lam,
                                      Grid3D qp, Grid3D qs, bool anelastic,
                                      int_pt bdry_width, int_pt nx, int_pt ny, int_pt nz
                                    ) {
  int_pt i, j, k, h = bdry_width;
  for( j = 0; j < ny; j++ ) {
    for( k = 0; k < nz; k++ ) {
#ifdef YASK
      glam->writeElem( glam->readElem( h, h+j, h+k, 0 ), h-1, h+j, h+k, 0 );
      glam->writeElem( glam->readElem( h+nx-1, h+j, h+k, 0 ), h+nx, h+j, h+k, 0 );

      gmu->writeElem( gmu->readElem(h, h+j, h+k, 0), h-1, h+j, h+k, 0 );
      gmu->writeElem(gmu->readElem( h+nx-1, h+j, h+k, 0), h+nx, h+j, h+k, 0 );

      gd1->writeElem(gd1->readElem( h, h+j, h+k, 0), h-1, h+j, h+k, 0 );
      gd1->writeElem(gd1->readElem( h+nx-1, h+j, h+k, 0), h+nx, h+j, h+k, 0 );
#else
      lam[h-1][h+j][h+k]  = lam[h][h+j][h+k];
      lam[h+nx][h+j][h+k] = lam[h+nx-1][h+j][h+k];
      mu[h-1][h+j][h+k]   = mu[h][h+j][h+k];
      mu[h+nx][h+j][h+k]  = mu[h+nx-1][h+j][h+k];
      d1[h-1][h+j][h+k]   = d1[h][h+j][h+k];
      d1[h+nx][h+j][h+k]  = d1[h+nx-1][h+j][h+k];
#endif
    }
  }

  for( i = 0; i < nx; i++ ) {
    for( k = 0; k < nz; k++ ) {
#ifdef YASK
      glam->writeElem( glam->readElem( h+i, h, h+k, 0 ), h+i, h-1, h+k, 0 );
      glam->writeElem( glam->readElem( h+i-1, h+ny-1, h+k, 0 ), h+i, h+ny, h+k, 0 );
      
      gmu->writeElem( gmu->readElem( h+i, h, h+k, 0 ), h+i, h-1, h+k, 0 );
      gmu->writeElem( gmu->readElem( h+i, h+ny-1, h+k, 0 ), h+i, h+ny, h+k, 0 );

      gd1->writeElem( gd1->readElem( h+i, h, h+k, 0), h+i, h-1, h+k, 0 );
      gd1->writeElem( gd1->readElem( h+i, h+ny-1, h+k, 0), h+i, h+ny, h+k, 0 );
#else
      lam[h+i][h-1][h+k]  = lam[h+i][h][h+k];
      lam[h+i][h+ny][h+k] = lam[h+i][h+ny-1][h+k];
      mu[h+i][h-1][h+k]   = mu[h+i][h][h+k];
      mu[h+i][h+ny][h+k]  = mu[h+i][h+ny-1][h+k];
      d1[h+i][h-1][h+k]   = d1[h+i][h][h+k];
      d1[h+i][h+ny][h+k]  = d1[h+i][h+ny-1][h+k];
#endif
    }
  }

  for( i = 0; i < nx; i++ ) {
    for( j = 0; j < ny; j++ ) {
#ifdef YASK
      glam->writeElem( glam->readElem( h+i, h+j, h, 0 ), h+i, h+j, h-1, 0 );
      gmu->writeElem( gmu->readElem( h+i, h+j, h, 0 ), h+i, h+j, h-1, 0 );
      gd1->writeElem( gd1->readElem( h+i, h+j, h, 0 ), h+i, h+j, h-1, 0 );
#else
      lam[h+i][h+j][h-1]  = lam[h+i][h+j][h];
      mu[h+i][h+j][h-1]   = mu[h+i][h+j][h];
      d1[h+i][h+j][h-1]   = d1[h+i][h+j][h];
#endif
    }
  }

  //! 12 border lines
  for( i = 0; i < nx; i++ ) {
#ifdef YASK
    glam->writeElem( glam->readElem( h+i, h, h, 0 ), h+i, h-1, h-1, 0 );
    gmu->writeElem( gmu->readElem( h+i, h, h, 0 ), h+i, h-1, h-1, 0 );
    gd1->writeElem( gd1->readElem( h+i, h, h, 0 ), h+i, h-1, h-1, 0 );
    glam->writeElem( glam->readElem( h+i, h+ny-1, h, 0 ), h+i, h+ny, h-1, 0 );
    gmu->writeElem( gmu->readElem( h+i, h+ny-1, h, 0 ), h+i, h+ny, h-1, 0 );
    gd1->writeElem( gd1->readElem( h+i, h+ny-1, h, 0 ), h+i, h+ny, h-1, 0 );
    glam->writeElem( glam->readElem( h+i, h, h+nz-1, 0 ), h+i, h-1, h+nz, 0 );
    gmu->writeElem( gmu->readElem( h+i, h, h+nz-1, 0 ), h+i, h-1, h+nz, 0 );
    gd1->writeElem( gd1->readElem( h+i, h, h+nz-1, 0 ), h+i, h-1, h+nz, 0 );
    glam->writeElem( glam->readElem( h+i, h+ny-1, h+nz-1, 0 ), h+i, h+ny, h+nz, 0 );
    gmu->writeElem( gmu->readElem( h+i, h+ny-1, h+nz-1, 0 ), h+i, h+ny, h+nz, 0 );
    gd1->writeElem( gd1->readElem( h+i, h+ny-1, h+nz-1, 0 ), h+i, h+ny, h+nz, 0 );
#else
    lam[h+i][h-1][h-1]    = lam[h+i][h][h];
    mu[h+i][h-1][h-1]     = mu[h+i][h][h];
    d1[h+i][h-1][h-1]     = d1[h+i][h][h];
    lam[h+i][h+ny][h-1]   = lam[h+i][h+ny-1][h];
    mu[h+i][h+ny][h-1]    = mu[h+i][h+ny-1][h];
    d1[h+i][h+ny][h-1]    = d1[h+i][h+ny-1][h];
    lam[h+i][h-1][h+nz]   = lam[h+i][h][h+nz-1];
    mu[h+i][h-1][h+nz]    = mu[h+i][h][h+nz-1];
    d1[h+i][h-1][h+nz]    = d1[h+i][h][h+nz-1];
    lam[h+i][h+ny][h+nz]  = lam[h+i][h+ny-1][h+nz-1];
    mu[h+i][h+ny][h+nz]   = mu[h+i][h+ny-1][h+nz-1];
    d1[h+i][h+ny][h+nz]   = d1[h+i][h+ny-1][h+nz-1];
#endif
  }

  for( j = 0; j < ny; j++ ) {
#ifdef YASK
    glam->writeElem( glam->readElem( h, h+j, h, 0 ), h-1, h+j, h-1, 0 );
    gmu->writeElem( gmu->readElem( h, h+j, h, 0 ), h-1, h+j, h-1, 0 );
    gd1->writeElem( gd1->readElem( h, h+j, h, 0 ), h-1, h+j, h-1, 0 );
    glam->writeElem( glam->readElem( h+nx-1, h+j, h, 0 ), h+nx, h+j, h-1, 0 );
    gmu->writeElem( gmu->readElem( h+nx-1, h+j, h, 0 ), h+nx, h+j, h-1, 0 );
    gd1->writeElem( gd1->readElem( h+nx-1, h+j, h, 0 ), h+nx, h+j, h-1, 0 );
    glam->writeElem( glam->readElem( h, h+j, h+nz-1, 0 ), h-1, h+j, h+nz, 0 );
    gmu->writeElem( gmu->readElem( h, h+j, h+nz-1, 0 ), h-1, h+j, h+nz, 0 );
    gd1->writeElem( gd1->readElem( h, h+j, h+nz-1, 0 ), h-1, h+j, h+nz, 0 );
    glam->writeElem( glam->readElem( h+nx-1, h+j, h+nz-1, 0 ), h+nx, h+j, h+nz, 0 );
    gmu->writeElem( gmu->readElem( h+nx-1, h+j, h+nz-1, 0 ), h+nx, h+j, h+nz, 0 );
    gd1->writeElem( gd1->readElem( h+nx-1, h+j, h+nz-1, 0 ), h+nx, h+j, h+nz, 0 );
#else
    lam[h-1][h+j][h-1]    = lam[h][h+j][h];
    mu[h-1][h+j][h-1]     = mu[h][h+j][h];
    d1[h-1][h+j][h-1]     = d1[h][h+j][h];
    lam[h+nx][h+j][h-1]   = lam[h+nx-1][h+j][h];
    mu[h+nx][h+j][h-1]    = mu[h+nx-1][h+j][h];
    d1[h+nx][h+j][h-1]    = d1[h+nx-1][h+j][h];
    lam[h-1][h+j][h+nz]   = lam[h][h+j][h+nz-1];
    mu[h-1][h+j][h+nz]    = mu[h][h+j][h+nz-1];
    d1[h-1][h+j][h+nz]    = d1[h][h+j][h+nz-1];
    lam[h+nx][h+j][h+nz]  = lam[h+nx-1][h+j][h+nz-1];
    mu[h+nx][h+j][h+nz]   = mu[h+nx-1][h+j][h+nz-1];
    d1[h+nx][h+j][h+nz]   = d1[h+nx-1][h+j][h+nz-1];
#endif
  }

  for( k = 0; k < nz; k++ ) {
#ifdef YASK
    glam->writeElem( glam->readElem(h, h, h+k, 0), h-1, h-1, h+k, 0 );
    gmu->writeElem( gmu->readElem(h, h, h+k, 0), h-1, h-1, h+k, 0 );
    gd1->writeElem( gd1->readElem(h, h, h+k, 0), h-1, h-1, h+k, 0 );
    glam->writeElem( glam->readElem(h+nx-1, h, h+k, 0), h+nx, h-1, h+k, 0 );
    gmu->writeElem( gmu->readElem(h+nx-1, h, h+k, 0), h+nx, h-1, h+k, 0 );
    gd1->writeElem( gd1->readElem(h+nx-1, h, h+k, 0), h+nx, h-1, h+k, 0 );
    glam->writeElem( glam->readElem(h, h+ny-1, h+k, 0), h-1, h+ny, h+k, 0 );
    gmu->writeElem( gmu->readElem(h, h+ny-1, h+k, 0), h-1, h+ny, h+k, 0 );
    gd1->writeElem( gd1->readElem(h, h+ny-1, h+k, 0), h-1, h+ny, h+k, 0 );
    glam->writeElem( glam->readElem(h+nx-1, h+ny-1, h+k, 0), h+nx, h+ny, h+k, 0 );
    gmu->writeElem( gmu->readElem(h+nx-1, h+ny-1, h+k, 0), h+nx, h+ny, h+k, 0 );
    gd1->writeElem( gd1->readElem(h+nx-1, h+ny-1, h+k, 0), h+nx, h+ny, h+k, 0 );
#else
    lam[h-1][h-1][h+k]    = lam[h][h][h+k];
    mu[h-1][h-1][h+k]     = mu[h][h][h+k];
    d1[h-1][h-1][h+k]     = d1[h][h][h+k];
    lam[h+nx][h-1][h+k]   = lam[h+nx-1][h][h+k];
    mu[h+nx][h-1][h+k]    = mu[h+nx-1][h][h+k];
    d1[h+nx][h-1][h+k]    = d1[h+nx-1][h][h+k];
    lam[h-1][h+ny][h+k]   = lam[h][h+ny-1][h+k];
    mu[h-1][h+ny][h+k]    = mu[h][h+ny-1][h+k];
    d1[h-1][h+ny][h+k]    = d1[h][h+ny-1][h+k];
    lam[h+nx][h+ny][h+k]  = lam[h+nx-1][h+ny-1][h+k];
    mu[h+nx][h+ny][h+k]   = mu[h+nx-1][h+ny-1][h+k];
    d1[h+nx][h+ny][h+k]   = d1[h+nx-1][h+ny-1][h+k];
#endif
  }

  //! 8 Corners
#ifdef YASK
  glam->writeElem( glam->readElem( h, h, h, 0 ), h-1, h-1, h-1, 0 );
  gmu->writeElem( gmu->readElem( h, h, h, 0 ), h-1, h-1, h-1, 0 );
  gd1->writeElem( gd1->readElem( h, h, h, 0 ), h-1, h-1, h-1, 0 );

  glam->writeElem( glam->readElem( h+nx-1, h, h, 0 ), h+nx, h-1, h-1, 0 );
  gmu->writeElem( gmu->readElem( h+nx-1, h, h, 0 ), h+nx, h-1, h-1, 0 );
  gd1->writeElem( gd1->readElem( h+nx-1, h, h, 0 ), h+nx, h-1, h-1, 0 );

  glam->writeElem( glam->readElem( h, h+ny-1, h, 0 ), h-1, h+ny, h-1, 0 );
  gmu->writeElem( gmu->readElem( h, h+ny-1, h, 0 ), h-1, h+ny, h-1, 0 );
  gd1->writeElem( gd1->readElem( h, h+ny-1, h, 0 ), h-1, h+ny, h-1, 0 );

  glam->writeElem( glam->readElem( h, h, h+nz-1, 0 ), h-1, h-1, h+nz, 0 );
  gmu->writeElem( gmu->readElem( h, h, h+nz-1, 0 ), h-1, h-1, h+nz, 0 );
  gd1->writeElem( gd1->readElem( h, h, h+nz-1, 0 ), h-1, h-1, h+nz, 0 );

  glam->writeElem( glam->readElem( h+nx-1, h, h+nz-1, 0 ), h+nx, h-1, h+nz, 0 );
  gmu->writeElem( gmu->readElem( h+nx-1, h, h+nz-1, 0 ), h+nx, h-1, h+nz, 0 );
  gd1->writeElem( gd1->readElem( h+nx-1, h, h+nz-1, 0 ), h+nx, h-1, h+nz, 0 );

  glam->writeElem( glam->readElem( h+nx-1, h+ny-1, h, 0 ), h+nx, h+ny, h-1, 0 );
  gmu->writeElem( gmu->readElem( h+nx-1, h+ny-1, h, 0 ), h+nx, h+ny, h-1, 0 );
  gd1->writeElem( gd1->readElem( h+nx-1, h+ny-1, h, 0 ), h+nx, h+ny, h-1, 0 );

  glam->writeElem( glam->readElem( h,h+ny-1, h+nz-1, 0 ), h-1, h+ny, h+nz, 0 );
  gmu->writeElem( gmu->readElem( h, h+ny-1, h+nz-1, 0 ), h-1, h+ny, h+nz, 0 );
  gd1->writeElem( gd1->readElem( h, h+ny-1, h+nz-1, 0 ), h-1, h+ny, h+nz, 0 );

  glam->writeElem( glam->readElem( h+nx-1, h+ny-1, h+nz-1, 0 ), h+nx, h+ny, h+nz, 0 );
  gmu->writeElem( gmu->readElem( h+nx-1, h+ny-1, h+nz-1, 0 ), h+nx, h+ny, h+nz, 0 );
  gd1->writeElem( gd1->readElem( h+nx-1, h+ny-1, h+nz-1, 0 ), h+nx, h+ny, h+nz, 0 );
#else
  lam[h-1][h-1][h-1]      = lam[h][h][h];
  mu[h-1][h-1][h-1]       = mu[h][h][h];
  d1[h-1][h-1][h-1]       = d1[h][h][h];
  lam[h+nx][h-1][h-1]     = lam[h+nx-1][h][h];
  mu[h+nx][h-1][h-1]      = mu[h+nx-1][h][h]; 
  d1[h+nx][h-1][h-1]      = d1[h+nx-1][h][h];
  lam[h-1][h+ny][h-1]     = lam[h][h+ny-1][h];
  mu[h-1][h+ny][h-1]      = mu[h][h+ny-1][h];
  d1[h-1][h+ny][h-1]      = d1[h][h+ny-1][h];
  lam[h-1][h-1][h+nz]     = lam[h][h][h+nz-1]; 
  mu[h-1][h-1][h+nz]      = mu[h][h][h+nz-1];
  d1[h-1][h-1][h+nz]      = d1[h][h][h+nz-1];
  lam[h+nx][h-1][h+nz]    = lam[h+nx-1][h][h+nz-1];
  mu[h+nx][h-1][h+nz]     = mu[h+nx-1][h][h+nz-1];
  d1[h+nx][h-1][h+nz]     = d1[h+nx-1][h][h+nz-1];
  lam[h+nx][h+ny][h-1]    = lam[h+nx-1][h+ny-1][h];
  mu[h+nx][h+ny][h-1]     = mu[h+nx-1][h+ny-1][h];
  d1[h+nx][h+ny][h-1]     = d1[h+nx-1][h+ny-1][h];
  lam[h-1][h+ny][h+nz]    = lam[h][h+ny-1][h+nz-1];
  mu[h-1][h+ny][h+nz]     = mu[h][h+ny-1][h+nz-1];
  d1[h-1][h+ny][h+nz]     = d1[h][h+ny-1][h+nz-1];
  lam[h+nx][h+ny][h+nz]   = lam[h+nx-1][h+ny-1][h+nz-1];
  mu[h+nx][h+ny][h+nz]    = mu[h+nx-1][h+ny-1][h+nz-1];
  d1[h+nx][h+ny][h+nz]    = d1[h+nx-1][h+ny-1][h+nz-1];
#endif

  for( i = 0; i < nx; i++ ) {
    for( j = 0; j < ny; j++ ) {
      k = nz;

#ifdef YASK
      gd1->writeElem( gd1->readElem( h+i, h+j, h+k-1, 0 ), h+i, h+j, h+k, 0 );
      gmu->writeElem( gmu->readElem( h+i, h+j, h+k-1, 0 ), h+i, h+j, h+k, 0 );
      glam->writeElem( glam->readElem( h+i, h+j, h+k-1, 0 ), h+i, h+j, h+k, 0 );
#else
      d1[h+i][h+j][h+k]   = d1[h+i][h+j][h+k-1];
      mu[h+i][h+j][h+k]   = mu[h+i][h+j][h+k-1];
      lam[h+i][h+j][h+k]  = lam[h+i][h+j][h+k-1];
#endif

      if( anelastic ) {
        qp[h+i][h+j][h+k] = qp[h+i][h+j][h+k-1];
        qs[h+i][h+j][h+k] = qs[h+i][h+j][h+k-1];
      }
    }
  }
}

void odc::data::Mesh::finalize() {
#ifndef YASK
  odc::data::Delloc3D( m_density, odc::constants::boundary );
  odc::data::Delloc3D( m_lam, odc::constants::boundary );
  odc::data::Delloc3D( m_mu, odc::constants::boundary );
  odc::data::Delloc3D( m_lam_mu, odc::constants::boundary );
#endif

  if( m_usingAnelastic ) {
#ifndef YASK
    odc::data::Delloc3D( m_qp, odc::constants::boundary );
    odc::data::Delloc3D( m_qs, odc::constants::boundary );
    odc::data::Delloc3D( m_tau1, odc::constants::boundary );
    odc::data::Delloc3D( m_tau2, odc::constants::boundary );
    odc::data::Delloc3D( m_weights, odc::constants::boundary );
    odc::data::Delloc3Dww( m_weight_index, odc::constants::boundary );
#endif

    Delloc1D( m_coeff );
  }
}
