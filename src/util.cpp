/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 
 @section DESCRIPTION
 Misc utility functions.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "util.hpp"

double wall_time() {
  struct timeval t;
  if( gettimeofday(&t, NULL) ) {
    return 0;
  }

  return (double) t.tv_sec + 0.000001 * (double) t.tv_usec;
}

#ifdef YASK
void applyYASKStencil( STENCIL_CONTEXT context, STENCIL_EQUATIONS *stencils, int stencil_index, int_pt t,
                       int_pt start_x, int_pt start_y, int_pt start_z,
                       int_pt numX, int_pt numY, int_pt numZ ) {
  const idx_t step_rt = 1;
  const idx_t step_rn = 1;
  const idx_t step_rx = 16;
  const idx_t step_ry = 16;
  const idx_t step_rz = 128;

  int_pt end_x = start_x + numX;
  int_pt end_y = start_y + numY;
  int_pt end_z = start_z + numZ;

  start_x = (VLEN_X * CLEN_X) * (int_pt) ((start_x + VLEN_X * CLEN_X - 1) / (VLEN_X * CLEN_X));
  start_y = (VLEN_Y * CLEN_Y) * (int_pt) ((start_y + VLEN_Y * CLEN_Y - 1) / (VLEN_Y * CLEN_Y));
  start_z = (VLEN_Z * CLEN_Z) * (int_pt) ((start_z + VLEN_Z * CLEN_Z - 1) / (VLEN_Z * CLEN_Z));

  end_x   = ROUND_UP(end_x, CPTS_X);
  end_y   = ROUND_UP(end_y, CPTS_Y);
  end_z   = ROUND_UP(end_z, CPTS_Z);

  StencilBase *stencil = stencils->stencils[stencil_index];

  idx_t begin_rn  = 0;
  idx_t end_rn    = CPTS_N;
  idx_t begin_rx  = start_x;
  idx_t end_rx    = end_x;
  idx_t begin_ry  = start_y;
  idx_t end_ry    = end_y;
  idx_t begin_rz  = start_z;
  idx_t end_rz    = end_z;
  idx_t rt        = t;

#include "yask/stencil_region_loops.hpp"
}

void applyYASKStencilBdry( STENCIL_CONTEXT context, STENCIL_EQUATIONS *stencils, int stencil_index, int_pt t,
                           int_pt start_x, int_pt start_y, int_pt start_z,
                           int_pt numX, int_pt numY, int_pt numZ, bool xBdryMin, bool xBdryMax,
                           bool yBdryMin, bool yBdryMax, bool zBdryMin, bool zBdryMax,
                           PatchDecomp& patch_decomp ) {
  int_pt h = patch_decomp.m_patches[0].bdry_width;

  const idx_t step_rt = 1;
  const idx_t step_rn = 1;
  const idx_t step_rx = 4;
  const idx_t step_ry = 4;
  const idx_t step_rz = 8;

  bool doXMin, doXMax, doYMin, doYMax, doZMin, doZMax;

  //! PPP: assumes that 16 divides end_z - start_z after rounding

  int_pt end_x = start_x + numX;
  int_pt end_y = start_y + numY;
  int_pt end_z = start_z + numZ;

  start_x = (VLEN_X * CLEN_X) * (int_pt) ((start_x + VLEN_X * CLEN_X - 1) / (VLEN_X * CLEN_X));
  start_y = (VLEN_Y * CLEN_Y) * (int_pt) ((start_y + VLEN_Y * CLEN_Y - 1) / (VLEN_Y * CLEN_Y));
  start_z = (VLEN_Z * CLEN_Z) * (int_pt) ((start_z + VLEN_Z * CLEN_Z - 1) / (VLEN_Z * CLEN_Z));

  end_x   = ROUND_UP(end_x, CPTS_X);
  end_y   = ROUND_UP(end_y, CPTS_Y);
  end_z   = ROUND_UP(end_z, CPTS_Z);

  StencilBase *stencil = stencils->stencils[stencil_index];

  idx_t begin_rn  = 0;
  idx_t end_rn    = CPTS_N;
  idx_t rt        = t;

  for( idx_t begin_rx = start_x; begin_rx < end_x; begin_rx += step_rx ) {
    idx_t end_rx = begin_rx + step_rx;

    doXMin = (begin_rx == start_x && xBdryMin);
    doXMax = (end_rx == end_x     && xBdryMax);

    for( idx_t begin_ry = start_y; begin_ry < end_y; begin_ry += step_ry ) {
      idx_t end_ry = begin_ry + step_ry;

      doYMin = (begin_ry == start_y && yBdryMin);
      doYMax = (end_ry == end_y     && yBdryMax);

      for( idx_t begin_rz = start_z; begin_rz < end_z; begin_rz += step_rz) {
        idx_t end_rz = begin_rz + step_rz;

        doZMin = (begin_rz == start_z && zBdryMin);
        doZMax = (end_rz == end_z     && zBdryMax);

        if( doXMin && stencil_index == 0 ) {
          patch_decomp.copyStressBoundaryFromBuffer( odc::parallel::Mpi::m_buffRecv[0][1][1], -1, 0, 0,
                                                     start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        if( doXMax && stencil_index == 0 ) {
          patch_decomp.copyStressBoundaryFromBuffer( odc::parallel::Mpi::m_buffRecv[2][1][1], 1, 0, 0,
                                                     start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        if( doYMin && stencil_index == 0 ) {
          patch_decomp.copyStressBoundaryFromBuffer( odc::parallel::Mpi::m_buffRecv[1][0][1], 0, -1, 0,
                                                     start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        if( doYMax && stencil_index == 0 ) {
          patch_decomp.copyStressBoundaryFromBuffer( odc::parallel::Mpi::m_buffRecv[1][2][1], 0, 1, 0,
                                                     start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        //! PPP do z

        /*if(doXMin && stencil_index == 1) {
          patch_decomp.copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[0][1][1], -1, 0, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
        if(doXMax && stencil_index == 1) {
          patch_decomp.copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[2][1][1], 1, 0, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
        if(doYMin && stencil_index == 1) {
          patch_decomp.copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[1][0][1], 0, -1, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
        if(doYMax && stencil_index == 1) {
          patch_decomp.copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[1][2][1], 0, 1, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
          }*/

#include "yask/stencil_region_loops.hpp"

        if( doXMin && stencil_index == 0 ) {
          patch_decomp.copyVelBoundaryToBuffer( odc::parallel::Mpi::m_buffSend[0][1][1], -1, 0, 0,
                                                start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        if( doXMax && stencil_index == 0 ) {
          patch_decomp.copyVelBoundaryToBuffer( odc::parallel::Mpi::m_buffSend[2][1][1], 1, 0, 0,
                                                start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        if( doYMin && stencil_index == 0 ) {
          patch_decomp.copyVelBoundaryToBuffer( odc::parallel::Mpi::m_buffSend[1][0][1], 0, -1, 0,
                                                start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        if( doYMax && stencil_index == 0 ) {
          patch_decomp.copyVelBoundaryToBuffer( odc::parallel::Mpi::m_buffSend[1][2][1], 0, 1, 0,
                                                start_x - h, start_y - h, start_z - h, end_x - h, end_y - h, end_z - h, t + 1 );
        }
        //! PPP do z

        if( doXMin && stencil_index == 1 ) {
          ;//patch_decomp.copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[0][1][1], -1, 0, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
        if( doXMax && stencil_index == 1 ) {
          ;//patch_decomp.copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[2][1][1], 1, 0, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
        if( doYMin && stencil_index == 1 ) {
          ;//patch_decomp.copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[1][0][1], 0, -1, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
        if( doYMax && stencil_index == 1 ) {
          ;//patch_decomp.copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[1][2][1], 0, 1, 0, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, t+1);
        }
      }
    }
  }
}

#endif
