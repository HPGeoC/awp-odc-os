/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Kernels using spatial cache blocking.

 @section LICENSE
 Copyright (c) 2016, Regents of the University of California
 All rights reserved.

 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu_scb.h"
#include "instrumentation.hpp"
#include <cassert>

#define N_SCB_X 8
#define N_SCB_Y 8
#define N_SCB_Z 512

void odc::kernels::SCB::update_velocity( real   * __restrict io_velX,
                                         real   * __restrict io_velY,
                                         real   * __restrict io_velZ,
                                         int_pt              i_nX,
                                         int_pt              i_nY,
                                         int_pt              i_nZ,
                                         int_pt              i_strideX,
                                         int_pt              i_strideY,
                                         int_pt              i_strideZ,
                                         real   * __restrict i_stressXx,
                                         real   * __restrict i_stressXy,
                                         real   * __restrict i_stressXz,
                                         real   * __restrict i_stressYy,
                                         real   * __restrict i_stressYz,
                                         real   * __restrict i_stressZz,
                                         real   * __restrict i_crjX,
                                         real   * __restrict i_crjY,
                                         real   * __restrict i_crjZ,
                                         real   * __restrict i_density,
                                         real                i_dT,
                                         real                i_dH ) {
    SCOREP_USER_FUNC_BEGIN()

    // assert a valid blocking
    assert( i_nX >= N_SCB_X );
    assert( i_nY >= N_SCB_Y );
    assert( i_nZ >= N_SCB_Z );

    // compute constants
    real l_c1 = (9.0  / 8.0  ) * (i_dT / i_dH);
    real l_c2 = (-1.0 / 24.0 ) * (i_dT / i_dH);

    /*
     * Iterations over blocks
     */
    // loop over blocks in x-dimension
    for( int_pt l_blockX = 0; l_blockX < i_nX; l_blockX+=N_SCB_X ) {
      // loop over blocks in y-dimension
      for( int_pt l_blockY = 0; l_blockY < i_nY; l_blockY+=N_SCB_Y ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < i_nZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          /*
           * iterations inside the blocks
           */
          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < N_SCB_X; l_x++ ) {
            // loop over points in y-dimension
            for( int_pt l_y = 0; l_y < N_SCB_Y; l_y++ ) {
              // derive stencil indices
              int_pt l_pos  = l_posBlock;
                     l_pos += l_x * i_strideX;
                     l_pos += l_y * i_strideY;

              int_pt l_posXp1 = l_pos +     i_strideX;
              int_pt l_posXp2 = l_pos + 2 * i_strideX;
              int_pt l_posXm1 = l_pos -     i_strideX;
              int_pt l_posXm2 = l_pos - 2 * i_strideX;

              int_pt l_posYp1 = l_pos +     i_strideY;
              int_pt l_posYp2 = l_pos + 2 * i_strideY;
              int_pt l_posYm1 = l_pos -     i_strideY;
              int_pt l_posYm2 = l_pos - 2 * i_strideY;

              // this is the leading dimension, no stride!
              int_pt l_posZp1 = l_pos + 1;
              int_pt l_posZp2 = l_pos + 2;
              int_pt l_posZm1 = l_pos - 1;
              int_pt l_posZm2 = l_pos - 2;

              // stride across multiple dimensions
              int_pt l_posYm1Zm1 = l_posYm1 - 1;
              int_pt l_posXp1Ym1 = l_posXp1 - i_strideY;
              int_pt l_posXp1Zm1 = l_posXp1 - 1;

              // loop over points in z-dimension
              for( int_pt l_z = 0; l_z < N_SCB_Z; l_z++ ) {
#if 0
                // derive stencil indices (in inner loop)
                int_pt l_pos  = l_posBlock;
                       l_pos += l_x * i_strideX;
                       l_pos += l_y * i_strideY;
                       l_pos += l_z * 1;

                int_pt l_posXp1 = l_pos +     i_strideX;
                int_pt l_posXp2 = l_pos + 2 * i_strideX;
                int_pt l_posXm1 = l_pos -     i_strideX;
                int_pt l_posXm2 = l_pos - 2 * i_strideX;

                int_pt l_posYp1 = l_pos +     i_strideY;
                int_pt l_posYp2 = l_pos + 2 * i_strideY;
                int_pt l_posYm1 = l_pos -     i_strideY;
                int_pt l_posYm2 = l_pos - 2 * i_strideY;

                // this is the leading dimension, no stride!
                int_pt l_posZp1 = l_pos + 1;
                int_pt l_posZp2 = l_pos + 2;
                int_pt l_posZm1 = l_pos - 1;
                int_pt l_posZm2 = l_pos - 2;

                // stride across multiple dimensions
                int_pt l_posYm1Zm1 = l_posYm1 - 1;
                int_pt l_posXp1Ym1 = l_posXp1 - i_strideY;
                int_pt l_posXp1Zm1 = l_posXp1 - 1;
#endif

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                // derive scalar for sponge layers, TODO: this is a waste of memory bandwidth
                real l_cerjan = i_crjX[l_pos] * i_crjY[l_pos] * i_crjZ[l_pos];

                // update the velocities
                io_velX[l_pos] += l_invDensityX * (   l_c1 * ( i_stressXx[l_pos   ] - i_stressXx[l_posXm1] +
                                                               i_stressXy[l_pos   ] - i_stressXy[l_posYm1] +
                                                               i_stressXz[l_pos   ] - i_stressXz[l_posZm1]
                                                  )
                                                    + l_c2 * ( i_stressXx[l_posXp1] - i_stressXx[l_posXm2] +
                                                               i_stressXy[l_posYp1] - i_stressXy[l_posYm2] +
                                                               i_stressXz[l_posZp1] - i_stressXz[l_posZm2]
                                                             )
                                                  );

                io_velY[l_pos] += l_invDensityY * (   l_c1 * ( i_stressXy[l_posXp1] - i_stressXy[l_pos   ] +
                                                               i_stressYy[l_posYp1] - i_stressYy[l_pos   ] +
                                                               i_stressYz[l_pos   ] - i_stressYz[l_posZm1]
                                                             )
                                                    + l_c2 * ( i_stressXy[l_posXp2] - i_stressXy[l_posXm1] +
                                                               i_stressYy[l_posYp2] - i_stressYy[l_posYm1] +
                                                               i_stressYz[l_posZp1] - i_stressYz[l_posZm2]
                                                             )
                                                 );
                io_velZ[l_pos] += l_invDensityZ * (   l_c1 * ( i_stressXz[l_posXp1] - i_stressXz[l_pos   ] +
                                                               i_stressYz[l_pos   ] - i_stressYz[l_posYm1] +
                                                               i_stressZz[l_posZp1] - i_stressZz[l_pos   ]
                                                             )
                                                    + l_c2 * ( i_stressXz[l_posXp2] - i_stressXz[l_posXm1] +
                                                               i_stressYz[l_posYp1] - i_stressYz[l_posYm2] +
                                                               i_stressZz[l_posZp2] - i_stressZz[l_posZm1] )
                                                  );

                // account for for sponge layers
                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                /* update stencil indices:
                 *
                 *         .                .
                 *    _ _ _.________________._ _ _
                 *         |                |
                 *         |                |
                 * N_SCB_Y |---> z-loop --->|
                 *         |                |
                 *    _ _ _|________________|_ _ _
                 *         .    N_SCB_Z     .
                 *         .                .
                 */
                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
#if 0
              /*
               * after the loop, we are at blockx, blocky, blockz+N_SCB_Z:
               *
               *         .                .
               *    _ _ _.________________._ _ _
               *         |                |
               *         |*               |     reach * by: jump (i_stride_y - N_SCB_Z)
               * N_SCB_Y |---> z-loop --->|#
               *         |                |/\
               *    _ _ _|________________| \
               *         .    N_SCB_Z     .  \_ point after loop
               *         .                .
               */
              int_pt l_jump =(i_strideY-N_SCB_Z);
              l_pos+=l_jump;       l_posXp1+=l_jump;    l_posXp2+=l_jump;    l_posXm1+=l_jump; l_posXm2+=l_jump;
              l_posYp1+=l_jump;    l_posYp2+=l_jump;    l_posYm1+=l_jump;    l_posYm2+=l_jump;
              l_posZp1+=l_jump;    l_posZp2+=l_jump;    l_posZm1+=l_jump;    l_posZm2+=l_jump;
              l_posYm1Zm1+=l_jump; l_posXp1Ym1+=l_jump; l_posXp1Zm1+=l_jump;
#endif
            }
          }

        }
      }
    }

    /*
     * TODO: Remainder handling.
     */
    if( i_nX % N_SCB_X != 0 ) assert( false );
    if( i_nY % N_SCB_Y != 0 ) assert( false );
    if( i_nZ % N_SCB_Z != 0 ) assert( false );

    SCOREP_USER_FUNC_END()
}
