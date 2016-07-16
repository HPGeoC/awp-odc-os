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

#define N_SCB_X 16
#define N_SCB_Y 32
#define N_SCB_Z 32

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
//    assert( i_nX >= N_SCB_X );
//    assert( i_nY >= N_SCB_Y );
//    assert( i_nZ >= N_SCB_Z );

    // compute constants
    real l_c1 = (9.0  / 8.0  ) * (i_dT / i_dH);
    real l_c2 = (-1.0 / 24.0 ) * (i_dT / i_dH);

    int_pt l_blockEndX = (i_nX / N_SCB_X) * N_SCB_X;
    int_pt l_blockEndY = (i_nY / N_SCB_Y) * N_SCB_Y;
    int_pt l_blockEndZ = (i_nZ / N_SCB_Z) * N_SCB_Z;
    
    /*
     * Iterations over blocks
     */
    // loop over blocks in x-dimension
    for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
      // loop over blocks in y-dimension
      for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
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

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                // derive scalar for sponge layers, TODO: this is a waste of memory bandwidth
                real l_cerjan = i_crjX[l_blockX + l_x] * i_crjY[l_blockY + l_y] * i_crjZ[l_blockZ + l_z];

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
            }
          }

        }
      }
    }


    /*
     * Iterations over blocks
     */
    // loop over blocks in x-dimension
    if(i_nX % N_SCB_X)
      for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
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

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockEndX + l_x] * i_crjY[l_blockY + l_y] * i_crjZ[l_blockZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
      }

    if(i_nY % N_SCB_Y)
      for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < N_SCB_X; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockX + l_x] * i_crjY[l_blockEndY + l_y] * i_crjZ[l_blockZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
      }


    if(i_nZ % N_SCB_Z)
      for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockX + l_x] * i_crjY[l_blockY + l_y] * i_crjZ[l_blockEndZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
      }
    
    if(i_nX % N_SCB_X)
      if(i_nY % N_SCB_Y)
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockEndX + l_x] * i_crjY[l_blockEndY + l_y] * i_crjZ[l_blockZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }

    if(i_nX % N_SCB_X)
      if(i_nZ % N_SCB_Z)
        for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockEndX + l_x] * i_crjY[l_blockY + l_y] * i_crjZ[l_blockEndZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }


    if(i_nY % N_SCB_Y)
      if(i_nZ % N_SCB_Z)
        for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < N_SCB_X; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockX + l_x] * i_crjY[l_blockEndY + l_y] * i_crjZ[l_blockEndZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
    
    if(i_nY % N_SCB_Y)
      if(i_nZ % N_SCB_Z)
        if(i_nX % N_SCB_X) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                // derive densities
                real l_invDensityX = (real) 0.25 * ( i_density[l_pos] + i_density[l_posYm1] + i_density[l_posZm1] + i_density[l_posYm1Zm1] );
                     l_invDensityX = (real) 1 / l_invDensityX;

                real l_invDensityY = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posZm1] + i_density[l_posXp1Zm1] );
                     l_invDensityY = (real) 1 / l_invDensityY;

                real l_invDensityZ = (real) 0.25 * ( i_density[l_pos] + i_density[l_posXp1] + i_density[l_posYm1] + i_density[l_posXp1Ym1] );
                     l_invDensityZ = (real) 1 / l_invDensityZ;

                real l_cerjan = i_crjX[l_blockEndX + l_x] * i_crjY[l_blockEndY + l_y] * i_crjZ[l_blockEndZ + l_z];

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

                io_velX[l_pos] *= l_cerjan;
                io_velY[l_pos] *= l_cerjan;
                io_velZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
    
    
    SCOREP_USER_FUNC_END()
}

void odc::kernels::SCB::update_stress_elastic( real   *i_velX,
                                       real   *i_velY,
                                       real   *i_velZ,
                                       int_pt  i_nX,
                                       int_pt  i_nY,
                                       int_pt  i_nZ,
                                       int_pt  i_strideX,
                                       int_pt  i_strideY,
                                       int_pt  i_strideZ,
                                       real   *io_stressXX,
                                       real   *io_stressXY,
                                       real   *io_stressXZ,
                                       real   *io_stressYY,
                                       real   *io_stressYZ,
                                       real   *io_stressZZ,
                                       real   *i_coeff,
                                       real   *i_crjX,
                                       real   *i_crjY,
                                       real   *i_crjZ,
                                       real   *i_density,
                                       real   *i_lambda,
                                       real   *i_mu,
                                       real   *i_lamMu,
                                       real    i_dT,
                                       real    i_dH) {
    
    real l_c1 = 9.0/8.0;
    real l_c2 = -1.0/24.0;
    

    int_pt l_blockEndX = (i_nX / N_SCB_X) * N_SCB_X;
    int_pt l_blockEndY = (i_nY / N_SCB_Y) * N_SCB_Y;
    int_pt l_blockEndZ = (i_nZ / N_SCB_Z) * N_SCB_Z;
    
    /*
     * Iterations over blocks
     */
    // loop over blocks in x-dimension
    for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
      // loop over blocks in y-dimension
      for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
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

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockX+l_x] * i_crjY[l_blockY+l_y] * i_crjZ[l_blockZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;
                

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
            }
          }

        }
      }
    }

    if(i_nX % N_SCB_X)
      for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
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

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockEndX+l_x] * i_crjY[l_blockY+l_y] * i_crjZ[l_blockZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
      }

    if(i_nY % N_SCB_Y)
      for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < N_SCB_X; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockX+l_x] * i_crjY[l_blockEndY+l_y] * i_crjZ[l_blockZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
      }


    if(i_nZ % N_SCB_Z)
      for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
        // loop over blocks in z-dimension
        for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockX+l_x] * i_crjY[l_blockY+l_y] * i_crjZ[l_blockEndZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
      }
    
    if(i_nX % N_SCB_X)
      if(i_nY % N_SCB_Y)
        for( int_pt l_blockZ = 0; l_blockZ < l_blockEndZ; l_blockZ+=N_SCB_Z ) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockEndX+l_x] * i_crjY[l_blockEndY+l_y] * i_crjZ[l_blockZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }

    if(i_nX % N_SCB_X)
      if(i_nZ % N_SCB_Z)
        for( int_pt l_blockY = 0; l_blockY < l_blockEndY; l_blockY+=N_SCB_Y ) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockEndX+l_x] * i_crjY[l_blockY+l_y] * i_crjZ[l_blockEndZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }


    if(i_nY % N_SCB_Y)
      if(i_nZ % N_SCB_Z)
        for( int_pt l_blockX = 0; l_blockX < l_blockEndX; l_blockX+=N_SCB_X ) {
          // derive block indices
          int_pt l_posBlock  = l_blockX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < N_SCB_X; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockX+l_x] * i_crjY[l_blockEndY+l_y] * i_crjZ[l_blockEndZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
    
    if(i_nY % N_SCB_Y)
      if(i_nZ % N_SCB_Z)
        if(i_nX % N_SCB_X) {
          // derive block indices
          int_pt l_posBlock  = l_blockEndX * i_strideX;
                 l_posBlock += l_blockEndY * i_strideY;
                 l_posBlock += l_blockEndZ * 1;

          // loop over points in x-dimension
          for( int_pt l_x = 0; l_x < i_nX - l_blockEndX; l_x++ ) {
              // loop over points in y-dimension
              for( int_pt l_y = 0; l_y < i_nY - l_blockEndY; l_y++ ) {
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
              for( int_pt l_z = 0; l_z < i_nZ - l_blockEndZ; l_z++ ) {

                int_pt l_posXp1Ym1Zm1 = l_pos + i_strideX - i_strideY - i_strideZ;
                
                // Calculate sponge layer multiplicative factor
                real l_cerjan = i_crjX[l_blockEndX+l_x] * i_crjY[l_blockEndY+l_y] * i_crjZ[l_blockEndZ+l_z];
                
                // Local average of lambda
                real l_lambda = 8.0/(i_lambda[l_pos] + i_lambda[l_posXp1]
                                         + i_lambda[l_posYm1] + i_lambda[l_posXp1Ym1]
                                         + i_lambda[l_posZm1] + i_lambda[l_posXp1Zm1]
                                         + i_lambda[l_posYm1Zm1] + i_lambda[l_posXp1Ym1Zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real l_muDiag = 8.0/(i_mu[l_pos] + i_mu[l_posXp1] + i_mu[l_posYm1] + i_mu[l_posXp1Ym1]
                                          + i_mu[l_posZm1] + i_mu[l_posXp1Zm1] + i_mu[l_posYm1Zm1]
                                          + i_mu[l_posXp1Ym1Zm1]);
                
                // Local average of mu for xy stress component
                real l_muXY = 2.0/(i_mu[l_pos] + i_mu[l_posZm1]);
                
                // Local average of mu for xz stress component
                real l_muXZ = 2.0/(i_mu[l_pos] + i_mu[l_posYm1]);
                
                // Local average of mu for yz stress component
                real l_muYZ = 2.0/(i_mu[l_pos] + i_mu[l_posXp1]);
                
                
                // Estimate strain from velocity
                real l_strainXX = (1.0/i_dH) * (  l_c1 * (i_velX[l_posXp1] - i_velX[l_pos])
                                                + l_c2 * (i_velX[l_posXp2] - i_velX[l_posXm1])
                                               );
                real l_strainYY = (1.0/i_dH) * (  l_c1 * (i_velY[l_pos]     - i_velY[l_posYm1])
                                                + l_c2 * (i_velY[l_posYp1] - i_velY[l_posYm2])
                                               );
                real l_strainZZ = (1.0/i_dH) * (  l_c1 * (i_velZ[l_pos]     - i_velZ[l_posZm1])
                                                + l_c2 * (i_velZ[l_posZp1] - i_velZ[l_posZm2])
                                               );

                
                real l_strainXY = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posYp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posYp2] - i_velX[l_posYm1])
                                      + l_c1 * (i_velY[l_pos]     - i_velY[l_posXm1])
                                      + l_c2 * (i_velY[l_posXp1] - i_velY[l_posXm2]));
                
                real l_strainXZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velX[l_posZp1] - i_velX[l_pos])
                                      + l_c2 * (i_velX[l_posZp2] - i_velX[l_posZm1])
                                      + l_c1 * (i_velZ[l_pos]     - i_velZ[l_posXm1])
                                      + l_c2 * (i_velZ[l_posXp1] - i_velZ[l_posXm2]));
                
                real l_strainYZ = 0.5 * (1.0/i_dH)
                                  * (   l_c1 * (i_velY[l_posZp1] - i_velY[l_pos])
                                      + l_c2 * (i_velY[l_posZp2] - i_velY[l_posZm1])
                                      + l_c1 * (i_velZ[l_posYp1] - i_velZ[l_pos])
                                      + l_c2 * (i_velZ[l_posYp2] - i_velZ[l_posYm1]));
                
                
                // Update tangential stress components

                real l_diagStrainTerm = l_lambda * (l_strainXX + l_strainYY + l_strainZZ) * i_dT;
                
                io_stressXX[l_pos] += 2.0*l_muDiag*l_strainXX*i_dT + l_diagStrainTerm;
                
                io_stressYY[l_pos] += 2.0*l_muDiag*l_strainYY*i_dT + l_diagStrainTerm;
                
                io_stressZZ[l_pos] += 2.0*l_muDiag*l_strainZZ*i_dT + l_diagStrainTerm;
                
                
                // Update shear stress components
                io_stressXY[l_pos] += 2.0 * l_muXY * l_strainXY * i_dT;
                io_stressXZ[l_pos] += 2.0 * l_muXZ * l_strainXZ * i_dT;
                io_stressYZ[l_pos] += 2.0 * l_muYZ * l_strainYZ * i_dT;
                
                // Apply Absorbing Boundary Condition
                io_stressXX[l_pos] *= l_cerjan;
                io_stressYY[l_pos] *= l_cerjan;
                io_stressZZ[l_pos] *= l_cerjan;
                
                io_stressXY[l_pos] *= l_cerjan;
                io_stressXZ[l_pos] *= l_cerjan;
                io_stressYZ[l_pos] *= l_cerjan;

                l_pos++;       l_posXp1++;    l_posXp2++;    l_posXm1++; l_posXm2++;
                l_posYp1++;    l_posYp2++;    l_posYm1++;    l_posYm2++;
                l_posZp1++;    l_posZp2++;    l_posZm1++;    l_posZm2++;
                l_posYm1Zm1++; l_posXp1Ym1++; l_posXp1Zm1++;

              }
            }
          }

        }
    
    

    
} 

