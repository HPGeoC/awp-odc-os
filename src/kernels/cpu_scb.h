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

#ifndef CPU_SCB_H_
#define CPU_SCB_H_

#include "constants.hpp"

namespace odc {
  namespace kernels {
    class SCB;
  }
}

/**
 * Kernels employing spatial cache-blocking.
 **/
class odc::kernels::SCB {
  public:
    /**
     * @param io_velX velocity component in x-direction.
     * @param io_velY velocity component in y-direction.
     * @param io_velZ velocity component in z-direction.
     * @param i_nX number of grid points in x-direction.
     * @param i_nY number of grid points in y-direction.
     * @param i_nZ number of grid points in z-direction.
     * @param i_strideX stride of x-dimension.
     * @param i_strideY stride of y-dimension.
     * @param i_strideZ stride of z-dimension.
     * @param i_stressXX stress component xx.
     * @param i_stressXY stress component xy.
     * @param i_stressXZ stress component xz.
     * @param i_stressYY stress component yy.
     * @param i_stressYZ stress component yz.
     * @param i_stressZZ stress component zz.
     * @param i_crjX cerjan in x-direction.
     * @param i_crjY cerjan in y-direction.
     * @param i_crjZ cerjan in z-direction.
     * @param i_density density.
     * @param i_dT time step.
     * @param i_dH mesh width.
     **/
    static void update_velocity( real   *io_velX,
                                 real   *io_velY,
                                 real   *io_velZ,
                                 int_pt  i_nX,
                                 int_pt  i_nY,
                                 int_pt  i_nZ,
                                 int_pt  i_strideX,
                                 int_pt  i_strideY,
                                 int_pt  i_strideZ,
                                 real   *i_stressXX,
                                 real   *i_stressXY,
                                 real   *i_stressXZ,
                                 real   *i_stressYY,
                                 real   *i_stressYZ,
                                 real   *i_stressZZ,
                                 real   *i_crjX,
                                 real   *i_crjY,
                                 real   *i_crjZ,
                                 real   *i_density,
                                 real    i_dT,
                                 real    i_dH );
   /**
     * @param i_velX velocity component in x-direction.
     * @param i_velY velocity component in y-direction.
     * @param i_velZ velocity component in z-direction.
     * @param i_nX number of grid points in x-direction.
     * @param i_nY number of grid points in y-direction.
     * @param i_nZ number of grid points in z-direction.
     * @param i_strideX stride of x-dimension.
     * @param i_strideY stride of y-dimension.
     * @param i_strideZ stride of z-dimension.
     * @param io_stressXX stress component xx.
     * @param io_stressXY stress component xy.
     * @param io_stressXZ stress component xz.
     * @param io_stressYY stress component yy.
     * @param io_stressYZ stress component yz.
     * @param io_stressZZ stress component zz.
     * @param i_coeff is the coefficient array (TODO: remove? unused here)
     * @param i_crjX cerjan in x-direction.
     * @param i_crjY cerjan in y-direction.
     * @param i_crjZ cerjan in z-direction.
     * @param i_density density.
     * @param i_lambda lambda parameter from mesh.
     * @param i_mu mu parameter from mesh.
     * @param i_lamMu lam_mu parameter from mesh (TODO: remove? unused here)
     * @param i_dT time step.
     * @param i_dH mesh width.
     **/

    static void update_stress_elastic( real   *i_velX,
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
                                       real    i_dH);

  
  
};

#endif
