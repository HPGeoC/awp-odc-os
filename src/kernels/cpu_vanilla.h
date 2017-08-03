/**
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CPU_VANILLA_H
#define CPU_VANILLA_H

#include <stdio.h>
#include "parallel/Mpi.hpp"
#include "data/PatchDecomp.hpp"
#include "constants.hpp"



void update_velocity(real *velocity_x, real *velocity_y, real *velocity_z, int_pt dim_x, int_pt dim_y,
                     int_pt dim_z, int_pt  stride_x, int_pt stride_y, int_pt stride_z, real *stress_xx, real *stress_xy, real *stress_xz,
                     real *stress_yy, real *stress_yz, real *stress_zz, real *crj_x,
                     real *crj_y, real *crj_z, real *density, real dt, real dh);



void update_stress_elastic(real *velocity_x, real *velocity_y, real *velocity_z, int_pt dim_x, int_pt dim_y,
                           int_pt dim_z, int_pt  stride_x, int_pt stride_y, int_pt stride_z, real *stress_xx, real *stress_xy, real *stress_xz,
                           real *stress_yy, real *stress_yz, real *stress_zz, real *coeff, real *crj_x,
                           real *crj_y, real *crj_z, real *density, real *lambda, real *mu, real *lam_mu,
                           real dt, real dh);




void update_stress_visco(real *velocity_x, real *velocity_y, real *velocity_z, int_pt dim_x, int_pt dim_y,
                         int_pt dim_z, int_pt  stride_x, int_pt stride_y, int_pt stride_z, int_pt lam_mu_stride_x, real *stress_xx, real *stress_xy, real *stress_xz,
                         real *stress_yy, real *stress_yz, real *stress_zz, real *coeff, real *crj_x,
                         real *crj_y, real *crj_z, real *density, real *tau1, real *tau2,
                         int *weight_index, real *weight, real *lambda, real *mu, real *lam_mu, real *qp, real *qs,
                         real dt, real dh, real *mem_xx, real *mem_yy, real *mem_zz, real *mem_xy,
                         real *mem_xz, real *mem_yz);




void update_stress_from_fault_sources(int_pt source_timestep, int READ_STEP, int num_model_dimensions,
                                      int *fault_nodes, int num_fault_nodes, int_pt dim_x, int_pt dim_y, int_pt dim_z,
                                      int_pt stride_x, int_pt  stride_y, int_pt stride_z,
                                      real *stress_xx_update, real *stress_xy_update, real *stress_xz_update, real *stress_yy_update,
                                      real *stress_yz_update, real *stress_zz_update,
                                      real **stress_xx_ptr, real **stress_xy_ptr, real **stress_xz_ptr, real **stress_yy_ptr,
                                      real **stress_yz_ptr, real **stress_zz_ptr,
				      real dt, real dh,
                                      PatchDecomp& pd, int_pt start_x, int_pt start_y, int_pt start_z,
                                      int_pt size_x, int_pt size_y, int_pt size_z, int_pt ptch);

void update_free_surface_boundary_stress(real *stress_zz, real *stress_xz, real *stress_yz,
                                         int_pt xstep, int_pt ystep, int_pt zstep,
                                         int_pt dim_x, int_pt dim_y, int_pt dim_z);

#ifdef YASK
void yask_update_free_surface_boundary_stress(Grid_TXYZ *stress_zz, Grid_TXYZ *stress_xz, Grid_TXYZ *stress_yz,
                                              int_pt start_x, int_pt start_y, int_pt start_z,
                                              int_pt size_x, int_pt size_y, int_pt size_z, int_pt timestep);
#endif

void update_free_surface_boundary_velocity(real *velocity_x, real *velocity_y, real *velocity_z,
                                           int_pt xstep, int_pt ystep, int_pt zstep,
                                           int_pt dim_x, int_pt dim_y, int_pt dim_z,
                                           real *lam_mu, int_pt lam_mu_xstep,
                                           bool on_x_max_bdry, bool on_y_zero_bdry);

#ifdef YASK
void yask_update_free_surface_boundary_velocity(Grid_TXYZ *velocity_x, Grid_TXYZ *velocity_y, Grid_TXYZ *velocity_z,
                                                int_pt start_x, int_pt start_y, int_pt start_z,
                                                int_pt size_x, int_pt size_y, int_pt size_z,
                                                real *lam_mu, int_pt lam_mu_xstep, int_pt timestep,
                                                bool on_x_max_bdry, bool on_y_zero_bdry);
#endif






#endif /* CPU_VANILLA_H */
