/**
 @brief CPU implementation of finite difference stencil computations
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu_vanilla.h"
#include "instrumentation.hpp"

// Half of the finite difference stencil size (since it looks at two nodes in
// each dimension it has a total size of 4)
#define HALF_STEP 0

// Reference implementation (replaces dvelcx)


void update_velocity( real   *velocity_x, real   *velocity_y, real  *velocity_z,
                     int_pt     dim_x,      int_pt     dim_y,      int_pt    dim_z,
                     int_pt  xstep,      int_pt  ystep,      int_pt zstep,
                     real   *stress_xx,  real   *stress_xy,  real *stress_xz,
                     real *stress_yy, real *stress_yz, real *stress_zz,
                     real *crj_x,      real *crj_y,      real *crj_z,
                     real *density, real dt, real dh) {
    
    SCOREP_USER_FUNC_BEGIN()
    
    real c1 = 9.0/8.0;
    real c2 = -1.0/24.0;
    
    for (int_pt ix=HALF_STEP; ix < dim_x-HALF_STEP; ix++) {
        for (int_pt iy=HALF_STEP; iy < dim_y-HALF_STEP; iy++) {
            for (int_pt iz=HALF_STEP; iz < dim_z-HALF_STEP; iz++) {
                
                
                // Calculate indices for finite difference stencils
                int_pt pos = ix*xstep + iy*ystep + iz*zstep;
                
                int_pt pos_xp1 = pos + xstep;
                int_pt pos_xp2 = pos + 2*xstep;
                int_pt pos_xm1 = pos - xstep;
                int_pt pos_xm2 = pos - 2*xstep;
                
                int_pt pos_yp1 = pos + ystep;
                int_pt pos_yp2 = pos + 2*ystep;
                int_pt pos_ym1 = pos - ystep;
                int_pt pos_ym2 = pos - 2*ystep;
                
                int_pt pos_zp1 = pos + zstep;
                int_pt pos_zp2 = pos + 2*zstep;
                int_pt pos_zm1 = pos - zstep;
                int_pt pos_zm2 = pos - 2*zstep;
                
                int_pt pos_ym1_zm1 = pos - ystep - zstep;
                int_pt pos_xp1_zm1 = pos + xstep - zstep;
                int_pt pos_xp1_ym1 = pos + xstep - ystep;
                
                // Calculate local densities
                real local_density_x = 0.25 * (density[pos] +
                                               density[pos_ym1] +
                                               density[pos_zm1] +
                                               density[pos_ym1_zm1]);
                
                real local_density_y = 0.25 * (density[pos] +
                                               density[pos_xp1] +
                                               density[pos_zm1] +
                                               density[pos_xp1_zm1]);
                
                real local_density_z = 0.25 * (density[pos] +
                                               density[pos_xp1] +
                                               density[pos_ym1] +
                                               density[pos_xp1_ym1]);
                
                // Calculate sponge layer multiplicative factor
                real cerjan = crj_x[ix] * crj_y[iy] * crj_z[iz];
                
                // Update velocities
                velocity_x[pos] += dt/(dh*local_density_x)*(
                                                            c1*(stress_xx[pos] - stress_xx[pos_xm1]) + c2*(stress_xx[pos_xp1] - stress_xx[pos_xm2]) +
                                                            c1*(stress_xy[pos] - stress_xy[pos_ym1]) + c2*(stress_xy[pos_yp1] - stress_xy[pos_ym2]) +
                                                            c1*(stress_xz[pos] - stress_xz[pos_zm1]) + c2*(stress_xz[pos_zp1] - stress_xz[pos_zm2]));
                
                velocity_y[pos] += dt/(dh*local_density_y)*(
                                                            c1*(stress_xy[pos_xp1] - stress_xy[pos]) + c2*(stress_xy[pos_xp2] - stress_xy[pos_xm1]) +
                                                            c1*(stress_yy[pos_yp1] - stress_yy[pos]) + c2*(stress_yy[pos_yp2] - stress_yy[pos_ym1]) +
                                                            c1*(stress_yz[pos] - stress_yz[pos_zm1]) + c2*(stress_yz[pos_zp1] - stress_yz[pos_zm2]));
                velocity_z[pos] += dt/(dh*local_density_z)*(
                                                            c1*(stress_xz[pos_xp1] - stress_xz[pos]) + c2*(stress_xz[pos_xp2] - stress_xz[pos_xm1]) +
                                                            c1*(stress_yz[pos] - stress_yz[pos_ym1]) + c2*(stress_yz[pos_yp1] - stress_yz[pos_ym2]) +
                                                            c1*(stress_zz[pos_zp1] - stress_zz[pos]) + c2*(stress_zz[pos_zp2] - stress_zz[pos_zm1]));
                
                // Multiply by sponge layer factor
                velocity_x[pos] *= cerjan;
                velocity_y[pos] *= cerjan;
                velocity_z[pos] *= cerjan;
                
            } // end z dimension loop
        } // end y dimension loop
    }// end x dimension loop
    
    SCOREP_USER_FUNC_END()
}



real calculate_anelastic_coeff(real q, int_pt weight_index, real weight, real *coeff) {
    if(1.0/q <= 200.0) {
        q = (coeff[weight_index*2-2]*q*q + coeff[weight_index*2-1]*q)/weight;
    } else {
        q *= 0.5;
    }
    return q;
}


// Reference implementation (No anelastic attenuation)
void update_stress_elastic( real   *velocity_x, real   *velocity_y, real   *velocity_z,
                           int_pt     dim_x,      int_pt     dim_y,      int_pt     dim_z,
                           int_pt  xstep,      int_pt  ystep,      int_pt  zstep,
                           real   *stress_xx,  real   *stress_xy,  real   *stress_xz, real *stress_yy, real *stress_yz, real *stress_zz,
                           real   *coeff,
                           real   *crj_x,      real   *crj_y,      real   *crj_z,
                           real *density, real *lambda, real *mu, real *lam_mu,
                           real dt, real dh) {
    
    real c1 = 9.0/8.0;
    real c2 = -1.0/24.0;
    
    for (int_pt ix=HALF_STEP; ix < dim_x-HALF_STEP; ix++) {
        for (int_pt iy=HALF_STEP; iy < dim_y-HALF_STEP; iy++) {
            
            // Apply Finite Difference Stencils
            for (int_pt iz=HALF_STEP; iz < dim_z - HALF_STEP; iz++) {
                
                // Calculate indices for finite difference stencils
                int_pt pos = ix*xstep + iy*ystep + iz*zstep;
                
                int_pt pos_xp1 = pos + xstep;
                int_pt pos_xp2 = pos + 2*xstep;
                int_pt pos_xm1 = pos - xstep;
                int_pt pos_xm2 = pos - 2*xstep;
                
                int_pt pos_yp1 = pos + ystep;
                int_pt pos_yp2 = pos + 2*ystep;
                int_pt pos_ym1 = pos - ystep;
                int_pt pos_ym2 = pos - 2*ystep;
                
                int_pt pos_zp1 = pos + zstep;
                int_pt pos_zp2 = pos + 2*zstep;
                int_pt pos_zm1 = pos - zstep;
                int_pt pos_zm2 = pos - 2*zstep;
                
                int_pt pos_ym1_zm1 = pos - ystep - zstep;
                int_pt pos_xp1_zm1 = pos + xstep - zstep;
                int_pt pos_xp1_ym1 = pos + xstep - ystep;
                
                int_pt pos_xp1_ym1_zm1 = pos + xstep - ystep - zstep;
                
                // Calculate sponge layer multiplicative factor
                real cerjan = crj_x[ix] * crj_y[iy] * crj_z[iz];
                
                // Local average of lambda
                real local_lambda = 8.0/(lambda[pos] + lambda[pos_xp1] + lambda[pos_ym1] + lambda[pos_xp1_ym1]
                                         + lambda[pos_zm1] + lambda[pos_xp1_zm1] + lambda[pos_ym1_zm1] + lambda[pos_xp1_ym1_zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real local_mu_diag = 8.0/(mu[pos] + mu[pos_xp1] + mu[pos_ym1] + mu[pos_xp1_ym1]
                                          + mu[pos_zm1] + mu[pos_xp1_zm1] + mu[pos_ym1_zm1] + mu[pos_xp1_ym1_zm1]);
                
                // Local average of mu for xy stress component
                real local_mu_xy = 2.0/(mu[pos] + mu[pos_zm1]);
                
                // Local average of mu for xz stress component
                real local_mu_xz = 2.0/(mu[pos] + mu[pos_ym1]);
                
                // Local average of mu for yz stress component
                real local_mu_yz = 2.0/(mu[pos] + mu[pos_xp1]);
                
                
                // Estimate strain from velocity
                real strain_xx = (1.0/dh) * (c1*(velocity_x[pos_xp1]-velocity_x[pos]) + c2*(velocity_x[pos_xp2]-velocity_x[pos_xm1]));
                real strain_yy = (1.0/dh) * (c1*(velocity_y[pos]-velocity_y[pos_ym1]) + c2*(velocity_y[pos_yp1]-velocity_y[pos_ym2]));
                real strain_zz = (1.0/dh) * (c1*(velocity_z[pos]-velocity_z[pos_zm1]) + c2*(velocity_z[pos_zp1]-velocity_z[pos_zm2]));
                
                real strain_xy = 0.5 * (1.0/dh) * (c1*(velocity_x[pos_yp1]-velocity_x[pos]) + c2*(velocity_x[pos_yp2]-velocity_x[pos_ym1])+
                                                   c1*(velocity_y[pos]-velocity_y[pos_xm1]) + c2*(velocity_y[pos_xp1]-velocity_y[pos_xm2]));
                
                real strain_xz = 0.5 * (1.0/dh) * (c1*(velocity_x[pos_zp1]-velocity_x[pos]) + c2*(velocity_x[pos_zp2]-velocity_x[pos_zm1])+
                                                   c1*(velocity_z[pos]-velocity_z[pos_xm1]) + c2*(velocity_z[pos_xp1] - velocity_z[pos_xm2]));
                
                real strain_yz = 0.5 * (1.0/dh) * (c1*(velocity_y[pos_zp1]-velocity_y[pos]) + c2*(velocity_y[pos_zp2]-velocity_y[pos_zm1])+
                                                   c1*(velocity_z[pos_yp1]-velocity_z[pos]) + c2*(velocity_z[pos_yp2]-velocity_z[pos_ym1]));
                
                
                // Update tangential stress components
                stress_xx[pos] += 2.0*local_mu_diag*strain_xx*dt + local_lambda * (strain_xx + strain_yy + strain_zz)*dt;
                
                stress_yy[pos] += 2.0*local_mu_diag*strain_yy*dt + local_lambda * (strain_xx + strain_yy + strain_zz)*dt;
                
                stress_zz[pos] += 2.0*local_mu_diag*strain_zz*dt + local_lambda * (strain_xx + strain_yy + strain_zz)*dt;
                
                
                // Update shear stress components
                stress_xy[pos] += 2.0*local_mu_xy*strain_xy*dt;
                stress_xz[pos] += 2.0*local_mu_xz*strain_xz*dt;
                stress_yz[pos] += 2.0*local_mu_yz*strain_yz*dt;
                
                // Apply Absorbing Boundary Condition
                stress_xx[pos] *= cerjan;
                stress_yy[pos] *= cerjan;
                stress_zz[pos] *= cerjan;
                
                stress_xy[pos] *= cerjan;
                stress_xz[pos] *= cerjan;
                stress_yz[pos] *= cerjan;
                
            } // end z dimension loop
            
        } // end y dimension loop
    }// end x dimension loop
} // end function update_stress_elastic


// Reference implementation (replaces dstrqc)

void update_free_surface_boundary_velocity(real *velocity_x, real *velocity_y, real *velocity_z,
                                           int_pt xstep, int_pt ystep, int_pt zstep,
                                           int_pt dim_x, int_pt dim_y, int_pt dim_z,
                                           real *lam_mu, int_pt lam_mu_xstep,
                                           bool on_x_max_bdry, bool on_y_zero_bdry) {
    
    
    for (int_pt ix=HALF_STEP; ix < dim_x-HALF_STEP; ix++) {
        for (int_pt iy=HALF_STEP; iy < dim_y-HALF_STEP; iy++) {
            
            // Apply Free Surface Boundary Conditions for velocity
            // Note: The highest z index corresponds to the surface
            int_pt izs = dim_z-HALF_STEP-1;
            int_pt pos = ix*xstep + iy*ystep + izs*zstep;
            int_pt pos_xp1 = pos + xstep;
            int_pt pos_xm1 = pos - xstep;
            int_pt pos_yp1 = pos + ystep;
            int_pt pos_ym1 = pos - ystep;
            int_pt pos_zp1 = pos + zstep;
            int_pt pos_zm1 = pos - zstep;
            
            velocity_x[pos_zp1] = velocity_x[pos] - (velocity_z[pos] - velocity_z[pos_xm1]);
            velocity_y[pos_zp1] = velocity_y[pos] - (velocity_z[pos_yp1] - velocity_z[pos]);
            
            real dvx = 0.0;
            real dvy = 0.0;
            // TODO: For MPI, this should check to see if this is the last x point on the GLOBAL grid
            if (!on_x_max_bdry || ix < dim_x-HALF_STEP-1) {
                dvx = velocity_x[pos_xp1] - (velocity_z[pos_xp1] - velocity_z[pos]);
            }
            
            // TODO: For MPI, this should check to see if this is the first y point on the GLOBAL grid
            if (!on_y_zero_bdry || iy > 0) {
                dvy = velocity_y[pos_ym1] - (velocity_z[pos] - velocity_z[pos_ym1]);
            }
            
            // TODO(Josh): Remove hardcoded '5' in the following (comes from 2*bdry+1)
            velocity_z[pos_zp1] = velocity_z[pos_zm1] - lam_mu[ix*lam_mu_xstep*5 + iy*5]* 
            ((dvx - velocity_x[pos_zp1]) + (velocity_x[pos_xp1] - velocity_x[pos]) +
             (velocity_y[pos_zp1] - dvy) + (velocity_y[pos] - velocity_y[pos_ym1]));
            
        }
    }
    
}

#ifdef YASK
void yask_update_free_surface_boundary_velocity(Grid_TXYZ *velocity_x, Grid_TXYZ *velocity_y, Grid_TXYZ *velocity_z,
                                                int_pt start_x, int_pt start_y, int_pt start_z,
                                                int_pt size_x, int_pt size_y, int_pt size_z,
                                                real *lam_mu, int_pt lam_mu_xstep, int_pt timestep,
                                                bool on_x_max_bdry, bool on_y_zero_bdry) {
    
    
    for (int_pt ix=start_x; ix < start_x+size_x; ix++) {
        for (int_pt iy=start_y; iy < start_y+size_y; iy++) {
            
            // Apply Free Surface Boundary Conditions for velocity
            // Note: The highest z index corresponds to the surface
            int_pt izs = start_z+size_z-1;

            real vel_x = velocity_x->readElem(timestep, ix, iy, izs, 0);
            real vel_y = velocity_y->readElem(timestep, ix, iy, izs, 0);
            real vel_z = velocity_z->readElem(timestep, ix, iy, izs, 0);

            real vel_z_xp1 = velocity_z->readElem(timestep, ix+1, iy, izs, 0);
            real vel_z_xm1 = velocity_z->readElem(timestep, ix-1, iy, izs, 0);
            real vel_z_yp1 = velocity_z->readElem(timestep, ix, iy+1, izs, 0);
            real vel_z_ym1 = velocity_z->readElem(timestep, ix, iy-1, izs, 0);
            real vel_z_zm1 = velocity_z->readElem(timestep, ix, iy, izs-1, 0);

            real vel_x_xp1 = velocity_x->readElem(timestep, ix+1, iy, izs, 0);

            real vel_y_xp1 = velocity_y->readElem(timestep, ix+1, iy, izs, 0);
            real vel_y_ym1 = velocity_y->readElem(timestep, ix, iy-1, izs, 0);


            velocity_x->writeElem(vel_x - (vel_z - vel_z_xm1), timestep, ix, iy, izs+1, 0);
            velocity_y->writeElem(vel_y - (vel_z_yp1 - vel_z), timestep, ix, iy, izs+1, 0);            

            real vel_x_zp1 = velocity_x->readElem(timestep, ix, iy, izs+1, 0);
            real vel_y_zp1 = velocity_y->readElem(timestep, ix, iy, izs+1, 0);
            
            real dvx = 0.0;
            real dvy = 0.0;
            if (!on_x_max_bdry || ix < start_x+size_x-1) {
                dvx = vel_x_xp1 - (vel_z_xp1 - vel_z);
            }
            
            if (!on_y_zero_bdry || iy > start_y) {
                dvy = vel_y_ym1 - (vel_z - vel_z_ym1);
            }
            
            // TODO(Josh): Remove hardcoded '5' in the following (comes from 2*bdry+1 I think)

            real tmp = vel_z_zm1 - lam_mu[ix*lam_mu_xstep*5 + iy*5]* 
            ((dvx - vel_x_zp1) + (vel_x_xp1 - vel_x) +
             (vel_y_zp1 - dvy) + (vel_y - vel_y_ym1));
            
            velocity_z->writeElem(tmp, timestep, ix, iy, izs+1, 0);
                       
        }
    }
    
}
#endif

void update_free_surface_boundary_stress(real *stress_zz, real *stress_xz, real *stress_yz,
                                         int_pt xstep, int_pt ystep, int_pt zstep,
                                         int_pt dim_x, int_pt dim_y, int_pt dim_z) {
    
    
    for (int_pt ix=HALF_STEP; ix < dim_x-HALF_STEP; ix++) {
        for (int_pt iy=HALF_STEP; iy < dim_y-HALF_STEP; iy++) {
            
            // Apply Free Surface Boundary Conditions for velocity
            // Note: The highest z index corresponds to the surface
            int_pt izs = dim_z-HALF_STEP-1;
            int_pt pos = ix*xstep + iy*ystep + izs*zstep;
            int_pt pos_zp2 = pos + 2*zstep;
            int_pt pos_zp1 = pos + zstep;
            int_pt pos_zm1 = pos - zstep;
            int_pt pos_zm2 = pos - 2*zstep;
            
            // Apply Free Surface Boundary conditions for stress components
            
            stress_zz[pos_zp1] = -stress_zz[pos];
            stress_xz[pos] = 0.0;
            stress_yz[pos] = 0.0;
            
            stress_zz[pos_zp2] = -stress_zz[pos_zm1];
            stress_xz[pos_zp1] = -stress_xz[pos_zm1];
            stress_yz[pos_zp1] = -stress_yz[pos_zm1];
            
            stress_xz[pos_zp2] = -stress_xz[pos_zm2];
            stress_yz[pos_zp2] = -stress_yz[pos_zm2];
            
        }
    }
    
} // end function update_free_surface_boundary_stress

#ifdef YASK
void yask_update_free_surface_boundary_stress(Grid_TXYZ *stress_zz, Grid_TXYZ *stress_xz, Grid_TXYZ *stress_yz,
                                              int_pt start_x, int_pt start_y, int_pt start_z,
                                              int_pt size_x, int_pt size_y, int_pt size_z, int_pt timestep) {
    
    
    for (int_pt ix=start_x; ix < start_x+size_x; ix++) {
        for (int_pt iy=start_y; iy < start_y+size_y; iy++) {
            
            // Apply Free Surface Boundary Conditions for velocity
            // Note: The highest z index corresponds to the surface
            int_pt izs = start_z+size_z-1;
           
            // Apply Free Surface Boundary conditions for stress components
            stress_zz->writeElem(-stress_zz->readElem(timestep,ix,iy,izs,0), timestep, ix, iy, izs+1, 0);
            stress_xz->writeElem(0.0, timestep, ix, iy, izs, 0);
            stress_yz->writeElem(0.0, timestep, ix, iy, izs, 0);
                        
            stress_zz->writeElem(-stress_zz->readElem(timestep,ix,iy,izs-1,0), timestep, ix, iy, izs+2, 0);
            stress_xz->writeElem(-stress_xz->readElem(timestep,ix,iy,izs-1,0), timestep, ix, iy, izs+1, 0);
            stress_yz->writeElem(-stress_yz->readElem(timestep,ix,iy,izs-1,0), timestep, ix, iy, izs+1, 0);
            
            stress_xz->writeElem(-stress_xz->readElem(timestep,ix,iy,izs-2,0), timestep, ix, iy, izs+2, 0);
            stress_yz->writeElem(-stress_yz->readElem(timestep,ix,iy,izs-2,0), timestep, ix, iy, izs+2, 0);            
        }
    }
    
} // end function update_free_surface_boundary_stress
#endif


void update_stress_visco(real *velocity_x, real *velocity_y, real *velocity_z,
                         int_pt dim_x, int_pt dim_y, int_pt dim_z,
                         int_pt  xstep, int_pt ystep, int_pt zstep,
                         int_pt lam_mu_xstep,
                         real *stress_xx, real *stress_xy, real *stress_xz,
                         real *stress_yy, real *stress_yz, real *stress_zz, real *coeff, real *crj_x,
                         real *crj_y, real *crj_z, real *density, real *tau1, real *tau2,
                         int *weight_index, real *weight, real *lambda, real *mu, real *lam_mu, real *qp, real *qs,
                         real dt, real dh, real *mem_xx, real *mem_yy, real *mem_zz, real *mem_xy,
                         real *mem_xz, real *mem_yz) {
    
    
    real c1 = 9.0/8.0;
    real c2 = -1.0/24.0;
    
    for (int_pt ix=HALF_STEP; ix < dim_x-HALF_STEP; ix++) {
        for (int_pt iy=HALF_STEP; iy < dim_y-HALF_STEP; iy++) {
            
            // Apply Finite Difference Stencils
            for (int_pt iz=HALF_STEP; iz < dim_z - HALF_STEP; iz++) {
                
                // Calculate indices for finite difference stencils
                int_pt pos = ix*xstep + iy*ystep + iz*zstep;
                
                int_pt pos_xp1 = pos + xstep;
                int_pt pos_xp2 = pos + 2*xstep;
                int_pt pos_xm1 = pos - xstep;
                int_pt pos_xm2 = pos - 2*xstep;
                
                int_pt pos_yp1 = pos + ystep;
                int_pt pos_yp2 = pos + 2*ystep;
                int_pt pos_ym1 = pos - ystep;
                int_pt pos_ym2 = pos - 2*ystep;
                
                int_pt pos_zp1 = pos + zstep;
                int_pt pos_zp2 = pos + 2*zstep;
                int_pt pos_zm1 = pos - zstep;
                int_pt pos_zm2 = pos - 2*zstep;
                
                int_pt pos_ym1_zm1 = pos - ystep - zstep;
                int_pt pos_xp1_zm1 = pos + xstep - zstep;
                int_pt pos_xp1_ym1 = pos + xstep - ystep;
                
                int_pt pos_xp1_ym1_zm1 = pos + xstep - ystep - zstep;
                
                // Calculate sponge layer multiplicative factor
                real cerjan = crj_x[ix] * crj_y[iy] * crj_z[iz];
                
                // Local average of lambda
                real local_lambda = 8.0/(lambda[pos] + lambda[pos_xp1] + lambda[pos_ym1] + lambda[pos_xp1_ym1]
                                         + lambda[pos_zm1] + lambda[pos_xp1_zm1] + lambda[pos_ym1_zm1] + lambda[pos_xp1_ym1_zm1]);
                
                // Local average of mu for diagonal stress components (xx, yy, and zz)
                real local_mu_diag = 8.0/(mu[pos] + mu[pos_xp1] + mu[pos_ym1] + mu[pos_xp1_ym1]
                                          + mu[pos_zm1] + mu[pos_xp1_zm1] + mu[pos_ym1_zm1] + mu[pos_xp1_ym1_zm1]);
                
                // Local average of mu for xy stress component
                real local_mu_xy = 2.0/(mu[pos] + mu[pos_zm1]);
                
                // Local average of mu for xz stress component
                real local_mu_xz = 2.0/(mu[pos] + mu[pos_ym1]);
                
                // Local average of mu for yz stress component
                real local_mu_yz = 2.0/(mu[pos] + mu[pos_xp1]);
                
                // Local quality factor for P waves
                real local_qp = 0.125*(qp[pos] + qp[pos_xp1] + qp[pos_ym1] + qp[pos_xp1_ym1] + qp[pos_zm1] +
                                       qp[pos_xp1_zm1] + qp[pos_ym1_zm1] + qp[pos_xp1_ym1_zm1]);
                
                // Local quality factor for S waves for diagonal stress components
                real local_qs_diag = 0.125*(qs[pos] + qs[pos_xp1] + qs[pos_ym1] + qs[pos_xp1_ym1] + qs[pos_zm1] +
                                            qs[pos_xp1_zm1] + qs[pos_ym1_zm1] + qs[pos_xp1_ym1_zm1]);
                
                // Local quality factor for S waves for xy stress component
                real local_qs_xy = 0.5*(qs[pos] + qs[pos_zm1]);
                
                // Local quality factor for S waves for xz stress component
                real local_qs_xz = 0.5*(qs[pos] + qs[pos_ym1]);
                
                // Local quality factor for S waves for yz stress component
                real local_qs_yz = 0.5*(qs[pos] + qs[pos_xp1]);
                
                // Anelastic Coefficients
                real ap = calculate_anelastic_coeff(local_qp, weight_index[pos], weight[pos], coeff);
                real as_diag = calculate_anelastic_coeff(local_qs_diag, weight_index[pos], weight[pos], coeff);
                real as_xy = calculate_anelastic_coeff(local_qs_xy, weight_index[pos], weight[pos], coeff);
                real as_xz = calculate_anelastic_coeff(local_qs_xz, weight_index[pos], weight[pos], coeff);
                real as_yz = calculate_anelastic_coeff(local_qs_yz, weight_index[pos], weight[pos], coeff);
                
                
                // Estimate strain from velocity
                real strain_xx = (1.0/dh) * (c1*(velocity_x[pos_xp1]-velocity_x[pos]) + c2*(velocity_x[pos_xp2]-velocity_x[pos_xm1]));
                real strain_yy = (1.0/dh) * (c1*(velocity_y[pos]-velocity_y[pos_ym1]) + c2*(velocity_y[pos_yp1]-velocity_y[pos_ym2]));
                real strain_zz = (1.0/dh) * (c1*(velocity_z[pos]-velocity_z[pos_zm1]) + c2*(velocity_z[pos_zp1]-velocity_z[pos_zm2]));
                
                real strain_xy = 0.5 * (1.0/dh) * (c1*(velocity_x[pos_yp1]-velocity_x[pos]) + c2*(velocity_x[pos_yp2]-velocity_x[pos_ym1])+
                                                   c1*(velocity_y[pos]-velocity_y[pos_xm1]) + c2*(velocity_y[pos_xp1]-velocity_y[pos_xm2]));
                
                real strain_xz = 0.5 * (1.0/dh) * (c1*(velocity_x[pos_zp1]-velocity_x[pos]) + c2*(velocity_x[pos_zp2]-velocity_x[pos_zm1])+
                                                   c1*(velocity_z[pos]-velocity_z[pos_xm1]) + c2*(velocity_z[pos_xp1] - velocity_z[pos_xm2]));
                
                real strain_yz = 0.5 * (1.0/dh) * (c1*(velocity_y[pos_zp1]-velocity_y[pos]) + c2*(velocity_y[pos_zp2]-velocity_y[pos_zm1])+
                                                   c1*(velocity_z[pos_yp1]-velocity_z[pos]) + c2*(velocity_z[pos_yp2]-velocity_z[pos_ym1]));
                
                
                // Update memory variables for stress components
                real mem_xx_old = mem_xx[pos];
                real mem_yy_old = mem_yy[pos];
                real mem_zz_old = mem_zz[pos];
                
                mem_xx[pos] = tau2[pos]*mem_xx[pos] + tau1[pos]*weight[pos]*(2.0*local_mu_diag*as_diag*(strain_yy+strain_zz)-
                                                                             (2.0*local_mu_diag + local_lambda)*ap*(strain_xx+strain_yy+strain_zz));
                
                mem_yy[pos] = tau2[pos]*mem_yy[pos] + tau1[pos]*weight[pos]*(2.0*local_mu_diag*as_diag*(strain_xx+strain_zz)-
                                                                             (2.0*local_mu_diag + local_lambda)*ap*(strain_xx+strain_yy+strain_zz));
                
                mem_zz[pos] = tau2[pos]*mem_zz[pos] + tau1[pos]*weight[pos]*(2.0*local_mu_diag*as_diag*(strain_xx+strain_yy)-
                                                                             (2.0*local_mu_diag + local_lambda)*ap*(strain_xx+strain_yy+strain_zz));
                
                // Update tangential stress components
                stress_xx[pos] += 2.0*local_mu_diag*strain_xx*dt + local_lambda * (strain_xx + strain_yy + strain_zz)*dt +
                mem_xx[pos]*dt + mem_xx_old*dt;
                
                stress_yy[pos] += 2.0*local_mu_diag*strain_yy*dt + local_lambda * (strain_xx + strain_yy + strain_zz)*dt +
                mem_yy[pos]*dt + mem_yy_old*dt;
                
                stress_zz[pos] += 2.0*local_mu_diag*strain_zz*dt + local_lambda * (strain_xx + strain_yy + strain_zz)*dt +
                mem_zz[pos]*dt + mem_zz_old*dt;
                
                
                // Update memory variables for shear stress components
                real mem_xy_old = mem_xy[pos];
                real mem_xz_old = mem_xz[pos];
                real mem_yz_old = mem_yz[pos];
                
                mem_xy[pos] = tau2[pos]*mem_xy[pos] - tau1[pos]*weight[pos] * (2.0*local_mu_xy*as_xy*strain_xy);
                mem_xz[pos] = tau2[pos]*mem_xz[pos] - tau1[pos]*weight[pos] * (2.0*local_mu_xz*as_xz*strain_xz);
                mem_yz[pos] = tau2[pos]*mem_yz[pos] - tau1[pos]*weight[pos] * (2.0*local_mu_yz*as_yz*strain_yz);
                
                // Update shear stress components
                stress_xy[pos] += 2.0*local_mu_xy*strain_xy*dt + mem_xy[pos]*dt + mem_xy_old*dt;
                stress_xz[pos] += 2.0*local_mu_xz*strain_xz*dt + mem_xz[pos]*dt + mem_xz_old*dt;
                stress_yz[pos] += 2.0*local_mu_yz*strain_yz*dt + mem_yz[pos]*dt + mem_yz_old*dt;
                
                // Apply Absorbing Boundary Condition
                stress_xx[pos] *= cerjan;
                stress_yy[pos] *= cerjan;
                stress_zz[pos] *= cerjan;
                
                stress_xy[pos] *= cerjan;
                stress_xz[pos] *= cerjan;
                stress_yz[pos] *= cerjan;
                
            } // end z dimension loop
            
        } // end y dimension loop
    }// end x dimension loop
    
} // end function update_stress_visco






// Reference implementation (replaces addsrc_cu)

void update_stress_from_fault_sources(int_pt source_timestep, int READ_STEP, int num_model_dimensions,
                                      int *fault_nodes, int num_fault_nodes, int_pt dim_x, int_pt dim_y, int_pt dim_z,
                                      int_pt  xstep, int_pt ystep, int_pt zstep,
                                      real *stress_xx_update, real *stress_xy_update, real *stress_xz_update, real *stress_yy_update,
                                      real *stress_yz_update, real *stress_zz_update,
                                      real **stress_xx_ptr, real **stress_xy_ptr, real **stress_xz_ptr, real **stress_yy_ptr,
                                      real **stress_yz_ptr, real **stress_zz_ptr,
				      real dt, real dh,
                                      PatchDecomp& pd, int_pt start_x, int_pt start_y, int_pt start_z,
                                      int_pt size_x, int_pt size_y, int_pt size_z, int_pt ptch) {
    
    real coeff = dt/(dh*dh*dh);
    
    for (int_pt j=0; j < num_fault_nodes; j++) {

        real* sXX = stress_xx_ptr[j];
        real* sXY = stress_xy_ptr[j];
        real* sXZ = stress_xz_ptr[j];
        real* sYY = stress_yy_ptr[j];
        real* sYZ = stress_yz_ptr[j];
        real* sZZ = stress_zz_ptr[j];

        *sXX -= coeff*stress_xx_update[j*READ_STEP+source_timestep];
        *sXY -= coeff*stress_xy_update[j*READ_STEP+source_timestep];
        *sXZ -= coeff*stress_xz_update[j*READ_STEP+source_timestep];
        *sYY -= coeff*stress_yy_update[j*READ_STEP+source_timestep];
        *sYZ -= coeff*stress_yz_update[j*READ_STEP+source_timestep];
        *sZZ -= coeff*stress_zz_update[j*READ_STEP+source_timestep];
    }
}




