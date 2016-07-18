/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Main file.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "parallel/Mpi.hpp"
#include "parallel/OpenMP.h"

#include <algorithm>
#include <iostream>
#include <cassert>

#include "io/OptionParser.h"
#include "io/Sources.hpp"
#include "io/OutputWriter.hpp"

#include "data/SoA.hpp"
#include "data/common.hpp"
#include "data/Mesh.hpp"
#include "data/Cerjan.hpp"
#include "data/PatchDecomp.hpp"

#include "kernels/cpu_vanilla.h"
#include "kernels/cpu_scb.h" 

#include "constants.hpp"

#include "data/Grid.hpp"


#include <cmath>
#include <iomanip>

#include <time.h>
#include <sys/time.h>

double wall_time()
{
  struct timeval t;
  if(gettimeofday(&t, NULL))
  {
    return 0;
  }
  return (double) t.tv_sec + 0.000001 * (double) t.tv_usec;
}

// define statics
int odc::parallel::Mpi::m_rank;
int odc::parallel::Mpi::m_size;

#define PATCH_X 1024
#define PATCH_Y 64
#define PATCH_Z 512

#define BDRY_SIZE 24

// TODO: add logging

int main( int i_argc, char *i_argv[] ) {

    std::cout << "welcome to AWP-ODC" << std::endl;    
    
    
    std::cout << "starting MPI" << std::endl;
    // Fire up MPI, dummy call for non-mpi compilation
    odc::parallel::Mpi::initialize( i_argc, i_argv );
    
    
    // parse options
    odc::io::OptionParser l_options( i_argc, i_argv );

    
    std::cout << "setting up data structures" << std::endl;

    // initialize patches
    PatchDecomp patch_decomp;
    patch_decomp.initialize(l_options, l_options.m_nX, l_options.m_nY, l_options.m_nZ,
                            PATCH_X, PATCH_Y, PATCH_Z, BDRY_SIZE);
    
    
    // set up checkpoint writer
    odc::io::CheckpointWriter l_checkpoint(l_options.m_chkFile, l_options.m_nD,
                                           l_options.m_nTiSkp, l_options.m_nZ);
    
    l_checkpoint.writeInitialStats(l_options.m_nTiSkp, l_options.m_dT, l_options.m_dH,
                                   l_options.m_nX, l_options.m_nY,
                                   l_options.m_nZ, l_options.m_numTimesteps,
                                   l_options.m_arbc, l_options.m_nPc, l_options.m_nVe,
                                   l_options.m_fac, l_options.m_q0, l_options.m_ex, l_options.m_fp,
                                   patch_decomp.getVse(false), patch_decomp.getVse(true),
                                   patch_decomp.getVpe(false), patch_decomp.getVpe(true),
                                   patch_decomp.getDde(false), patch_decomp.getDde(true));
    

    patch_decomp.synchronize(true);
    
    
    // set up output writer
    std::cout << "initialized output writer: " << std::endl;
    odc::io::OutputWriter l_output(l_options);
    
    // initialize sources
    std::cout << "initializing sources" << std::endl;
    
    // TODO: Number of nodes (nx, ny, nz) should be aware of MPI partitioning.
    odc::io::Sources l_sources(l_options.m_iFault,
                               l_options.m_nSrc,
                               l_options.m_readStep,
                               l_options.m_nSt,
                               l_options.m_nZ,
                               l_options.m_nX, l_options.m_nY, l_options.m_nZ,
                               l_options.m_inSrc,
                               l_options.m_inSrcI2 );
    
        
    // If one or more source fault nodes are owned by this process then call "addsrc" to update the stress tensor values
    std::cout << "Add initial rupture source" << std::endl;
    l_sources.addsrc(0, l_options.m_dH, l_options.m_dT, l_options.m_nSt,
                     l_options.m_readStep, 3, patch_decomp);

    patch_decomp.synchronize();
    
    // Calculate strides
    int_pt strideX = (l_options.m_nY+2*odc::constants::boundary)*(l_options.m_nZ+2*odc::constants::boundary);
    int_pt strideY = l_options.m_nZ+2*odc::constants::boundary;
    int_pt strideZ = 1;
    
    int_pt lamMuStrideX = l_options.m_nY+2*odc::constants::boundary;

    int_pt numUpdatesPerIter = BDRY_SIZE/6;
    int_pt startMultUpdates = l_options.m_nSt;
    if((l_options.m_nSt % numUpdatesPerIter) != 0)
      startMultUpdates += (l_options.m_nSt - (l_options.m_nSt % numUpdatesPerIter));
      
#pragma omp parallel
{
    double start_time = -1.;
    int start_ts = 0;
    
    int tdsPerWgrp[4] = {1,16,16,16};

    odc::parallel::OpenMP l_omp(l_options.m_nX, l_options.m_nY, l_options.m_nZ,
                                1, tdsPerWgrp, patch_decomp);
    int_pt l_maxPtchsPerWgrp = l_omp.maxNumPtchsPerWgrp();
    
    int_pt start_x, start_y, start_z;
    int_pt size_x, size_y, size_z;
    
    //  Main LOOP Starts
    for (int_pt tloop=0; tloop<=l_options.m_numTimesteps / numUpdatesPerIter; tloop++) {
      for(int_pt ptch = 0; ptch < l_maxPtchsPerWgrp; ptch++) {

        int_pt p_id = l_omp.getPatchNumber(ptch);

        Patch* p = &patch_decomp.m_patches[p_id];
        int_pt h = p->bdry_width;

        bool on_x_max_bdry = l_omp.isOnXMaxBdry(p_id);
        bool on_y_zero_bdry = l_omp.isOnYZeroBdry(p_id);
        bool on_z_bdry = l_omp.isOnZBdry(p_id);
        
        int_pt l_start[3];
        int_pt l_size[3];
        l_omp.getTrdExtent(p_id, l_start, l_size);

        start_x = l_start[0]; start_y = l_start[1]; start_z = l_start[2];
        size_x = l_size[0]; size_y = l_size[1]; size_z = l_size[2];

//        #pragma omp barrier
        

        int_pt n_tval = numUpdatesPerIter;
        if(tloop < startMultUpdates)
          n_tval = 1;
        
        for(int_pt tval=0; tval < n_tval; tval++) {

          int_pt tstep = 0;
          if(tloop > startMultUpdates)
            tstep = (tloop-startMultUpdates) * n_tval + startMultUpdates + tval + 1; 
          else
            tstep = tloop * 1 + tval + 1; 
          
        
          if(l_omp.getThreadNumAll() == 0 && ptch == 0) {
            std::cout << "Beginning  timestep: " << tstep <<  ' ' << tval << ' ' << tloop << ' ' << startMultUpdates << std::endl;
          }


//          #pragma omp barrier
          if(l_omp.participates(ptch)) {

            odc::kernels::SCB::update_velocity(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z],
                        &p->soa.m_velocityZ[start_x][start_y][start_z], size_x, size_y,
                        size_z, p->strideX, p->strideY, p->strideZ, &p->soa.m_stressXX[start_x][start_y][start_z],
                        &p->soa.m_stressXY[start_x][start_y][start_z], &p->soa.m_stressXZ[start_x][start_y][start_z],
                        &p->soa.m_stressYY[start_x][start_y][start_z], &p->soa.m_stressYZ[start_x][start_y][start_z],
                        &p->soa.m_stressZZ[start_x][start_y][start_z], &p->cerjan.m_spongeCoeffX[start_x],
                        &p->cerjan.m_spongeCoeffY[start_y], &p->cerjan.m_spongeCoeffZ[start_z],
                        &p->mesh.m_density[start_x][start_y][start_z], l_options.m_dT, l_options.m_dH);
          }

//          #pragma omp barrier
          
          if(l_omp.participates(ptch) && on_z_bdry) {
                   update_free_surface_boundary_velocity(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z], &p->soa.m_velocityZ[start_x][start_y][start_z], p->strideX, p->strideY, p->strideZ,
                                                         size_x, size_y, size_z, &p->mesh.m_lam_mu[start_x][start_y][0], p->lamMuStrideX,
                                                         on_x_max_bdry, on_y_zero_bdry);
          }

//          #pragma omp barrier

          if(l_omp.participates(ptch)) {        
/*            update_stress_visco(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z],
                            &p->soa.m_velocityZ[start_x][start_y][start_z], size_x, size_y,
                            size_z, p->strideX, p->strideY, p->strideZ, p->lamMuStrideX,
                            &p->soa.m_stressXX[start_x][start_y][start_z],
                            &p->soa.m_stressXY[start_x][start_y][start_z], &p->soa.m_stressXZ[start_x][start_y][start_z],
                            &p->soa.m_stressYY[start_x][start_y][start_z], &p->soa.m_stressYZ[start_x][start_y][start_z],
                            &p->soa.m_stressZZ[start_x][start_y][start_z], &p->mesh.m_coeff[0],
                            &p->cerjan.m_spongeCoeffX[start_x], &p->cerjan.m_spongeCoeffY[start_y],
                            &p->cerjan.m_spongeCoeffZ[start_z], &p->mesh.m_density[start_x][start_y][start_z],
                            &p->mesh.m_tau1[start_x][start_y][start_z], &p->mesh.m_tau2[start_x][start_y][start_z],
                            &p->mesh.m_weight_index[start_x][start_y][start_z], &p->mesh.m_weights[start_x][start_y][start_z],
                            &p->mesh.m_lam[start_x][start_y][start_z], &p->mesh.m_mu[start_x][start_y][start_z],
                            &p->mesh.m_lam_mu[start_x][start_y][0], &p->mesh.m_qp[start_x][start_y][start_z],
                            &p->mesh.m_qs[start_x][start_y][start_z], l_options.m_dT, l_options.m_dH,
                            &p->soa.m_memXX[start_x][start_y][start_z], &p->soa.m_memYY[start_x][start_y][start_z], &p->soa.m_memZZ[start_x][start_y][start_z],
                            &p->soa.m_memXY[start_x][start_y][start_z], &p->soa.m_memXZ[start_x][start_y][start_z], &p->soa.m_memYZ[start_x][start_y][start_z]);*/
            odc::kernels::SCB::update_stress_elastic(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z],
                            &p->soa.m_velocityZ[start_x][start_y][start_z], size_x, size_y,
                            size_z, p->strideX, p->strideY, p->strideZ,
                            &p->soa.m_stressXX[start_x][start_y][start_z],
                            &p->soa.m_stressXY[start_x][start_y][start_z], &p->soa.m_stressXZ[start_x][start_y][start_z],
                            &p->soa.m_stressYY[start_x][start_y][start_z], &p->soa.m_stressYZ[start_x][start_y][start_z],
                            &p->soa.m_stressZZ[start_x][start_y][start_z], &p->mesh.m_coeff[0],
                            &p->cerjan.m_spongeCoeffX[start_x], &p->cerjan.m_spongeCoeffY[start_y],
                            &p->cerjan.m_spongeCoeffZ[start_z], &p->mesh.m_density[start_x][start_y][start_z],
                            &p->mesh.m_lam[start_x][start_y][start_z], &p->mesh.m_mu[start_x][start_y][start_z],
                            &p->mesh.m_lam_mu[start_x][start_y][0], l_options.m_dT, l_options.m_dH);
          }
//          #pragma omp barrier

          if(l_omp.participates(ptch) && on_z_bdry) {
            update_free_surface_boundary_stress(&p->soa.m_stressZZ[start_x][start_y][start_z], &p->soa.m_stressXZ[start_x][start_y][start_z], &p->soa.m_stressYZ[start_x][start_y][start_z],
                                             p->strideX, p->strideY, p->strideZ, size_x, size_y, size_z);
          }
        
//          #pragma omp barrier

          if (tstep < l_options.m_nSt) {
            update_stress_from_fault_sources(tstep, l_options.m_readStep, 3,
                                             &l_sources.m_ptpSrc[0], l_sources.m_nPsrc,
                                             l_options.m_nX, l_options.m_nY, l_options.m_nZ,
                                             strideX, strideY, strideZ,
                                             &l_sources.m_ptAxx[0], &l_sources.
                                             m_ptAxy[0], &l_sources.m_ptAxz[0], &l_sources.m_ptAyy[0], &l_sources.m_ptAyz[0],
                                             &l_sources.m_ptAzz[0], l_options.m_dT, l_options.m_dH,
                                             patch_decomp, start_x - BDRY_SIZE, start_y - BDRY_SIZE, start_z - BDRY_SIZE,
                                             size_x, size_y, size_z, p_id);
          }

          if(l_omp.getThreadNumAll() == 0 && ptch == l_maxPtchsPerWgrp - 1 && tval == n_tval-1) {
            if (tstep >= l_options.m_nSt) {
              if (start_time < 0) {
                start_time = wall_time();
                start_ts = tstep;              
                std::cout << "start time is " << start_time << std::endl;
              }

              else {
                double cur_time = wall_time();
                double avg = (cur_time - start_time) / (tstep - start_ts);

                std::cout << "Time on tstep " << tstep << ": " << cur_time << "; avg = " << avg << std::endl;
              }
            }
          
            patch_decomp.synchronize();
            
            l_output.update(tstep, patch_decomp, l_options.m_nZ);
            l_checkpoint.writeUpdatedStats(tstep, patch_decomp);
          }
        }

//        #pragma omp barrier
      }
    }
}
    
    // release memory
    std::cout << "releasing memory" << std::endl;
    patch_decomp.finalize();
    l_checkpoint.finalize();
    l_output.finalize();
    
    // close mpi
    std::cout << "closing mpi" << std::endl;
    odc::parallel::Mpi::finalize();    
}
