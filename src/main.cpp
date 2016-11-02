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

// TODO(Josh): verify that source term indexing matches GPU, comment thoroughly
// TODO(Josh): add checks for READ_STEP
// TODO(Josh): handle buffer-term source term insertions
// TODO: add logging

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
#include "data/Grid.hpp"

#include "kernels/cpu_vanilla.h"
#include "kernels/cpu_scb.h" 

#include "constants.hpp"
#include "util.hpp"

#include <cmath>
#include <iomanip>


int main( int i_argc, char *i_argv[] )
{
  std::cout << "Welcome to AWP-ODC-OS" << std::endl;    
    
  // parse options
  std::cout << "Parsing command line options" << std::endl;
  odc::io::OptionParser l_options( i_argc, i_argv );

    
#ifdef AWP_USE_MPI    
  std::cout << "starting MPI" << std::endl;
#endif
    
  // Fire up MPI, dummy call for non-mpi compilation
  if(!odc::parallel::Mpi::initialize( i_argc, i_argv, l_options ))
  {
    std::cout << "\tError during MPI initialization, aborting." << std::endl;
    return 0;
  }

  // Get MPI details (just for code brevity)
  const int l_rank = odc::parallel::Mpi::m_rank;
  const int_pt l_rangeX = odc::parallel::Mpi::m_rangeX;
  const int_pt l_rangeY = odc::parallel::Mpi::m_rangeY;
  const int_pt l_rangeZ = odc::parallel::Mpi::m_rangeZ;
  const int_pt l_numRanks = odc::parallel::Mpi::m_size;
    
  std::cout << "setting up data structures" << std::endl;

  // initialize patches
  PatchDecomp patch_decomp;
  patch_decomp.initialize(l_options, l_rangeX, l_rangeY, l_rangeZ,
			  PATCH_X, PATCH_Y, PATCH_Z, BDRY_SIZE);
        
  // set up checkpoint writer
  odc::io::CheckpointWriter l_checkpoint(l_options.m_chkFile, l_options.m_nD,
					 l_options.m_nTiSkp, l_options.m_nZ);

  if(l_rank == 0)
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
  if(l_rank == 0)
    std::cout << "initialized output writer: " << std::endl;
  odc::io::OutputWriter l_output(l_options);
    
  // initialize sources
  if(l_rank == 0)
    std::cout << "initializing sources" << std::endl;
    
  // TODO: Number of nodes (nx, ny, nz) should be aware of MPI partitioning.
  odc::io::Sources l_sources(l_options.m_iFault,
			     l_options.m_nSrc,
			     l_options.m_readStep,
			     l_options.m_nSt,
			     l_options.m_nZ,
			     l_options.m_nX, l_options.m_nY, l_options.m_nZ,
			     l_options.m_inSrc,
			     l_options.m_inSrcI2,
                             patch_decomp);
    
        
  // If one or more source fault nodes are owned by this process then call "addsrc" to update the stress tensor values
  if(l_rank == 0)
    std::cout << "Add initial rupture source" << std::endl;
  int_pt initial_ts = 0;
#ifdef YASK
  //TODO(Josh): why is this offset needed?
  initial_ts = 0;
#endif
  l_sources.addsrc(initial_ts, l_options.m_dH, l_options.m_dT, l_options.m_nSt,
		   l_options.m_readStep, 3, patch_decomp);

  // PPP: bring this back
  //for(int i_dir=0; i_dir<3; i_dir++)
  //  patch_decomp.stressMpiSynchronize(i_dir,0);
    
  patch_decomp.synchronize();
    
  // Calculate strides
  int_pt strideX = (l_options.m_nY+2*odc::constants::boundary)*(l_options.m_nZ+2*odc::constants::boundary);
  int_pt strideY = l_options.m_nZ+2*odc::constants::boundary;
  int_pt strideZ = 1; 
  int_pt lamMuStrideX = l_options.m_nY+2*odc::constants::boundary;

  // Set the number of computational threads we will use
  const int numCompThreads = 62;
  // PPP: This is LEGACY and should be removed:
  int tdsPerWgrp[2] = {numCompThreads,1};

  // TODO(Josh): make this an int_pt?
  volatile int nextWP[numCompThreads][2];
 
    
#pragma omp parallel num_threads(numCompThreads+1)
  {
    // Some performance monitoring variables
    double start_time = -1.;
    int start_ts = 0;

    odc::parallel::OpenMP l_omp(odc::parallel::Mpi::m_rangeX, odc::parallel::Mpi::m_rangeY, odc::parallel::Mpi::m_rangeZ, 1, tdsPerWgrp[0], patch_decomp);
    
    int_pt start_x, start_y, start_z;
    int_pt end_x, end_y, end_z;
    int_pt size_x, size_y, size_z;

    // Determine if current thread is computational or management
    bool amCompThread = l_omp.isComputationThread();
    bool amManageThread = !amCompThread;

    odc::parallel::OmpManager l_ompManager(numCompThreads, l_omp.m_nWP, l_omp);

    int compThreadId = -1;
    if(amCompThread)
      compThreadId = l_omp.getThreadNumGrp();

    // For computational threads, this keeps track of last WP assigned
    int lastAssignment = 0;    

#pragma omp barrier
    
    //  Main LOOP Starts
    for (int_pt tstep=1; tstep<=l_options.m_numTimesteps; tstep++)
    {
      // Currently patch decomp is disabled, so current patch is always 0
      int_pt p_id = 0;
      // For brevity of code, set some local variables
      Patch* p = &patch_decomp.m_patches[p_id];
      int_pt h = p->bdry_width;        
        
      if(amManageThread && l_rank == 0)
      {
        std::cout << "Beginning  timestep: " << tstep << std::endl;
      }

      if(amManageThread)
      {
        l_ompManager.initialWorkPackages(&nextWP[0][0]);
      }

      // Barrier until we have assigned initial work packages
#pragma omp barrier	  


      // Main computation thread process loop
      lastAssignment = 0;
      while(amCompThread)
      {
        int nextWPId = nextWP[compThreadId][lastAssignment];
	// Negative WP IDs signal that management thread has yet to assign
	while(nextWPId < 0)
	{
	  // Check the other WP slot for this thread
	  if(nextWP[compThreadId][1-lastAssignment] > 0)
	    lastAssignment = 1-lastAssignment;	      
	  nextWPId = nextWP[compThreadId][lastAssignment];
        }

	// If WP ID is larger than number of WPs, this is signal to quit.
	// Before we do this though, make sure that the other WP slot for
	// this thread doesn't contain an unfinished WP
	if(nextWPId > l_omp.m_nWP)
	{
	  if(nextWP[compThreadId][1-lastAssignment] <= l_omp.m_nWP)
	  {
	    // We have found a new WP to do, jump back to start
	    lastAssignment = 1-lastAssignment;
	    continue;
	  }
	  else
	  {
	    // All slots for this thread are done, and management thread
	    // has signalled to quit.  So let's get out of here.
	    break;
	  }
	}


	// Time to actually do the work.  Establish WP extent
	odc::parallel::WorkPackage& l_curWP = l_omp.m_workPackages[nextWPId-1];

        start_x = l_curWP.start[0] + h;
	start_y = l_curWP.start[1] + h;
	start_z = l_curWP.start[2] + h;

	end_x = l_curWP.end[0] + h;
	end_y = l_curWP.end[1] + h;
	end_z = l_curWP.end[2] + h;

	size_x = end_x - start_x;
	size_y = end_y - start_y;
	size_z = end_z - start_z;


	if(l_curWP.type == odc::parallel::WP_VelUpdate)
	{
	  if(l_curWP.copyFromBuffer)
	  {
	    for(int l_x = 0; l_x <= 2; l_x++)
	    {
	      for(int l_y = 0; l_y <= 2; l_y++)
	      {
		for(int l_z = 0; l_z <= 2; l_z++)
		{
		  if(l_curWP.mpiDir[l_x][l_y][l_z])
		    patch_decomp.copyStressBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[l_x][l_y][l_z], l_x-1, l_y-1, l_z-1, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, tstep+1);
		}
	      }
	    }
	  }
	    
#ifdef YASK
	  applyYASKStencil(p->yask_context, &(p->yask_stencils), 0, tstep, start_x, start_y, start_z,
			   size_x, size_y, size_z);
#else  
	  odc::kernels::SCB::update_velocity(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z],
					     &p->soa.m_velocityZ[start_x][start_y][start_z], size_x, size_y,
					     size_z, p->strideX, p->strideY, p->strideZ, &p->soa.m_stressXX[start_x][start_y][start_z],
					     &p->soa.m_stressXY[start_x][start_y][start_z], &p->soa.m_stressXZ[start_x][start_y][start_z],
					     &p->soa.m_stressYY[start_x][start_y][start_z], &p->soa.m_stressYZ[start_x][start_y][start_z],
					     &p->soa.m_stressZZ[start_x][start_y][start_z], &p->cerjan.m_spongeCoeffX[start_x],
					     &p->cerjan.m_spongeCoeffY[start_y], &p->cerjan.m_spongeCoeffZ[start_z],
					     &p->mesh.m_density[start_x][start_y][start_z], l_options.m_dT, l_options.m_dH);
#endif

	  if(l_curWP.copyToBuffer)
	  {
	    for(int l_x = 0; l_x <= 2; l_x++)
	    {
	      for(int l_y = 0; l_y <= 2; l_y++)
	      {
		for(int l_z = 0; l_z <= 2; l_z++)
		{
		  if(l_curWP.mpiDir[l_x][l_y][l_z])
		  {
		    patch_decomp.copyVelBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[l_x][l_y][l_z], l_x-1, l_y-1, l_z-1, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, tstep+1);
		  }
		}
	      }
	    }
	  }
	      
	}

	else if(l_curWP.type == odc::parallel::WP_StressUpdate)
	{

	  if(l_curWP.copyFromBuffer)
	  {
	    for(int l_x = 0; l_x <= 2; l_x++)
	    {
	      for(int l_y = 0; l_y <= 2; l_y++)
	      {
		for(int l_z = 0; l_z <= 2; l_z++)
		{
		  if(l_curWP.mpiDir[l_x][l_y][l_z])
		    patch_decomp.copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[l_x][l_y][l_z], l_x-1, l_y-1, l_z-1, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, tstep+1);
		}
	      }
	    }
	  }

	  if(l_curWP.freeSurface)
	  {
#ifdef YASK
	    yask_update_free_surface_boundary_velocity((Grid_TXYZ *)p->yask_context.vel_x,
						       (Grid_TXYZ *)p->yask_context.vel_y,
						       (Grid_TXYZ *)p->yask_context.vel_z,
						       start_x, start_y, start_z,
						       size_x, size_y, size_z,
						       &p->mesh.m_lam_mu[0][0][0], p->lamMuStrideX,
						       tstep+1, l_curWP.xMaxBdry, l_curWP.yMinBdry);
#else
	    update_free_surface_boundary_velocity(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z], &p->soa.m_velocityZ[start_x][start_y][start_z],
						  p->strideX, p->strideY, p->strideZ,
						  size_x, size_y, size_z, &p->mesh.m_lam_mu[start_x][start_y][0], p->lamMuStrideX,
						  l_curWP.xMaxBdry, l_curWP.yMinBdry);
#endif		
	  }

	      
#ifdef YASK
	  applyYASKStencil(p->yask_context, &(p->yask_stencils), 1, tstep, start_x, start_y, start_z,
			   size_x, size_y, size_z);
#else
	  update_stress_visco(&p->soa.m_velocityX[start_x][start_y][start_z], &p->soa.m_velocityY[start_x][start_y][start_z],
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
                              &p->soa.m_memXY[start_x][start_y][start_z], &p->soa.m_memXZ[start_x][start_y][start_z], &p->soa.m_memYZ[start_x][start_y][start_z]);
#endif

	  if(l_curWP.freeSurface)
	  {
#ifdef YASK
	    yask_update_free_surface_boundary_stress((Grid_TXYZ *)p->yask_context.stress_zz,(Grid_TXYZ *)p->yask_context.stress_xz,
						     (Grid_TXYZ *)p->yask_context.stress_yz, start_x, start_y, start_z,
						     size_x, size_y, size_z, tstep+1);            
#else
	    update_free_surface_boundary_stress(&p->soa.m_stressZZ[start_x][start_y][start_z], &p->soa.m_stressXZ[start_x][start_y][start_z], &p->soa.m_stressYZ[start_x][start_y][start_z],
						p->strideX, p->strideY, p->strideZ, size_x, size_y, size_z);
#endif		
	  }

	  if(l_curWP.copyToBuffer)
	  {
	    for(int l_x = 0; l_x <= 2; l_x++)
	    {
	      for(int l_y = 0; l_y <= 2; l_y++)
	      {
		for(int l_z = 0; l_z <= 2; l_z++)
		{
		  if(l_curWP.mpiDir[l_x][l_y][l_z])
		    patch_decomp.copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[l_x][l_y][l_z], l_x-1, l_y-1, l_z-1, start_x-h, start_y-h, start_z-h, end_x-h, end_y-h, end_z-h, tstep+1);
		}
	      }
	    }
	  }	      
	      
	}

	else if(l_curWP.type == odc::parallel::WP_MPI_Vel)
	{
	  for(int i=0; i<3; i++)
	    odc::parallel::Mpi::sendRecvBuffers(3, i);
	}

	else if(l_curWP.type == odc::parallel::WP_MPI_Stress)
	{
	  for(int i=0; i<3; i++)
	    odc::parallel::Mpi::sendRecvBuffers(6, i);
	}
	    

	// We are done with this WP.  Mark it as negative to signal to management thread
	// that it is complete
	nextWP[compThreadId][lastAssignment] = -nextWP[compThreadId][lastAssignment]; 
	lastAssignment = 1 - lastAssignment;
      }


      // Main management thread process loop
      if(amManageThread)
      {
	int num_abort = 0;
	int next_task;
	while(num_abort < numCompThreads)
	{
	  num_abort = 0;
	  for(int l_td = 0; l_td < numCompThreads; l_td++)
	  {
	    for(int l_task=0; l_task<2; l_task++)
	    {
	      if(nextWP[l_td][l_task] < 0)
	      {
		l_ompManager.setDone(-nextWP[l_td][l_task]);
		if(l_ompManager.tasksLeft())
		{
		  next_task = l_ompManager.nextTask();
		  if(next_task >= 0)
		    nextWP[l_td][l_task] = next_task + 1;
		}
		else
		{
		  nextWP[l_td][l_task] = l_omp.m_nWP+1;
		}
	      }
	    }

	    if(nextWP[l_td][0] > l_omp.m_nWP && nextWP[l_td][1] > l_omp.m_nWP)
	      num_abort++;
	  }
	}
      }


      // We are done with the main updates of this loop.  Just remains to update source and handle output
#pragma omp barrier
	  
      // PPP: parallelize this
      if (tstep < l_options.m_nSt && amManageThread) {  
	update_stress_from_fault_sources(tstep, l_options.m_readStep, 3,
					 &l_sources.m_ptpSrc[0], l_sources.m_nPsrc,
					 l_options.m_nX, l_options.m_nY, l_options.m_nZ,
					 strideX, strideY, strideZ,
					 &l_sources.m_ptAxx[0], &l_sources.m_ptAxy[0], &l_sources.m_ptAxz[0],
					 &l_sources.m_ptAyy[0], &l_sources.m_ptAyz[0], &l_sources.m_ptAzz[0],
					 &l_sources.m_locStrXX[0], &l_sources.m_locStrXY[0], &l_sources.m_locStrXZ[0],
					 &l_sources.m_locStrYY[0], &l_sources.m_locStrYZ[0], &l_sources.m_locStrZZ[0],
					 l_options.m_dT, l_options.m_dH,
					 patch_decomp, 0, 0, 0,
					 odc::parallel::Mpi::m_rangeX + 2*h, odc::parallel::Mpi::m_rangeY + 2*h,
					 odc::parallel::Mpi::m_rangeZ + 2*h, p_id);
      }

#pragma omp barrier

      // For one thread on each rank, handle output / management
      if(l_omp.getThreadNumAll() == 0)
      {
	if (tstep > 10 && l_rank == 0)
	{
	  if (start_time < 0)
	  {
	    start_time = wall_time();
	    start_ts = tstep;              
	    std::cout << "start time is " << start_time << std::endl;
	  }

	  else
	  {
	    double cur_time = wall_time();
	    double avg = (cur_time - start_time) / (tstep - start_ts);

	    std::cout << "Time on tstep " << tstep << ": " << cur_time << "; avg = " << avg << std::endl;
	  }
	}
          
	patch_decomp.synchronize();

	//l_output.update(tstep, patch_decomp);
	l_checkpoint.writeUpdatedStats(tstep, patch_decomp);
      }
    }
#pragma omp barrier     
  }
    
  // release memory
  std::cout << "releasing memory" << std::endl;
  patch_decomp.finalize();
  l_checkpoint.finalize();
  l_output.finalize();
    
  // close mpi
  std::cout << "closing mpi" << std::endl;
  odc::parallel::Mpi::finalize();

  return 0;
}
