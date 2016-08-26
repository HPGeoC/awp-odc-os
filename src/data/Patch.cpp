/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 
 @section DESCRIPTION
 Patch data structure for mcdram blocking.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Patch.hpp"

#include <iostream>

Patch::Patch()
{
}

Patch::Patch(int_pt _nx, int_pt _ny, int_pt _nz, int_pt _bw)
        : nx(_nx), ny(_ny), nz(_nz), bdry_width(_bw)
{
  size_x = nx + 2*bdry_width;
  size_y = ny + 2*bdry_width;
  size_z = nz + 2*bdry_width;
  
  
  soa.initialize(size_x, size_y, size_z);
  soa.allocate();

  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      for(int k=0; k<3; k++)
        neighbors[i][j][k] = 0;
}

void Patch::initialize(odc::io::OptionParser i_options, int_pt _nx, int_pt _ny, int_pt _nz, int_pt _bw,
                       int_pt i_globalX, int_pt i_globalY, int_pt i_globalZ, Grid1D i_inputBuffer)
{
  nx = _nx;
  ny = _ny;
  nz = _nz;
  bdry_width = _bw;

  size_x = nx + 2*bdry_width;
  size_y = ny + 2*bdry_width;
  size_z = nz + 2*bdry_width;
 
#ifdef YASK
  yask_context.dn = 4;
  yask_context.dx = size_x;
  yask_context.dy = size_y;
  yask_context.dz = size_z;
    
  yask_context.rt = 1;
  yask_context.rn = 0;
  yask_context.rx = 0; yask_context.ry = 0; yask_context.rz = 0;
  yask_context.bt = 1;
  yask_context.bn = 1;
  yask_context.bx = DEF_BLOCK_SIZE;
  yask_context.by = DEF_BLOCK_SIZE;
  yask_context.bz = DEF_BLOCK_SIZE;    
    
  yask_context.pn = roundUp(0, VLEN_N, "# extra padding in n");
  yask_context.px = roundUp(2, VLEN_X, "# extra padding in x");
  yask_context.py = roundUp(2, VLEN_Y, "# extra padding in y");
  yask_context.pz = roundUp(2, VLEN_Z, "# extra padding in z");

  idx_t halo_size = 4/2; // TODO: make dim-specific.
  yask_context.hn = 0;
  yask_context.hx = ROUND_UP(halo_size, VLEN_X);
  yask_context.hy = ROUND_UP(halo_size, VLEN_Y);
  yask_context.hz = ROUND_UP(halo_size, VLEN_Z);

  yask_context.allocGrids();
  yask_context.allocParams();

  (*(yask_context.h))() = i_options.m_dH;
  (*(yask_context.delta_t))() = i_options.m_dT;
#else
  std::cout << "\t setting up patch soa" << std::endl;
  soa.initialize(size_x, size_y, size_z);
  std::cout << "\t size is " << soa.getSize() << std::endl;
  soa.allocate();
#endif

  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      for(int k=0; k<3; k++)
        neighbors[i][j][k] = 0;


  // TODO(Josh): don't hardcode the true (analestic) here;
  //             the advantage of having it is that all memory gets
  //             allocated, just in case we need it

#ifdef YASK  
  mesh.initialize(i_options, nx, ny, nz, bdry_width, true, i_inputBuffer, i_globalX, i_globalY, i_globalZ,
                  (Grid_XYZ*) yask_context.rho, (Grid_XYZ*) yask_context.mu, (Grid_XYZ*) yask_context.lambda,
                  (Grid_XYZ*) yask_context.weight, (Grid_XYZ*) yask_context.tau2, (Grid_XYZ*) yask_context.anelastic_ap,
                  (Grid_XYZ*) yask_context.anelastic_as_diag, (Grid_XYZ*) yask_context.anelastic_xy,
                  (Grid_XYZ*) yask_context.anelastic_xz, (Grid_XYZ*) yask_context.anelastic_yz);
#else
  mesh.initialize(i_options, nx, ny, nz, bdry_width, true, i_inputBuffer, i_globalX, i_globalY, i_globalZ);
#endif
  
  int_pt coords[] = {i_globalX+odc::parallel::Mpi::m_startX, i_globalY+odc::parallel::Mpi::m_startY, i_globalZ+odc::parallel::Mpi::m_startZ};
  cerjan.initialize(i_options, nx, ny, nz, bdry_width, coords);

  
#ifdef YASK
  for(int_pt l_x = 0; l_x<nx+2*bdry_width; l_x++)
  {
    for(int_pt l_y = 0; l_y<ny+2*bdry_width; l_y++)
    {
      for(int_pt l_z = 0; l_z<nz+2*bdry_width; l_z++)
      {
        double l_cerj = cerjan.m_spongeCoeffX[l_x] * cerjan.m_spongeCoeffY[l_y] * cerjan.m_spongeCoeffZ[l_z];
        yask_context.sponge->writeElem(l_cerj, l_x, l_y, l_z, 0);
      }
    }
  }

  for (StencilBase *stencil : yask_stencils.stencils)
  {
        stencil->init(yask_context);
  }
  
#endif
  

  strideZ = 1;
  strideY = (size_z + 2*odc::constants::boundary);
  strideX = (size_y + 2*odc::constants::boundary) * strideY;

  lamMuStrideX = (size_y + 2*odc::constants::boundary);
}

void Patch::synchronize(bool allGrids)
{
  for(int dir_x=-1; dir_x<=1; dir_x++)
  {
    for(int dir_y=-1; dir_y<=1; dir_y++)
    {
      for(int dir_z=-1; dir_z<=1; dir_z++)
      {
        if(dir_x != 0 || dir_y != 0 || dir_z != 0)
          synchronize(dir_x, dir_y, dir_z, allGrids);
      }
    }
  }
}

void Patch::synchronize(int dir_x, int dir_y, int dir_z, bool allGrids)
{
  Patch* source_patch = neighbors[dir_x+1][dir_y+1][dir_z+1];

  if(!source_patch) // if there is no such neighbor, nothing to do
    return;
  
  int_pt h = bdry_width;

  int_pt x_start = h-1, x_end = size_x - h + 1; //remove -1,+1
  int_pt y_start = h-1, y_end = size_y - h + 1; //remove -1,+1
  int_pt z_start = h-1, z_end = size_z - h + 1; //remove -1,+1

  int_pt mx = 0, my = 0, mz = 0;

  if(dir_x < 0)
  {
    x_start = -1;//0;
    x_end = h;
    mx = size_x - 2*h;
  }
  else if(dir_x > 0)
  {
    x_start = size_x - h;
    x_end = size_x+1;//remove +1
    mx = -size_x + 2*h;
  }

  if(dir_y < 0)
  {
    y_start = -1;//0
    y_end = h;
    my = size_y - 2*h;
  }
  else if(dir_y > 0)
  {
    y_start = size_y - h;
    y_end = size_y+1;//remove +1
    my = -size_y + 2*h;
  }
  
  if(dir_z < 0)
  {
    z_start = -1;//0
    z_end = h;
    mz = size_z - 2*h;
  }
  else if(dir_z > 0)
  {
    z_start = size_z - h;
    z_end = size_z+1;//remove +1
    mz = -size_z + 2*h;
  }

  // TODO(Josh): we don't need the extra 1 in each direction, except for mesh at beginning
  //             maybe make new method just for mesh, and usually don't do the extra +- 1?
  for(int_pt x=x_start; x<x_end; x++)
  {
    for(int_pt y=y_start; y<y_end; y++)
    {
      for(int_pt z=z_start; z<z_end; z++)
      {
#ifdef YASK
        //TODO(Josh): implement patch synchronization for YASK data
#else
        soa.m_velocityX[x][y][z] = source_patch->soa.m_velocityX[x+mx][y+my][z+mz];
        soa.m_velocityY[x][y][z] = source_patch->soa.m_velocityY[x+mx][y+my][z+mz];
        soa.m_velocityZ[x][y][z] = source_patch->soa.m_velocityZ[x+mx][y+my][z+mz];

        soa.m_stressXX[x][y][z] = source_patch->soa.m_stressXX[x+mx][y+my][z+mz];        
        soa.m_stressXY[x][y][z] = source_patch->soa.m_stressXY[x+mx][y+my][z+mz];        
        soa.m_stressXZ[x][y][z] = source_patch->soa.m_stressXZ[x+mx][y+my][z+mz];        
        soa.m_stressYY[x][y][z] = source_patch->soa.m_stressYY[x+mx][y+my][z+mz];        
        soa.m_stressYZ[x][y][z] = source_patch->soa.m_stressYZ[x+mx][y+my][z+mz];        
        soa.m_stressZZ[x][y][z] = source_patch->soa.m_stressZZ[x+mx][y+my][z+mz];

        soa.m_memXX[x][y][z] = source_patch->soa.m_memXX[x+mx][y+my][z+mz];
        soa.m_memYY[x][y][z] = source_patch->soa.m_memYY[x+mx][y+my][z+mz];
        soa.m_memZZ[x][y][z] = source_patch->soa.m_memZZ[x+mx][y+my][z+mz];
        soa.m_memXY[x][y][z] = source_patch->soa.m_memXY[x+mx][y+my][z+mz];
        soa.m_memXZ[x][y][z] = source_patch->soa.m_memXZ[x+mx][y+my][z+mz];
        soa.m_memYZ[x][y][z] = source_patch->soa.m_memYZ[x+mx][y+my][z+mz];
#endif
        
        if(allGrids)
        {
#ifdef YASK
        //TODO(Josh): implement patch synchronization for YASK data
#else
          mesh.m_density[x][y][z] = source_patch->mesh.m_density[x+mx][y+my][z+mz];
          mesh.m_lam[x][y][z] = source_patch->mesh.m_lam[x+mx][y+my][z+mz];
          mesh.m_mu[x][y][z] = source_patch->mesh.m_mu[x+mx][y+my][z+mz];
#endif
          
          if(mesh.m_usingAnelastic)
          {
            mesh.m_qp[x][y][z] = source_patch->mesh.m_qp[x+mx][y+my][z+mz];
            mesh.m_qs[x][y][z] = source_patch->mesh.m_qs[x+mx][y+my][z+mz];
            mesh.m_tau1[x][y][z] = source_patch->mesh.m_tau1[x+mx][y+my][z+mz];
            mesh.m_tau2[x][y][z] = source_patch->mesh.m_tau2[x+mx][y+my][z+mz];
            mesh.m_weights[x][y][z] = source_patch->mesh.m_weights[x+mx][y+my][z+mz];
            mesh.m_weight_index[x][y][z] = source_patch->mesh.m_weight_index[x+mx][y+my][z+mz]; 
          }
        }
      }
      if(allGrids && mesh.m_usingAnelastic)
      {
        mesh.m_lam_mu[x][y][0] = source_patch->mesh.m_lam_mu[x+mx][y+my][0];
      }
    }
  }
  
  return;
}


Patch::~Patch()
{
#ifndef YASK
  soa.finalize();
#endif
  mesh.finalize();
  cerjan.finalize();
}
