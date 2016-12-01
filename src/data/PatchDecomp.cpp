/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 
 @section DESCRIPTION
 Patch decomposition data structure.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "parallel/Mpi.hpp"
#include "PatchDecomp.hpp"

#include <iostream>

void PatchDecomp::initialize(odc::io::OptionParser i_options, int_pt xSize, int_pt ySize, int_pt zSize,
                             int_pt xPatchSize, int_pt yPatchSize, int_pt zPatchSize,
                             int_pt overlapSize)
{
  m_patchXSize = xPatchSize;
  m_patchYSize = yPatchSize;
  m_patchZSize = zPatchSize;

  m_numXGridPoints = xSize;
  m_numYGridPoints = ySize;
  m_numZGridPoints = zSize;
  m_numGridPoints = m_numXGridPoints * m_numYGridPoints * m_numZGridPoints;

  m_overlapSize = overlapSize;

  m_numXPatches = (m_numXGridPoints + (m_patchXSize - 1)) / m_patchXSize;
  m_numYPatches = (m_numYGridPoints + (m_patchYSize - 1)) / m_patchYSize;
  m_numZPatches = (m_numZGridPoints + (m_patchZSize - 1)) / m_patchZSize;
  m_numPatches = m_numXPatches * m_numYPatches * m_numZPatches; 

  m_patches = new Patch[m_numPatches];


  m_idToGridX = new int_pt[m_numPatches];
  m_idToGridY = new int_pt[m_numPatches];
  m_idToGridZ = new int_pt[m_numPatches];
  
  m_coordToId = new int**[m_numXPatches];
  int curId = 0;
  for(int i=0; i<m_numXPatches; i++)
  {
    m_coordToId[i] = new int*[m_numYPatches];
    for(int j=0; j<m_numYPatches; j++)
    {
      m_coordToId[i][j] = new int[m_numZPatches];
      for(int k=0; k<m_numZPatches; k++)
      {
        m_idToGridX[curId] = i;
        m_idToGridY[curId] = j;
        m_idToGridZ[curId] = k;

        m_coordToId[i][j][k] = curId;

        curId++;
      }
    }
  }

  // Read mesh input file
  int nvar = i_options.m_nVar;
  int_pt num_pts = m_numXGridPoints * m_numYGridPoints * m_numZGridPoints;
  Grid1D inputBuffer = odc::data::Alloc1D(nvar * num_pts);
  FILE* file;
  file = fopen(i_options.m_inVel,"rb");
  if(!file)
  {
    std::cout << "can't open file " << i_options.m_inVel << std::endl;
    return;
  }
  if(!fread(inputBuffer,sizeof(float),nvar*num_pts,file))
  {
    std::cout << "can't read file " << i_options.m_inVel << std::endl;
    return;
  }

  
  for(int i=0; i<m_numPatches; i++)
  {
    int_pt grid_x = m_idToGridX[i];
    int_pt grid_y = m_idToGridY[i];
    int_pt grid_z = m_idToGridZ[i];    
    
    int_pt patch_start_x = grid_x * m_patchXSize;
    int_pt patch_start_y = grid_y * m_patchYSize;
    int_pt patch_start_z = grid_z * m_patchZSize;
    int_pt patch_end_x = (grid_x + 1) * m_patchXSize;
    int_pt patch_end_y = (grid_y + 1) * m_patchYSize;
    int_pt patch_end_z = (grid_z + 1) * m_patchZSize;

    if(patch_end_x > xSize)
      patch_end_x = xSize;
    if(patch_end_y > ySize)
      patch_end_y = ySize;
    if(patch_end_z > zSize)
      patch_end_z = zSize;

    int_pt buffer_offset = ( patch_start_z * m_numYGridPoints * m_numXGridPoints
                           + patch_start_y * m_numXGridPoints
                           + patch_start_x ) * nvar;
    m_patches[i].initialize(i_options,
                            patch_end_x - patch_start_x, patch_end_y - patch_start_y,
                            patch_end_z - patch_start_z, m_overlapSize, patch_start_x,
                            patch_start_y, patch_start_z, &inputBuffer[buffer_offset]);

    for(int j=-1; j<=1; j++)
    {
      for(int k=-1; k<=1; k++)
      {
        for(int l=-1; l<=1; l++)
        {
          if(grid_x + j >= 0 && grid_x + j < m_numXPatches
             && grid_y + k >= 0 && grid_y + k < m_numYPatches
             && grid_z + l >= 0 && grid_z + l < m_numZPatches)
          {
            m_patches[i].neighbors[j+1][k+1][l+1] = &m_patches[m_coordToId[grid_x+j][grid_y+k][grid_z+l]];
          }
        }
      }
    }    
    
  }


  odc::data::Delloc1D(inputBuffer);
}

void PatchDecomp::synchronize(bool allGrids)
{
  for(int i=0; i<m_numPatches; i++)
    m_patches[i].synchronize(allGrids);
}

void PatchDecomp::finalize()
{
  delete[] m_patches;

  for(int i=0; i<m_numXPatches; i++)
  {
    for(int j=0; j<m_numYPatches; j++)
    {
      delete[] m_coordToId[i][j];
    }
    delete[] m_coordToId[i];
  }
  delete[] m_coordToId;

  delete[] m_idToGridX;
  delete[] m_idToGridY;
  delete[] m_idToGridZ;  
}

int PatchDecomp::globalToPatch(int_pt x, int_pt y, int_pt z)
{
  int_pt patch_x = x / m_patchXSize;
  int_pt patch_y = y / m_patchYSize;
  int_pt patch_z = z / m_patchZSize;

  return m_coordToId[patch_x][patch_y][patch_z];
}

int_pt PatchDecomp::globalToLocalX(int_pt x, int_pt y, int_pt z)
{
  int patch_id = globalToPatch(x,y,z);
  return (x % m_patchXSize) + m_patches[patch_id].bdry_width;
}

int_pt PatchDecomp::globalToLocalY(int_pt x, int_pt y, int_pt z)
{
  int patch_id = globalToPatch(x,y,z);  
  return (y % m_patchYSize) + m_patches[patch_id].bdry_width;
}

int_pt PatchDecomp::globalToLocalZ(int_pt x, int_pt y, int_pt z)
{
  int patch_id = globalToPatch(x,y,z);  
  return (z % m_patchZSize) + m_patches[patch_id].bdry_width;
}

int_pt PatchDecomp::localToGlobalX(int_pt i_ptch, int_pt i_x, int_pt i_y, int_pt i_z)
{
  return m_idToGridX[i_ptch] * m_patchXSize + i_x;
}

int_pt PatchDecomp::localToGlobalY(int_pt i_ptch, int_pt i_x, int_pt i_y, int_pt i_z)
{
  return m_idToGridY[i_ptch] * m_patchYSize + i_y;
}

int_pt PatchDecomp::localToGlobalZ(int_pt i_ptch, int_pt i_x, int_pt i_y, int_pt i_z)
{
  return m_idToGridZ[i_ptch] * m_patchZSize + i_z;
}

real PatchDecomp::getVelX(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep)
{
  int_pt l_ptch = globalToPatch(i_x,i_y,i_z);
  int_pt l_locX = globalToLocalX(i_x,i_y,i_z);
  int_pt l_locY = globalToLocalY(i_x,i_y,i_z);
  int_pt l_locZ = globalToLocalZ(i_x,i_y,i_z);
  
#ifdef YASK
  return m_patches[l_ptch].yask_context.vel_x->readElem(i_timestep, l_locX, l_locY, l_locZ, 0);
#else
  return m_patches[l_ptch].soa.m_velocityX[l_locX][l_locY][l_locZ];
#endif
}

real PatchDecomp::getVelX(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.vel_x->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_velocityX[i_locX][i_locY][i_locZ];
#endif
}


real PatchDecomp::getVelY(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep)
{
  int_pt l_ptch = globalToPatch(i_x,i_y,i_z);
  int_pt l_locX = globalToLocalX(i_x,i_y,i_z);
  int_pt l_locY = globalToLocalY(i_x,i_y,i_z);
  int_pt l_locZ = globalToLocalZ(i_x,i_y,i_z);

#ifdef YASK
  return m_patches[l_ptch].yask_context.vel_y->readElem(i_timestep, l_locX, l_locY, l_locZ, 0);
#else
  return m_patches[l_ptch].soa.m_velocityY[l_locX][l_locY][l_locZ];
#endif
}

real PatchDecomp::getVelY(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.vel_y->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_velocityY[i_locX][i_locY][i_locZ];
#endif
}

real PatchDecomp::getVelZ(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep)
{
  int_pt l_ptch = globalToPatch(i_x,i_y,i_z);
  int_pt l_locX = globalToLocalX(i_x,i_y,i_z);
  int_pt l_locY = globalToLocalY(i_x,i_y,i_z);
  int_pt l_locZ = globalToLocalZ(i_x,i_y,i_z);

#ifdef YASK
  return m_patches[l_ptch].yask_context.vel_z->readElem(i_timestep, l_locX, l_locY, l_locZ, 0);
#else
  return m_patches[l_ptch].soa.m_velocityZ[l_locX][l_locY][l_locZ];
#endif
}

real PatchDecomp::getVelZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.vel_z->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_velocityZ[i_locX][i_locY][i_locZ];
#endif
}

void PatchDecomp::setVelX(real i_vel, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.vel_x->writeElem(i_vel, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_velocityX[i_locX][i_locY][i_locZ] = i_vel;
#endif
}

void PatchDecomp::setVelY(real i_vel, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.vel_y->writeElem(i_vel, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_velocityY[i_locX][i_locY][i_locZ] = i_vel;
#endif
}

void PatchDecomp::setVelZ(real i_vel, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.vel_z->writeElem(i_vel, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_velocityZ[i_locX][i_locY][i_locZ] = i_vel;
#endif
}

real PatchDecomp::getStressXX(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.stress_xx->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_stressXX[i_locX][i_locY][i_locZ];
#endif
}

real PatchDecomp::getStressXY(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.stress_xy->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_stressXY[i_locX][i_locY][i_locZ];
#endif
}

real PatchDecomp::getStressXZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.stress_xz->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_stressXZ[i_locX][i_locY][i_locZ];
#endif
}

real PatchDecomp::getStressYY(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.stress_yy->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_stressYY[i_locX][i_locY][i_locZ];
#endif
}

real PatchDecomp::getStressYZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.stress_yz->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_stressYZ[i_locX][i_locY][i_locZ];
#endif
}

real PatchDecomp::getStressZZ(int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  return m_patches[i_ptch].yask_context.stress_zz->readElem(i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  return m_patches[i_ptch].soa.m_stressZZ[i_locX][i_locY][i_locZ];
#endif
}

void PatchDecomp::setStressXX(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.stress_xx->writeElem(i_stress, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_stressXX[i_locX][i_locY][i_locZ] = i_stress;
#endif
}

void PatchDecomp::setStressXY(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.stress_xy->writeElem(i_stress, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_stressXY[i_locX][i_locY][i_locZ] = i_stress;
#endif
}

void PatchDecomp::setStressXZ(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.stress_xz->writeElem(i_stress, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_stressXZ[i_locX][i_locY][i_locZ] = i_stress;
#endif
}

void PatchDecomp::setStressYY(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.stress_yy->writeElem(i_stress, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_stressYY[i_locX][i_locY][i_locZ] = i_stress;
#endif
}

void PatchDecomp::setStressYZ(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.stress_yz->writeElem(i_stress, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_stressYZ[i_locX][i_locY][i_locZ] = i_stress;
#endif
}

void PatchDecomp::setStressZZ(real i_stress, int_pt i_ptch, int_pt i_locX, int_pt i_locY, int_pt i_locZ, int_pt i_timestep)
{
#ifdef YASK
  m_patches[i_ptch].yask_context.stress_zz->writeElem(i_stress, i_timestep, i_locX, i_locY, i_locZ, 0);
#else
  m_patches[i_ptch].soa.m_stressZZ[i_locX][i_locY][i_locZ] = i_stress;
#endif
}

void PatchDecomp::copyVelToBuffer(real* o_bufferX, real* o_bufferY, real* o_bufferZ,
                                  int_pt i_firstX, int_pt i_lastX, int_pt i_skipX,
                                  int_pt i_firstY, int_pt i_lastY, int_pt i_skipY,
                                  int_pt i_firstZ, int_pt i_lastZ, int_pt i_skipZ,
                                  int_pt i_timestep)
{
  // TODO(Josh): optimize this very slow code...
  int_pt bufInd = 0;
  for (int_pt iz = m_numZGridPoints - i_firstZ - 1; iz >= m_numZGridPoints - i_lastZ - 1; iz -= i_skipZ)
  {
    for (int_pt iy = i_firstY; iy <= i_lastY; iy += i_skipY)
    {
      for (int_pt ix = i_firstX; ix <= i_lastX; ix += i_skipX)
      {
        o_bufferX[bufInd] = getVelX(ix,iy,iz,i_timestep);
        o_bufferY[bufInd] = getVelY(ix,iy,iz,i_timestep);
        o_bufferZ[bufInd] = getVelZ(ix,iy,iz,i_timestep);
                    
        bufInd++;
      }
    }
  } 
}

void PatchDecomp::velMpiSynchronize(int i_dir, int_pt i_timestep)
{
  const int num_vel_grids = 3;
  int x_prev = 1, y_prev = 1, z_prev = 1, x_next = 1, y_next = 1, z_next = 1;
  
  if(i_dir == 0)
    x_prev = 0, x_next = 2;
  else if(i_dir == 1)
    y_prev = 0, y_next = 2;
  else
    z_prev = 0, z_next = 2;

  if(odc::parallel::Mpi::m_neighborRanks[x_prev][y_prev][z_prev] != -1)
    copyVelBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[x_prev][y_prev][z_prev],
					 -1, 0, 0, i_timestep+1);  

  if(odc::parallel::Mpi::m_neighborRanks[x_next][y_next][z_next] != -1)
    copyVelBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[x_next][y_next][z_next],
					 +1, 0, 0, i_timestep+1);  

  odc::parallel::Mpi::sendRecvBuffers(num_vel_grids,i_dir);

  if(odc::parallel::Mpi::m_neighborRanks[x_prev][y_prev][z_prev] != -1)
    copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[x_prev][y_prev][z_prev],
					 -1, 0, 0, i_timestep+1);  

  if(odc::parallel::Mpi::m_neighborRanks[x_next][y_next][z_next] != -1)
    copyVelBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[x_next][y_next][z_next],
					 +1, 0, 0, i_timestep+1);    
}

void PatchDecomp::stressMpiSynchronize(int i_dir, int_pt i_timestep)
{
  const int num_stress_grids = 6;  
  int x_prev = 1, y_prev = 1, z_prev = 1, x_next = 1, y_next = 1, z_next = 1;
  
  if(i_dir == 0)
    x_prev = 0, x_next = 2;
  else if(i_dir == 1)
    y_prev = 0, y_next = 2;
  else
    z_prev = 0, z_next = 2;

  if(odc::parallel::Mpi::m_neighborRanks[x_prev][y_prev][z_prev] != -1)
    copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[x_prev][y_prev][z_prev],
					 -1, 0, 0, i_timestep+1);  

  if(odc::parallel::Mpi::m_neighborRanks[x_next][y_next][z_next] != -1)
    copyStressBoundaryToBuffer(odc::parallel::Mpi::m_buffSend[x_next][y_next][z_next],
					 +1, 0, 0, i_timestep+1);  

  odc::parallel::Mpi::sendRecvBuffers(num_stress_grids,i_dir);

  if(odc::parallel::Mpi::m_neighborRanks[x_prev][y_prev][z_prev] != -1)
    copyStressBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[x_prev][y_prev][z_prev],
					 -1, 0, 0, i_timestep+1);  

  if(odc::parallel::Mpi::m_neighborRanks[x_next][y_next][z_next] != -1)
    copyStressBoundaryFromBuffer(odc::parallel::Mpi::m_buffRecv[x_next][y_next][z_next],
					 +1, 0, 0, i_timestep+1);    
}

void PatchDecomp::copyVelBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt timestep)
{
  int_pt startX = 0;
  int_pt endX = m_numXGridPoints;
  int_pt startY = 0;
  int_pt endY = m_numYGridPoints;
  int_pt startZ = 0;
  int_pt endZ = m_numZGridPoints;

  if(i_dirX == -1)
    endX = 2;
  if(i_dirX == 1)
    startX = endX-2;
  if(i_dirY == -1)
    endY = 2;
  if(i_dirY == 1)
    startY = endY-2;
  if(i_dirZ == -1)
    endZ = 2;
  if(i_dirZ == 1)
    startZ = endZ-2;

  
  // TODO(Josh): optimize this very slow code...
  int_pt bufInd = 0;
  for(int dim = 0; dim < 3; dim++)
  {
    for(int_pt ix=startX; ix<endX; ix++)
    {
      for(int_pt iy=startY; iy<endY; iy++) 
      {
        for(int_pt iz=startZ; iz<endZ; iz++)
        {
	  int patchId = globalToPatch(ix,iy,iz);
	  int_pt localX = (ix % m_patchXSize) + m_patches[patchId].bdry_width;
	  int_pt localY = (iy % m_patchYSize) + m_patches[patchId].bdry_width;
	  int_pt localZ = (iz % m_patchZSize) + m_patches[patchId].bdry_width;
	  real tmp;
	  if(dim == 0)
	    tmp = getVelX(patchId, localX, localY, localZ, timestep);
	  else if(dim == 1)
	    tmp = getVelY(patchId, localX, localY, localZ, timestep);
	  else
	    tmp = getVelZ(patchId, localX, localY, localZ, timestep);
          o_buffer[bufInd++] = tmp;
        }
      }
    }
  }
}

void PatchDecomp::copyVelBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
					  int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep)
{
  int_pt l_sizeX = m_numXGridPoints;
  int_pt l_sizeY = m_numYGridPoints;
  int_pt l_sizeZ = m_numZGridPoints;

  if(i_dirX == -1)
  {
    i_startX = 0;
    i_endX = 2;
  }
  if(i_dirX == 1)
  {
    i_startX = l_sizeX-2;
    i_endX = l_sizeX;
  }
  if(i_dirY == -1)
  {
    i_startY = 0;
    i_endY = 2;
  }
  if(i_dirY == 1)
  {
    i_startY = l_sizeY-2;
    i_endY = l_sizeY;
  }
  if(i_dirZ == -1)
  {
    i_startZ = 0;
    i_endZ = 2;
  }
  if(i_dirZ == 1)
  {
    i_startZ = l_sizeZ-2;
    i_endZ = l_sizeZ;
  }
  
  if(i_dirX)
    l_sizeX = 2;
  if(i_dirY)
    l_sizeY = 2;
  if(i_dirZ)
    l_sizeZ = 2;

  int_pt l_strideZ = 1;
  int_pt l_strideY = l_strideZ * l_sizeZ;
  int_pt l_strideX = l_strideY * l_sizeY;
  
  int_pt l_initialBufInd = 0;
  int_pt l_oneDimSize = 0;

  if(i_dirX)
  {
    l_initialBufInd = i_startY * l_strideY + i_startZ * l_strideZ;
  }
  if(i_dirY)
  {
    l_initialBufInd = i_startX * l_strideX + i_startZ * l_strideZ;
  }
  if(i_dirZ)
  {
    l_initialBufInd = i_startX * l_strideX + i_startY * l_strideY;
  }

  l_oneDimSize = l_sizeX * l_strideX;
  

  // Note(Josh): Assumes everything is within a single patch
  int l_patchId = globalToPatch(i_startX,i_startY,i_startZ);
  
  int_pt l_h = m_patches[l_patchId].bdry_width;
  
  i_startX += l_h;
  i_startY += l_h;
  i_startZ += l_h;
  
  i_endX += l_h;
  i_endY += l_h;
  i_endZ += l_h;

  int_pt l_skipY = l_strideY - (i_endZ - i_startZ);
  int_pt l_skipX = l_strideX - (i_endY - i_startY) * l_strideY;

  int_pt l_bufInd = l_initialBufInd;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getVelX(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getVelY(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 2 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getVelZ(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }
}


void PatchDecomp::copyVelBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt timestep)
{
  int_pt startX = 0;
  int_pt endX = m_numXGridPoints;
  int_pt startY = 0;
  int_pt endY = m_numYGridPoints;
  int_pt startZ = 0;
  int_pt endZ = m_numZGridPoints;

  // to determine patch ownership we will call globalToPatch, which doesn't
  // understand indices corresponding to padding; so for such indices we shift a little
  int_pt patchShiftX = 0;
  int_pt patchShiftY = 0;
  int_pt patchShiftZ = 0;  

  if(i_dirX == -1)
  {
    startX = -2;
    endX = 0;
    patchShiftX = 2;
  }
  if(i_dirX == 1)
  {
    startX = endX;
    endX += 2;
    patchShiftX = -2;
  }
  if(i_dirY == -1)
  {
    startY = -2;
    endY = 0;
    patchShiftY = 2;
  }
  if(i_dirY == 1)
  {
    startY = endY;
    endY += 2;
    patchShiftY = -2;
  }
  if(i_dirZ == -1)
  {
    startZ = -2;
    endZ = 0;
    patchShiftZ = 2;
  }
  if(i_dirZ == 1)
  {
    startZ = endZ;
    endZ += 2;
    patchShiftZ = -2;
  }
  
  // TODO(Josh): optimize this very slow code...
  int_pt bufInd = 0;
  for(int dim = 0; dim < 3; dim++)
  {
    for(int_pt ix=startX; ix<endX; ix++)
    {
      for(int_pt iy=startY; iy<endY; iy++) 
      {
        for(int_pt iz=startZ; iz<endZ; iz++)
        {
	  int patchId = globalToPatch(ix+patchShiftX,iy+patchShiftY,iz+patchShiftZ);
	  int_pt localX = ((ix+patchShiftX) % m_patchXSize) + m_patches[patchId].bdry_width - patchShiftX;
	  int_pt localY = ((iy+patchShiftY) % m_patchYSize) + m_patches[patchId].bdry_width - patchShiftY;
	  int_pt localZ = ((iz+patchShiftZ) % m_patchZSize) + m_patches[patchId].bdry_width - patchShiftZ;
	  real tmp = o_buffer[bufInd++];
	  if(dim == 0)
	    setVelX(tmp, patchId, localX, localY, localZ, timestep);
	  else if(dim == 1)
	    setVelY(tmp, patchId, localX, localY, localZ, timestep);
	  else
	    setVelZ(tmp, patchId, localX, localY, localZ, timestep);
        }
      }
    }
  }
}

void PatchDecomp::copyVelBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
					  int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep)
{
  int_pt l_sizeX = m_numXGridPoints;
  int_pt l_sizeY = m_numYGridPoints;
  int_pt l_sizeZ = m_numZGridPoints;

  int_pt l_patchShiftX = 0;
  int_pt l_patchShiftY = 0;
  int_pt l_patchShiftZ = 0;  
  
  if(i_dirX == -1)
  {
    i_startX = -2;
    i_endX = 0;
    l_patchShiftX = 2;
  }
  if(i_dirX == 1)
  {
    i_startX = l_sizeX;
    i_endX = l_sizeX+2;
    l_patchShiftX = -2;    
  }
  if(i_dirY == -1)
  {
    i_startY = -2;
    i_endY = 0;
    l_patchShiftY = 2;
  }
  if(i_dirY == 1)
  {
    i_startY = l_sizeY;
    i_endY = l_sizeY+2;
    l_patchShiftY = -2;
  }
  if(i_dirZ == -1)
  {
    i_startZ = -2;
    i_endZ = 0;
    l_patchShiftZ = 2;
  }
  if(i_dirZ == 1)
  {
    i_startZ = l_sizeZ;
    i_endZ = l_sizeZ+2;
    l_patchShiftZ = -2;
  }
  
  if(i_dirX)
    l_sizeX = 2;
  if(i_dirY)
    l_sizeY = 2;
  if(i_dirZ)
    l_sizeZ = 2;

  int_pt l_strideZ = 1;
  int_pt l_strideY = l_strideZ * l_sizeZ;
  int_pt l_strideX = l_strideY * l_sizeY;
  
  int_pt l_initialBufInd = 0;
  int_pt l_oneDimSize = 0;

  if(i_dirX)
  {
    l_initialBufInd = i_startY * l_strideY + i_startZ * l_strideZ;
  }
  if(i_dirY)
  {
    l_initialBufInd = i_startX * l_strideX + i_startZ * l_strideZ;
  }
  if(i_dirZ)
  {
    l_initialBufInd = i_startX * l_strideX + i_startY * l_strideY;
  }

  l_oneDimSize = l_sizeX * l_strideX;
  

  // Note(Josh): Assumes everything is within a single patch
  int l_patchId = globalToPatch(i_startX+l_patchShiftX,i_startY+l_patchShiftY,i_startZ+l_patchShiftZ);
  
  int_pt l_h = m_patches[l_patchId].bdry_width;
  
  i_startX += l_h;
  i_startY += l_h;
  i_startZ += l_h;
  
  i_endX += l_h;
  i_endY += l_h;
  i_endZ += l_h;

  int_pt l_skipY = l_strideY - (i_endZ - i_startZ);
  int_pt l_skipX = l_strideX - (i_endY - i_startY) * l_strideY;

  int_pt l_bufInd = l_initialBufInd;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setVelX(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setVelY(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 2 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setVelZ(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }
}


void PatchDecomp::copyStressBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt timestep)
{
  int_pt startX = 0;
  int_pt endX = m_numXGridPoints;
  int_pt startY = 0;
  int_pt endY = m_numYGridPoints;
  int_pt startZ = 0;
  int_pt endZ = m_numZGridPoints;

  if(i_dirX == -1)
    endX = 2;
  if(i_dirX == 1)
    startX = endX-2;
  if(i_dirY == -1)
    endY = 2;
  if(i_dirY == 1)
    startY = endY-2;
  if(i_dirZ == -1)
    endZ = 2;
  if(i_dirZ == 1)
    startZ = endZ-2;

  
  // TODO(Josh): optimize this very slow code...
  int_pt bufInd = 0;
  for(int dim = 0; dim < 6; dim++)
  {
    for(int_pt ix=startX; ix<endX; ix++)
    {
      for(int_pt iy=startY; iy<endY; iy++) 
      {
        for(int_pt iz=startZ; iz<endZ; iz++)
        {
	  int patchId = globalToPatch(ix,iy,iz);
	  int_pt localX = (ix % m_patchXSize) + m_patches[patchId].bdry_width;
	  int_pt localY = (iy % m_patchYSize) + m_patches[patchId].bdry_width;
	  int_pt localZ = (iz % m_patchZSize) + m_patches[patchId].bdry_width;
	  real tmp;
	  if(dim == 0)
	    tmp = getStressXX(patchId, localX, localY, localZ, timestep);
	  else if(dim == 1)
	    tmp = getStressXY(patchId, localX, localY, localZ, timestep);
	  else if(dim == 2)
	    tmp = getStressXZ(patchId, localX, localY, localZ, timestep);
	  else if(dim == 3)
	    tmp = getStressYY(patchId, localX, localY, localZ, timestep);
	  else if(dim == 4)
	    tmp = getStressYZ(patchId, localX, localY, localZ, timestep);
	  else
	    tmp = getStressZZ(patchId, localX, localY, localZ, timestep);
          o_buffer[bufInd++] = tmp;
        }
      }
    }
  }
}

void PatchDecomp::copyStressBoundaryToBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
					  int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep)
{
  int_pt l_sizeX = m_numXGridPoints;
  int_pt l_sizeY = m_numYGridPoints;
  int_pt l_sizeZ = m_numZGridPoints;

  if(i_dirX == -1)
  {
    i_startX = 0;
    i_endX = 2;
  }
  if(i_dirX == 1)
  {
    i_startX = l_sizeX-2;
    i_endX = l_sizeX;
  }
  if(i_dirY == -1)
  {
    i_startY = 0;
    i_endY = 2;
  }
  if(i_dirY == 1)
  {
    i_startY = l_sizeY-2;
    i_endY = l_sizeY;
  }
  if(i_dirZ == -1)
  {
    i_startZ = 0;
    i_endZ = 2;
  }
  if(i_dirZ == 1)
  {
    i_startZ = l_sizeZ-2;
    i_endZ = l_sizeZ;
  }
  
  if(i_dirX)
    l_sizeX = 2;
  if(i_dirY)
    l_sizeY = 2;
  if(i_dirZ)
    l_sizeZ = 2;

  int_pt l_strideZ = 1;
  int_pt l_strideY = l_strideZ * l_sizeZ;
  int_pt l_strideX = l_strideY * l_sizeY;
  
  int_pt l_initialBufInd = 0;
  int_pt l_oneDimSize = 0;

  if(i_dirX)
  {
    l_initialBufInd = i_startY * l_strideY + i_startZ * l_strideZ;
  }
  if(i_dirY)
  {
    l_initialBufInd = i_startX * l_strideX + i_startZ * l_strideZ;
  }
  if(i_dirZ)
  {
    l_initialBufInd = i_startX * l_strideX + i_startY * l_strideY;
  }

  l_oneDimSize = l_sizeX * l_strideX;
  

  // Note(Josh): Assumes everything is within a single patch
  int l_patchId = globalToPatch(i_startX,i_startY,i_startZ);
  
  int_pt l_h = m_patches[l_patchId].bdry_width;
  
  i_startX += l_h;
  i_startY += l_h;
  i_startZ += l_h;
  
  i_endX += l_h;
  i_endY += l_h;
  i_endZ += l_h;

  int_pt l_skipY = l_strideY - (i_endZ - i_startZ);
  int_pt l_skipX = l_strideX - (i_endY - i_startY) * l_strideY;

  int_pt l_bufInd = l_initialBufInd;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getStressXX(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getStressXY(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 2 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getStressXZ(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 3 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getStressYY(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 4 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getStressYZ(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 5 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	o_buffer[l_bufInd] = getStressZZ(l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }
}


void PatchDecomp::copyStressBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt timestep)
{
  int_pt startX = 0;
  int_pt endX = m_numXGridPoints;
  int_pt startY = 0;
  int_pt endY = m_numYGridPoints;
  int_pt startZ = 0;
  int_pt endZ = m_numZGridPoints;

  // to determine patch ownership we will call globalToPatch, which doesn't
  // understand indices corresponding to padding; so for such indices we shift a little
  int_pt patchShiftX = 0;
  int_pt patchShiftY = 0;
  int_pt patchShiftZ = 0;  

  if(i_dirX == -1)
  {
    startX = -2;
    endX = 0;
    patchShiftX = 2;
  }
  if(i_dirX == 1)
  {
    startX = endX;
    endX += 2;
    patchShiftX = -2;
  }
  if(i_dirY == -1)
  {
    startY = -2;
    endY = 0;
    patchShiftY = 2;
  }
  if(i_dirY == 1)
  {
    startY = endY;
    endY += 2;
    patchShiftY = -2;
  }
  if(i_dirZ == -1)
  {
    startZ = -2;
    endZ = 0;
    patchShiftZ = 2;
  }
  if(i_dirZ == 1)
  {
    startZ = endZ;
    endZ += 2;
    patchShiftZ = -2;
  }
  
  // TODO(Josh): optimize this very slow code...
  int_pt bufInd = 0;
  for(int dim = 0; dim < 6; dim++)
  {
    for(int_pt ix=startX; ix<endX; ix++)
    {
      for(int_pt iy=startY; iy<endY; iy++) 
      {
        for(int_pt iz=startZ; iz<endZ; iz++)
        {
	  int patchId = globalToPatch(ix+patchShiftX,iy+patchShiftY,iz+patchShiftZ);
	  int_pt localX = ((ix+patchShiftX) % m_patchXSize) + m_patches[patchId].bdry_width - patchShiftX;
	  int_pt localY = ((iy+patchShiftY) % m_patchYSize) + m_patches[patchId].bdry_width - patchShiftY;
	  int_pt localZ = ((iz+patchShiftZ) % m_patchZSize) + m_patches[patchId].bdry_width - patchShiftZ;
	  real tmp = o_buffer[bufInd++];
	  if(dim == 0)
	    setStressXX(tmp, patchId, localX, localY, localZ, timestep);
	  else if(dim == 1)
	    setStressXY(tmp, patchId, localX, localY, localZ, timestep);
	  else if(dim == 2)
	    setStressXZ(tmp, patchId, localX, localY, localZ, timestep);
	  else if(dim == 3)
	    setStressYY(tmp, patchId, localX, localY, localZ, timestep);
	  else if(dim == 4)
	    setStressYZ(tmp, patchId, localX, localY, localZ, timestep);
	  else
	    setStressZZ(tmp, patchId, localX, localY, localZ, timestep);
        }
      }
    }
  }
}

void PatchDecomp::copyStressBoundaryFromBuffer(real* o_buffer, int i_dirX, int i_dirY,
			                  int i_dirZ, int_pt i_startX, int_pt i_startY, int_pt i_startZ,
					  int_pt i_endX, int_pt i_endY, int_pt i_endZ, int_pt i_timestep)
{
  int_pt l_sizeX = m_numXGridPoints;
  int_pt l_sizeY = m_numYGridPoints;
  int_pt l_sizeZ = m_numZGridPoints;

  int_pt l_patchShiftX = 0;
  int_pt l_patchShiftY = 0;
  int_pt l_patchShiftZ = 0;  
  
  if(i_dirX == -1)
  {
    i_startX = -2;
    i_endX = 0;
    l_patchShiftX = 2;
  }
  if(i_dirX == 1)
  {
    i_startX = l_sizeX;
    i_endX = l_sizeX+2;
    l_patchShiftX = -2;    
  }
  if(i_dirY == -1)
  {
    i_startY = -2;
    i_endY = 0;
    l_patchShiftY = 2;
  }
  if(i_dirY == 1)
  {
    i_startY = l_sizeY;
    i_endY = l_sizeY+2;
    l_patchShiftY = -2;
  }
  if(i_dirZ == -1)
  {
    i_startZ = -2;
    i_endZ = 0;
    l_patchShiftZ = 2;
  }
  if(i_dirZ == 1)
  {
    i_startZ = l_sizeZ;
    i_endZ = l_sizeZ+2;
    l_patchShiftZ = -2;
  }
  
  if(i_dirX)
    l_sizeX = 2;
  if(i_dirY)
    l_sizeY = 2;
  if(i_dirZ)
    l_sizeZ = 2;

  int_pt l_strideZ = 1;
  int_pt l_strideY = l_strideZ * l_sizeZ;
  int_pt l_strideX = l_strideY * l_sizeY;
  
  int_pt l_initialBufInd = 0;
  int_pt l_oneDimSize = 0;

  if(i_dirX)
  {
    l_initialBufInd = i_startY * l_strideY + i_startZ * l_strideZ;
  }
  if(i_dirY)
  {
    l_initialBufInd = i_startX * l_strideX + i_startZ * l_strideZ;
  }
  if(i_dirZ)
  {
    l_initialBufInd = i_startX * l_strideX + i_startY * l_strideY;
  }

  l_oneDimSize = l_sizeX * l_strideX;
  

  // Note(Josh): Assumes everything is within a single patch
  int l_patchId = globalToPatch(i_startX+l_patchShiftX,i_startY+l_patchShiftY,i_startZ+l_patchShiftZ);
  
  int_pt l_h = m_patches[l_patchId].bdry_width;
  
  i_startX += l_h;
  i_startY += l_h;
  i_startZ += l_h;
  
  i_endX += l_h;
  i_endY += l_h;
  i_endZ += l_h;

  int_pt l_skipY = l_strideY - (i_endZ - i_startZ);
  int_pt l_skipX = l_strideX - (i_endY - i_startY) * l_strideY;

  int_pt l_bufInd = l_initialBufInd;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setStressXX(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setStressXY(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 2 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setStressXZ(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 3 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setStressYY(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }

  l_bufInd = l_initialBufInd + 4 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setStressYZ(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }
  
  l_bufInd = l_initialBufInd + 5 * l_oneDimSize;
  for(int_pt ix=i_startX; ix<i_endX; ix++)
  {
    for(int_pt iy=i_startY; iy<i_endY; iy++)
    {
      for(int_pt iz=i_startZ; iz<i_endZ; iz++)
      {
	setStressZZ(o_buffer[l_bufInd], l_patchId, ix, iy, iz, i_timestep);
	l_bufInd += l_strideZ;
      }
      l_bufInd += l_skipY;      
    }
    l_bufInd += l_skipX;
  }
}

real PatchDecomp::getVse(bool max)
{
  double min_vse = m_patches[0].mesh.m_vse[0];
  double max_vse = m_patches[0].mesh.m_vse[1];

  for(int i=1; i<m_numPatches; i++)
  {
    if(m_patches[i].mesh.m_vse[0] < min_vse)
      min_vse = m_patches[i].mesh.m_vse[0];
    if(m_patches[i].mesh.m_vse[1] > max_vse)
      max_vse = m_patches[i].mesh.m_vse[1];    
  }

  if(max)
    return max_vse;
  else
    return min_vse;
}

real PatchDecomp::getVpe(bool max)
{
  double min_vpe = m_patches[0].mesh.m_vpe[0];
  double max_vpe = m_patches[0].mesh.m_vpe[1];

  for(int i=1; i<m_numPatches; i++)
  {
    if(m_patches[i].mesh.m_vpe[0] < min_vpe)
      min_vpe = m_patches[i].mesh.m_vpe[0];
    if(m_patches[i].mesh.m_vpe[1] > max_vpe)
      max_vpe = m_patches[i].mesh.m_vpe[1];    
  }

  if(max)
    return max_vpe;
  else
    return min_vpe;
}

real PatchDecomp::getDde(bool max)
{
  double min_dde = m_patches[0].mesh.m_dde[0];
  double max_dde = m_patches[0].mesh.m_dde[1];

  for(int i=1; i<m_numPatches; i++)
  {
    if(m_patches[i].mesh.m_dde[0] < min_dde)
      min_dde = m_patches[i].mesh.m_dde[0];
    if(m_patches[i].mesh.m_dde[1] > max_dde)
      max_dde = m_patches[i].mesh.m_dde[1];    
  }

  if(max)
    return max_dde;
  else
    return min_dde;
}

