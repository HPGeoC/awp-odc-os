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

    std::cout << "about to initialize patch..." << std::endl;
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

real PatchDecomp::getVelX(int_pt i_x, int_pt i_y, int_pt i_z)
{
  int_pt l_ptch = globalToPatch(i_x,i_y,i_z);
  int_pt l_locX = globalToLocalX(i_x,i_y,i_z);
  int_pt l_locY = globalToLocalY(i_x,i_y,i_z);
  int_pt l_locZ = globalToLocalZ(i_x,i_y,i_z);

  return m_patches[l_ptch].soa.m_velocityX[l_locX][l_locY][l_locZ];
}

real PatchDecomp::getVelY(int_pt i_x, int_pt i_y, int_pt i_z)
{
  int_pt l_ptch = globalToPatch(i_x,i_y,i_z);
  int_pt l_locX = globalToLocalX(i_x,i_y,i_z);
  int_pt l_locY = globalToLocalY(i_x,i_y,i_z);
  int_pt l_locZ = globalToLocalZ(i_x,i_y,i_z);

  return m_patches[l_ptch].soa.m_velocityY[l_locX][l_locY][l_locZ];
}

real PatchDecomp::getVelZ(int_pt i_x, int_pt i_y, int_pt i_z)
{
  int_pt l_ptch = globalToPatch(i_x,i_y,i_z);
  int_pt l_locX = globalToLocalX(i_x,i_y,i_z);
  int_pt l_locY = globalToLocalY(i_x,i_y,i_z);
  int_pt l_locZ = globalToLocalZ(i_x,i_y,i_z);

  return m_patches[l_ptch].soa.m_velocityZ[l_locX][l_locY][l_locZ];
}

void PatchDecomp::copyVelToBuffer(real* o_bufferX, real* o_bufferY, real* o_bufferZ,
                                  int_pt i_firstX, int_pt i_lastX, int_pt i_skipX,
                                  int_pt i_firstY, int_pt i_lastY, int_pt i_skipY,
                                  int_pt i_firstZ, int_pt i_lastZ, int_pt i_skipZ)
{
  // TODO(Josh): optimize this very slow code...
  int_pt bufInd = 0;
  for (int_pt iz = m_numZGridPoints - i_firstZ - 1; iz >= m_numZGridPoints - i_lastZ; iz -= i_skipZ)
  {
    for (int_pt iy = i_firstY; iy <= i_lastY; iy += i_skipY)
    {
      for (int_pt ix = i_firstX; ix <= i_lastX; ix += i_skipX)
      {
        o_bufferX[bufInd] = getVelX(ix,iy,iz);
        o_bufferY[bufInd] = getVelY(ix,iy,iz);
        o_bufferZ[bufInd] = getVelZ(ix,iy,iz);
                    
        bufInd++;
      }
    }
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

