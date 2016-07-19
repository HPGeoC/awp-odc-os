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


#if !defined(PATCHDECOMP_H)
#define PATCHDECOMP_H

#include "io/OptionParser.h"
#include "Patch.hpp"

class PatchDecomp
{
public:
  int_pt* m_idToGridX;
  int_pt* m_idToGridY;
  int_pt* m_idToGridZ;

  int*** m_coordToId; // TODO(Josh): should be int_pt for larger runs
  
  Patch* m_patches;

  int_pt m_patchXSize, m_patchYSize, m_patchZSize;
  int_pt m_numXPatches, m_numYPatches, m_numZPatches;
  int_pt m_numPatches;

  int_pt m_numXGridPoints,m_numYGridPoints,m_numZGridPoints;
  int_pt m_numGridPoints;

  int_pt m_overlapSize;
  
  void initialize(odc::io::OptionParser i_options, int_pt xSize, int_pt ySize, int_pt zSize,
                  int_pt xPatchSize, int_pt yPatchSize, int_pt zPatchSize,
                  int_pt overlapSize);

  void finalize();

  void synchronize(bool allGrids=false);
  
  int    globalToPatch(int_pt x, int_pt y, int_pt z);
  int_pt globalToLocalX(int_pt x, int_pt y, int_pt z);
  int_pt globalToLocalY(int_pt x, int_pt y, int_pt z);  
  int_pt globalToLocalZ(int_pt x, int_pt y, int_pt z);

  int_pt localToGlobalX(int_pt i_ptch, int_pt x, int_pt y, int_pt z);
  int_pt localToGlobalY(int_pt i_ptch, int_pt x, int_pt y, int_pt z);
  int_pt localToGlobalZ(int_pt i_ptch, int_pt x, int_pt y, int_pt z);

  real   getVelX(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep);
  real   getVelY(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep);
  real   getVelZ(int_pt i_x, int_pt i_y, int_pt i_z, int_pt i_timestep);
  
  void copyVelToBuffer(real* o_bufferX, real* o_bufferY, real* o_bufferZ,
                       int_pt i_firstX, int_pt i_lastX, int_pt i_skipX,
                       int_pt i_firstY, int_pt i_lastY, int_pt i_skipY,
                       int_pt i_firstZ, int_pt i_lastZ, int_pt i_skipZ,
                       int_pt i_timestep);

  // These functions return the max/min tmpvs, tmpvp and tmpdd over all patches
  real getVse(bool max);
  real getVpe(bool max);
  real getDde(bool max);
  
};

#endif
