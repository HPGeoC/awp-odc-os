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


#if !defined(PATCH_H)

#include "constants.hpp"
#include "io/OptionParser.h"
#include "data/Grid.hpp"
#include "data/SoA.hpp"
#include "data/Mesh.hpp"
#include "data/Cerjan.hpp"
#include "parallel/Mpi.hpp"

#ifdef YASK
#include "yask/stencil.hpp"
#include "yask/stencil_calc.hpp"
#include "yask/stencil_code.hpp"

using namespace yask;
#endif


class Patch
{
public:
  Patch();
  Patch(int_pt _nx, int_pt _ny, int_pt _nz, int_pt _bw);
  ~Patch();

  void initialize(odc::io::OptionParser i_options, int_pt _nx, int_pt _ny, int_pt _nz, int_pt _bw,
                  int_pt i_globalX, int_pt i_globalY, int_pt i_globalZ, Grid1D i_inputBuffer);
  
  int_pt nx, ny, nz;
  int_pt bdry_width;

  int_pt strideX, strideY, strideZ;
  int_pt lamMuStrideX; 
  
  int_pt size_x, size_y, size_z;

  void synchronize(bool allGrids);
  void synchronize(int dir_x, int dir_y, int dir_z , bool allGrids);  

  Patch* neighbors[3][3][3];
  
  odc::data::SoA soa;
  odc::data::Mesh mesh;
  odc::data::Cerjan cerjan;

  #ifdef YASK
  STENCIL_EQUATIONS yask_stencils;
  STENCIL_CONTEXT yask_context;
  #endif

};

#define PATCH_H
#endif
