/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Parser for runtime options.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OPTION_PARSER_H_
#define OPTION_PARSER_H_

#include "constants.hpp"

namespace odc {
  namespace io {
    class OptionParser;
  }
}

class odc::io::OptionParser {
  //private:

  public:
    // Total simulation time in seconds.
    real m_tMax;

    // Number of timesteps
    int m_numTimesteps;
    
    // Spatial step size for x, y, and z dimensions in meters.
    real m_dH;

    // Time step size in seconds.
    real m_dT;

    // ARBC               Coefficient for PML (3-4), or Cerjan (0.90-0.96).
    real m_arbc;

    // TODO: missing description
    real m_pht;

    // PML or Cerjan ABC (1=PML, 0=Cerjan).
    int m_nPc;

    // ABC thickness (grid-points) PML <= 20, Cerjan >= 20.
    int m_nD;

    // Number of source nodes on fault.
    int m_nSrc;

    // Number of time steps in rupture functions.
    int m_nSt;

    // Number of variables in a grid point.
    int m_nVar;

    // Visco or elastic scheme (1=visco, 0=elastic).
    int m_nVe;

    // Initial media restart option(0=homogenous).
    int m_mediaStart;

    // Mode selection and fault or initial stress setting (1 or 2).
    int m_iFault;

    // Number of rupture timesteps to read at a time from the source file.
    int m_readStep;

    // CPU reads larger chunks and sends to GPU at every @c READ_STEP_GPU. 
    // If @c IFAULT==2 then @c READ_STEP must be divisible by @c READ_STEP_GPU.
    int m_readStepGpu;

    // Number of timesteps to skip when copying velocities from GPU to CPU.
    int m_nTiSkp;

    // Number of timesteps to skip when writing velocities from CPU to files. So the timesteps that get written to the files are
    // ( @c n*NTISKP*WRITE_STEP for @c n=1,2,...).
    int m_writeStep;

    // Number of nodes in the x dimension.
    int m_nX;

    // Number of nodes in the y dimension.
    int m_nY;

    // Number of nodes in the z dimension.
    int m_nZ;

    // Number of processes in the x dimension (using 2 dimensional MPI topology).
    int m_pX;

    // Number of processes in the y dimension (using 2 dimensional MPI topology).
    int m_pY;

    // Index (starting from 1) of the first x node to record values at (e.g. if @c NBGX==10, then the output file
    // will not have data for the first 9 nodes in the x dimension).
    int m_nBgX;

    // Index (starting from 1) of the last x node to record values at. Set to -1 to record all the way to the end.
    int m_nEdX;

    // Number of nodes to skip in the x dimension when recording values. (e.g. if @c NBGX==10, @c NEDX==40, @c NSKPX==10, then
    // x nodes 10, 20, 30, and 40 will have their values recorded in the output file.
    int m_nSkpX;

    // Index (starting from 1) of the first y node to record values at.
    int m_nBgY;

    // Index (starting from 1) of the last y node to record values at.
    int m_nEdY;

    // Number of nodes to skip in the y dimension when recording values.
    int m_nSkpY;

    // Index (starting from 1) of the first x node to record values at. Note that z==1 is the surface node.
    int m_nBgZ;

    // Index (starting from 1) of the last z node to record values at.
    int m_nEdZ;

    // Number of nodes to skip in the z dimension when recording values.
    int m_nSkpZ;

    // TODO: missing description
    real m_fac;

    // TODO: missing description
    real m_q0;

    // TODO: missing description
    real m_ex;

    // Q bandwidth central frequency.
    real m_fp;

    // IMode selection of dynamic rupture model.
    int m_iDyna;

    // Southern California Vp-Vs Q relationship enabling flag.
    int m_soCalQ;

    // Source input file (if @c IFAULT==2, then this is prefix of @c tpsrc).
    char m_inSrc[50];

    // Mesh input file.
    char m_inVel[50];

    // Output folder.
    char m_out[50];

    // Split source input file prefix for @c IFAULT==2 option.
    char m_inSrcI2[50];

    // Checkpoint statistics file to write to.
    char m_chkFile[50];

    /**
     * Constructor of the option parser.
     **/
    OptionParser( int i_argc, char **i_argv );
};

#endif
