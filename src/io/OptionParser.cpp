/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Parser for command line options.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "OptionParser.h"

#include <iostream>

extern "C" {
  void command( int argc,    char **argv,
	              float *TMAX, float *DH,       float *DT,   float *ARBC,    float *PHT,
                int *NPC,    int *ND,         int *NSRC,   int *NST,       int *NVAR,
                int *NVE,    int *MEDIASTART, int *IFAULT, int *READ_STEP, int *READ_STEP_GPU,
                int *NTISKP, int *WRITE_STEP,
                int *NX,     int *NY,         int *NZ,     int *PX,        int *PY,
                int *NBGX,   int *NEDX,       int *NSKPX, 
                int *NBGY,   int *NEDY,       int *NSKPY, 
                int *NBGZ,   int *NEDZ,       int *NSKPZ, 
                float *FAC,  float *Q0,       float *EX,   float *FP,      int *IDYNA,     int *SoCalQ,
                char *INSRC, char *INVEL,     char *OUT,   char *INSRC_I2, char *CHKFILE);
}

odc::io::OptionParser::OptionParser( int i_argc, char **i_argv ) {
  // TODO: Log
  std::cout << "parsing command line options" << std::endl;

  // initialize options from command line arguments
  command( i_argc, i_argv,
           &m_tMax,
           &m_dH,
           &m_dT,
           &m_arbc,
           &m_pht,
           &m_nPc,
           &m_nD,
           &m_nSrc,
           &m_nSt,
           &m_nVar,
           &m_nVe,
           &m_mediaStart,
           &m_iFault,
           &m_readStep, &m_readStepGpu,
           &m_nTiSkp,
           &m_writeStep,
           &m_nX, &m_nY, &m_nZ,
           &m_pX, &m_pY,
           &m_nBgX, &m_nEdX, &m_nSkpX,
           &m_nBgY, &m_nEdY, &m_nSkpY,
           &m_nBgZ, &m_nEdZ, &m_nSkpZ,
           &m_fac,
           &m_q0,
           &m_ex,
           &m_fp,
           &m_iDyna,
           &m_soCalQ,
            m_inSrc,
            m_inVel,
            m_out,
            m_inSrcI2,
            m_chkFile );
    
    m_numTimesteps = (int_pt)((m_tMax/m_dT) + 1);

  // print options
  // TODO: Log
  std::cout << "parsed options successfully:"            << std::endl
            << "\t" << "timesteps:\t\t" << m_numTimesteps<< std::endl
            << "\t" << "tMax:\t\t"      << m_tMax        << std::endl
            << "\t" << "dH:\t\t"        << m_dH          << std::endl
            << "\t" << "dT:\t\t"        << m_dT          << std::endl
            << "\t" << "arbc:\t\t"      << m_arbc        << std::endl
            << "\t" << "pht:\t\t"       << m_pht         << std::endl
            << "\t" << "nPc:\t\t"       << m_nPc         << std::endl
            << "\t" << "nD:\t\t"        << m_nD          << std::endl
            << "\t" << "nSrc:\t\t"      << m_nSrc        << std::endl
            << "\t" << "nSt:\t\t"       << m_nSt         << std::endl
            << "\t" << "nVar:\t\t"      << m_nVar        << std::endl
            << "\t" << "nVe:\t\t"       << m_nVe         << std::endl
            << "\t" << "mediaStart:\t"  << m_mediaStart  << std::endl
            << "\t" << "iFault:\t\t"    << m_iFault      << std::endl
            << "\t" << "readStep:\t"    << m_readStep    << std::endl
            << "\t" << "readStepGpu:\t" << m_readStepGpu << std::endl
            << "\t" << "nTiSkp:\t\t"    << m_nTiSkp      << std::endl
            << "\t" << "writeStep:\t"   << m_writeStep   << std::endl
            << "\t" << "nX:\t\t"        << m_nX          << std::endl
            << "\t" << "nY:\t\t"        << m_nY          << std::endl
            << "\t" << "nZ:\t\t"        << m_nZ          << std::endl
            << "\t" << "pX:\t\t"        << m_pX          << std::endl
            << "\t" << "pY:\t\t"        << m_pY          << std::endl
            << "\t" << "nBgX:\t\t"      << m_nBgX        << std::endl
            << "\t" << "nEdX:\t\t"      << m_nEdX        << std::endl
            << "\t" << "nSkpX:\t\t"     << m_nSkpX       << std::endl
            << "\t" << "nBgY:\t\t"      << m_nBgY        << std::endl
            << "\t" << "nEdY:\t\t"      << m_nEdY        << std::endl
            << "\t" << "nSkpY:\t\t"     << m_nSkpY       << std::endl
            << "\t" << "nBgZ:\t\t"      << m_nBgZ        << std::endl
            << "\t" << "nEdZ:\t\t"      << m_nEdZ        << std::endl
            << "\t" << "nSkpZ:\t\t"     << m_nSkpZ       << std::endl
            << "\t" << "fac:\t\t"       << m_fac         << std::endl
            << "\t" << "q0:\t\t"        << m_q0          << std::endl
            << "\t" << "ex:\t\t"        << m_ex          << std::endl
            << "\t" << "fp:\t\t"        << m_fp          << std::endl
            << "\t" << "iDyna:\t\t"     << m_iDyna       << std::endl
            << "\t" << "soCalQ:\t\t"    << m_soCalQ      << std::endl
            << "\t" << "inSrc:\t\t"     << m_inSrc       << std::endl
            << "\t" << "inVel:\t\t"     << m_inVel       << std::endl
            << "\t" << "out:\t\t"       << m_out         << std::endl
            << "\t" << "inSrcI2:\t"     << m_inSrcI2     << std::endl
            << "\t" << "chkFile:\t"     << m_chkFile     << std::endl;
    
    
}
