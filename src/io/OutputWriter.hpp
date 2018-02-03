/**
 @author Josh Tobin (rjtobin AT ucsd.edu)
 @author Rajdeep Konwar (rkonwar AT ucsd.edu)
 
 @section DESCRIPTION
 Output writer.

 @section LICENSE
 Copyright (c) 2015-2018, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OutputWriter_hpp
#define OutputWriter_hpp

#include "parallel/Mpi.hpp"

#include <cstring>
#include <cstdio>

#include "io/OptionParser.h"

#include "data/common.hpp"
#include "data/SoA.hpp"
#include "data/PatchDecomp.hpp"

namespace odc {
  namespace io {
    class OutputWriter;
    class CheckpointWriter;
    class ReceiverWriter;
  }
}

class odc::io::ReceiverWriter {
public:
  ReceiverWriter( const char *i_inputFileName,  const char *i_outputFileName,
                  const real &i_deltaH,         const int_pt &i_numZGridPoints );
  void    writeReceiverOutputFiles( const int_pt &i_currentTimeStep,
                                    const int_pt &i_numTimestepsToSkip,
                                    PatchDecomp& i_ptchDec );
  void    finalize();

private:
  int_pt  m_numberOfReceivers;
  int_pt  *m_arrayGridX;
  int_pt  *m_arrayGridY;
  int_pt  *m_arrayGridZ;
  bool    *m_ownedByThisRank;

  int_pt  m_buffSkip;
  int_pt  *m_buffCount;
  int_pt  **m_buffTimestep;
  real    **m_buffVelX;
  real    **m_buffVelY;
  real    **m_buffVelZ;

  char    m_receiverInputFileName[AWP_PATH_MAX];
  char    m_receiverOutputFileName[AWP_PATH_MAX];
  char    **m_receiverOutputLogs;

  FILE    *m_receiverInputFilePtr;
  FILE    *m_receiverOutputFilePtr;
};

class odc::io::CheckpointWriter {
public:
  CheckpointWriter( const char *i_fileName,             const int_pt &i_nd,
                    const int_pt &i_numTimestepsToSkip, const int_pt &i_numZGridPoints );

  void writeInitialStats( const int_pt &i_ntiskp,  const real i_dt,
                          const real &i_dh,        const int_pt &i_nxt,
                          const int_pt &i_nyt,     const int_pt &i_nzt,
                          const int_pt &i_nt,      const real &i_arbc,
                          const int_pt &i_npc,     const int_pt &i_nve,
                          const real &i_fac,       const real &i_q0,
                          const real &i_ex,        const real &i_fp,
                          const real &i_vse_min,   const real &i_vse_max,
                          const real &i_vpe_min,   const real &i_vpe_max,
                          const real &i_dde_min,   const real &i_dde_max );

  void writeUpdatedStats( int_pt i_currentTimeStep, PatchDecomp& i_ptchDec );
  void finalize();

private:
  int_pt m_nd;
  int_pt m_numTimestepsToSkip;
  int_pt m_numZGridPoints;

  int_pt m_rcdX, m_rcdY, m_rcdZ;
  bool m_ownedByThisRank;

  char m_checkPointFileName[AWP_PATH_MAX];
  FILE *m_checkPointFile = nullptr;
};

class odc::io::OutputWriter {
public:
  void update( int_pt i_timestep, PatchDecomp& i_ptchDec );

  void finalize();
  OutputWriter( odc::io::OptionParser& i_options );

private:
  char m_filenamebasex[AWP_PATH_MAX];
  char m_filenamebasey[AWP_PATH_MAX];
  char m_filenamebasez[AWP_PATH_MAX];

  char m_outputFolder[AWP_PATH_MAX];

  int_pt m_firstGlobalXNodeToRecord;
  int_pt m_firstGlobalYNodeToRecord;
  int_pt m_firstGlobalZNodeToRecord;

  int_pt m_lastGlobalXNodeToRecord;
  int_pt m_lastGlobalYNodeToRecord;
  int_pt m_lastGlobalZNodeToRecord;

  int_pt m_numGlobalXNodesToRecord;
  int_pt m_numGlobalYNodesToRecord;
  int_pt m_numGlobalZNodesToRecord;

  int m_numTimestepsToSkip;

  int_pt m_firstXNodeToRecord;
  int_pt m_firstYNodeToRecord;
  int_pt m_firstZNodeToRecord;

  int_pt m_lastXNodeToRecord;
  int_pt m_lastYNodeToRecord;
  int_pt m_lastZNodeToRecord;

  int m_writeStep;

  int_pt m_numXNodesToSkip;
  int_pt m_numYNodesToSkip;
  int_pt m_numZNodesToSkip;

  int_pt m_numXNodesToRecord;
  int_pt m_numYNodesToRecord;
  int_pt m_numZNodesToRecord;
  int_pt m_numGridPointsToRecord;

  real *m_velocityXWriteBuffer;
  real *m_velocityYWriteBuffer;
  real *m_velocityZWriteBuffer;

#ifdef AWP_USE_MPI
  MPI_Datatype m_filetype;
  MPI_Offset   m_displacement;

  void calcRecordingPoints( int *rec_nbgx, int *rec_nedx,
                            int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz,
                            int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
                            long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
                            int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY,
                            int NBGZ, int NEDZ, int NSKPZ, int *coord );
#endif
};

#endif /* OutputWriter_hpp */
