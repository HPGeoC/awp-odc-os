/**
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
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
    }
}

class odc::io::CheckpointWriter {
public:
    
    CheckpointWriter(char *i_fileName, int_pt nd, int_pt numTimestepsToSkip,
                     int_pt numZGridPoints);
    
    void writeInitialStats(int_pt ntiskp, real dt, real dh, int_pt nxt, int_pt nyt, int_pt nzt,
                           int_pt nt, real arbc, int_pt npc, int_pt nve, real fac, real q0, real ex, real fp,
                           real vse_min, real vse_max, real vpe_min, real vpe_max, real dde_min, real dde_max);

    void writeUpdatedStats(int_pt currentTimeStep, PatchDecomp& i_ptchDec);
    void finalize();
    
private:
    
    int_pt m_nd;
    int_pt m_numTimestepsToSkip;
    int_pt m_numZGridPoints;

    int_pt m_rcdX, m_rcdY, m_rcdZ;
    bool m_ownedByThisRank;
    
    char m_checkPointFileName[256];
    FILE *m_checkPointFile = nullptr;
    
};

class odc::io::OutputWriter {
    
public:
    void update(int_pt i_timestep, PatchDecomp& i_ptchDec);

    void finalize();
    OutputWriter(OptionParser i_options);
    
    
private:
    char m_filenamebasex[256];
    char m_filenamebasey[256];
    char m_filenamebasez[256];
    
    char m_outputFolder[50];
    
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
    
    MPI_Datatype m_filetype;
    MPI_Offset   m_displacement;
    
    real *m_velocityXWriteBuffer;
    real *m_velocityYWriteBuffer;
    real *m_velocityZWriteBuffer;
    
    void calcRecordingPoints(int *rec_nbgx, int *rec_nedx,
                             int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz,
                             int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
                             long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
                             int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY,
                             int NBGZ, int NEDZ, int NSKPZ, int *coord);
    

    
};



#endif /* OutputWriter_hpp */
