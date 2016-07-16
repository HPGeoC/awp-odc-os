/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Source terms.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SOURCES_H_
#define SOURCES_H_

// TODO: Required for sources
#include <mpi.h>

#include "data/PatchDecomp.hpp"
#include "constants.hpp"

extern "C" int inisource( int      rank,    int     IFAULT, int     NSRC,   int     READ_STEP, int     NST,     int     *SRCPROC, int    NZ, 
                          MPI_Comm MCW,     int     nxt,    int     nyt,    int     nzt,       int     *coords, int     maxdim,   int    *NPSRC,
                          PosInf   *ptpsrc, Grid1D  *ptaxx, Grid1D  *ptayy, Grid1D  *ptazz,    Grid1D  *ptaxz,  Grid1D  *ptayz,   Grid1D *ptaxy, char *INSRC, char *INSRC_I2);

namespace odc {
  namespace io {
    class Sources;
  }
}

class odc::io::Sources {
  public:
    // If one or more fault source nodes present, @c SRCPROC is set to rank (MPI). -1 otherwise
    int m_srcProc;

    // Number of fault source nodes owned by calling process.
    int m_nPsrc;

    // indices of nodes owned by calling process that are fault sources \n
    // <tt>psrc[i*maxdim]</tt>   = x node index of source fault @c i \n
    // <tt>psrc[i*maxdim+1]</tt> = y node index of source fault @c i \n
    // <tt>psrc[i*maxdim+2]</tt> = z node index of source fault @c i
    PosInf m_ptpSrc;

    // TODO: documentation
    Grid1D m_ptAxx;
    Grid1D m_ptAyy;
    Grid1D m_ptAzz;

    // TODO: documentation
    Grid1D m_ptAxy;
    Grid1D m_ptAxz;
    Grid1D m_ptAyz;

    // Disable default constructor
    Sources() = delete;

    // Constructor: Initializes the sources.
    Sources( int   i_iFault,
             int   i_nSrc,
             int   i_readStep,
             int   i_nSt,
             int   i_nZ,
             int   i_nXt, int i_nYt, int i_nZt,
             char *i_inSrc,
            char *i_inSrcI2 );
    
    
    void addsrc(int_pt i, float DH,   float DT,   int NST,  int READ_STEP, int dim, PatchDecomp& pd);
    
};

#endif
