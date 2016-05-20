/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)

 @section DESCRIPTION
 MPI parallelization.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifndef MPI_H_
#define MPI_H_

namespace odc {
    namespace parallel {
        class Mpi;
    }
}

class odc::parallel::Mpi {
    // private:
    
  public:
    //! current MPI rank
    static int m_rank;
    
    //! MPI coordinates for cartesian topology
    static int coords[3];
    
    //! total number of MPI ranks
    static int m_size;
    
    static void checkForError(int MPI_errno);
    
    /**
     * Initializes MPI.
     *
     * @param i_argc number of command line parameters.
     * @param i_argv values of command line parameters.
     **/
    static void initialize( int i_argc, char *i_argv[] ) {
        // set default values for non-mpi runs
        m_size = 1;
        m_rank = 0;
        coords[0] = 0;
        coords[1] = 0;
        coords[2] = 0;
        
    
        
#ifdef USE_MPI
        // initialize MPI, get size and rank
        MPI_Init(      &i_argc,        &i_argv);
        MPI_Comm_size( MPI_COMM_WORLD, &m_size);
        MPI_Comm_rank( MPI_COMM_WORLD, &m_rank);
#endif
    };
    
    /**
     * Finalizes MPI.
     **/
    static void finalize() {
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
#endif
    }
};

#endif
