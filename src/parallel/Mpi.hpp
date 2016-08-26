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

#include <cstdlib>
#include <iostream>
#include "io/OptionParser.h"

#ifdef AWP_USE_MPI
#include <mpi.h>
#endif

#ifdef YASK
#include "yask/stencil.hpp"
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

    //! Coordinates of the first index in the global grid belonging to this rank
    static int_pt m_startX;
    static int_pt m_startY;
    static int_pt m_startZ;

    //! Coordinates of the last index in the global grid belonging to this rank
    static int_pt m_endX;
    static int_pt m_endY;
    static int_pt m_endZ;
  
    //! Lengths of each dimension (in grid points) for this rank
    static int_pt m_rangeX;
    static int_pt m_rangeY;
    static int_pt m_rangeZ;

    //! total number of MPI ranks
    static int m_size;

    //! number of MPI ranks in each dimension
    static int m_ranksX;
    static int m_ranksY;
    static int m_ranksZ;


    //! buffers for MPI comms 
    static real* m_buffSend[3][3][3];
    static real* m_buffRecv[3][3][3];

    //! size of buffer required for _one grid_
    //! the actual allocation is a multiple of this
    static int_pt m_buffSendSize[3][3][3];
    static int_pt m_buffRecvSize[3][3][3];

    //! ranks of the neighboring MPI ranks; [1][1][1] corresponds to this rank itself;
    //! -1 means no neighbor in that direction 
    static int m_neighborRanks[3][3][3];
  
    /**
     * Check if a coordinate is in this rank.
     *
     * @param i_x the global grid x coordinate.
     * @param i_y the global grid y coordinate.
     * @param i_z the global grid z coordinate.
     **/
    static bool isInThisRank(int i_x, int i_y, int i_z)
    {
      return (m_startX <= i_x && i_x < m_endX && m_startY <= i_y && i_y < m_endY && m_startZ <= i_z && i_z < m_endZ);
    }

    /**
     * Convert MPI coordinate to MPI rank.
     *
     * @param i_x the x MPI rank coordinate.
     * @param i_y the y MPI rank coordinate.
     * @param i_z the z MPI rank coordinate.
     **/
    static int coordToRank(int i_x, int i_y, int i_z)
    {
      return i_z + i_y * m_ranksZ + i_x * m_ranksY * m_ranksZ;
    }

    /**
     * Convert MPI rank to MPI coordinate.
     *
     * @param o_coords the coordinates of the given MPI rank.
     * @param i_rank rank to determine coordinates of.
     **/  
    static void rankToCoord(int* o_coords, int i_rank)
    {
      o_coords[2] = i_rank % m_ranksZ;
      o_coords[1] = (i_rank / m_ranksZ) % m_ranksY;
      o_coords[0] = (i_rank / (m_ranksZ * m_ranksY));      
    }
  
    static void checkForError(int MPI_errno);
    
    /**
     * Initializes MPI.
     *
     * @param i_argc number of command line parameters.
     * @param i_argv values of command line parameters.
     * @param i_options parsed command line option object.
     **/
    static bool initialize( int i_argc, char *i_argv[], odc::io::OptionParser& i_options);

    /**
     * Send and receive buffers in x dimension.  Assumes buffers already filled!
     * This function is blocking.
     *
     * @param i_numGrids number of grids to send. 
     * @param i_dir 0 = x, 1 = y, 2 = z
    **/
    static void sendRecvBuffers(int i_numGrids, int i_dir);
    
    /**
     * Finalizes MPI.
     **/
    static void finalize() {
#ifdef AWP_USE_MPI
      for(int x=0; x<=2; x+=1)
      {
	for(int y=0; y<=2; y+=1)
	{
	  for(int z=0; z<=2; z+=1)
	  {
	    if(m_neighborRanks[x][y][z] != -1)
	    {
	      free(m_buffSend[x][y][z]);
	      free(m_buffRecv[x][y][z]);	      
	    }
	  }
	}
      }
      
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();
#endif
    }
};

#endif
