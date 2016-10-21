/**
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "parallel/Mpi.hpp"

void odc::parallel::Mpi::checkForError(int MPI_errno) {
}

int odc::parallel::Mpi::m_rank;    
int odc::parallel::Mpi::coords[3];

int_pt odc::parallel::Mpi::m_startX;
int_pt odc::parallel::Mpi::m_startY;
int_pt odc::parallel::Mpi::m_startZ;

int_pt odc::parallel::Mpi::m_endX;
int_pt odc::parallel::Mpi::m_endY;
int_pt odc::parallel::Mpi::m_endZ;
  
int_pt odc::parallel::Mpi::m_rangeX;
int_pt odc::parallel::Mpi::m_rangeY;
int_pt odc::parallel::Mpi::m_rangeZ;

int odc::parallel::Mpi::m_size;

int odc::parallel::Mpi::m_ranksX;
int odc::parallel::Mpi::m_ranksY;
int odc::parallel::Mpi::m_ranksZ;

real* odc::parallel::Mpi::m_buffSend[3][3][3];
real* odc::parallel::Mpi::m_buffRecv[3][3][3];
int_pt odc::parallel::Mpi::m_buffSendSize[3][3][3];
int_pt odc::parallel::Mpi::m_buffRecvSize[3][3][3];

int odc::parallel::Mpi::m_neighborRanks[3][3][3];

bool odc::parallel::Mpi::initialize( int i_argc, char *i_argv[], odc::io::OptionParser& i_options)
{
  // set default values for non-mpi runs
  m_size = 1;
  m_rank = 0;
  m_ranksX = 1;
  m_ranksY = 1;
  m_ranksZ = 1;
  coords[0] = 0;
  coords[1] = 0;
  coords[2] = 0;
  m_startX = m_startY = m_startZ = 0;
  m_rangeX = m_endX = i_options.m_nX;
  m_rangeY = m_endY = i_options.m_nY;
  m_rangeZ = m_endZ = i_options.m_nZ;
        
#ifdef AWP_USE_MPI
  // initialize threaded MPI, check that the correct thread level is provided
  int thread_environment;
  MPI_Init_thread(&i_argc, &i_argv, MPI_THREAD_SERIALIZED, &thread_environment);

  if(thread_environment < MPI_THREAD_SERIALIZED)
  {
    std::cerr << "Warning: could not initialize correct thread level for several threads" << std::endl;
  }
	
  MPI_Comm_size( MPI_COMM_WORLD, &m_size);
  MPI_Comm_rank( MPI_COMM_WORLD, &m_rank);

  // get desired MPI dimensions from the parsed command line args
  m_ranksX = i_options.m_pX;
  m_ranksY = i_options.m_pY;
  m_ranksZ = 1; // TODO(Josh): enable this in the option parser	

  if(m_ranksX * m_ranksY * m_ranksZ != m_size)
  {
    std::cerr << "Error: number of MPI ranks provided does not match number of MPI ranks provided" << std::endl;
    return false;
  }

  rankToCoord(coords, m_rank);

  m_startX = (i_options.m_nX * coords[0]) / m_ranksX;
  m_startY = (i_options.m_nY * coords[1]) / m_ranksY;
  m_startZ = (i_options.m_nZ * coords[2]) / m_ranksZ;

#ifdef YASK	
  if((m_startX % CPTS_X) || (m_startY % CPTS_Y) || (m_startZ % CPTS_Z))
  {
    std::cout << "Error: cannot divide into blocks divisible by YASK CPTS" << std::endl;
    return false;
  }
#endif

  m_endX = (i_options.m_nX * (coords[0]+1)) / m_ranksX;
  m_endY = (i_options.m_nY * (coords[1]+1)) / m_ranksY;
  m_endZ = (i_options.m_nZ * (coords[2]+1)) / m_ranksZ;

  m_rangeX = m_endX - m_startX;
  m_rangeY = m_endY - m_startY;
  m_rangeZ = m_endZ - m_startZ;	
	
  // estalish ranks of neighbors of this rank
  for(int x=-1; x<=1; x+=1)
  {
    for(int y=-1; y<=1; y+=1)
    {
      for(int z=-1; z<=1; z+=1)
      {
        int tx = coords[0] + x;
	int ty = coords[1] + y;
	int tz = coords[2] + z;
	      
	if(tx < 0 || tx >= m_ranksX || ty < 0 || ty >= m_ranksY || tz < 0 || tz >= m_ranksZ)
	  m_neighborRanks[1+x][1+y][1+z] = -1;
	else
	{
	  m_neighborRanks[1+x][1+y][1+z] = coordToRank(tx,ty,tz);
	}
      }
    }
  }
  //m_neighborRanks[0][1][1] = m_neighborRanks[2][1][1] = 1;
  //m_neighborRanks[1][0][1] = m_neighborRanks[1][2][1] = 1; // PPP: remove this
	
  if(m_endX > i_options.m_nX)
    m_endX = i_options.m_nX;
  if(m_endY > i_options.m_nY)
    m_endY = i_options.m_nY;
  if(m_endZ > i_options.m_nZ)
    m_endZ = i_options.m_nZ;

  const int max_num_grids = 9;
	
  if(m_neighborRanks[0][1][1] != -1)
  {
    m_buffSendSize[0][1][1] = m_rangeY * m_rangeZ * 2;
    m_buffRecvSize[0][1][1] = m_rangeY * m_rangeZ * 2;
    m_buffSend[0][1][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeY * m_rangeZ * 2);
    m_buffRecv[0][1][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeY * m_rangeZ * 2);
  }
  if(m_neighborRanks[2][1][1] != -1)
  {
    m_buffSendSize[2][1][1] = m_rangeY * m_rangeZ * 2;
    m_buffRecvSize[2][1][1] = m_rangeY * m_rangeZ * 2;
    m_buffSend[2][1][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeY * m_rangeZ * 2);
    m_buffRecv[2][1][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeY * m_rangeZ * 2);
  }

  if(m_neighborRanks[1][0][1] != -1)
  {
    m_buffSendSize[1][0][1] = m_rangeX * m_rangeZ * 2;
    m_buffRecvSize[1][0][1] = m_rangeX * m_rangeZ * 2;
    m_buffSend[1][0][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeZ * 2);
    m_buffRecv[1][0][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeZ * 2);
  }
  if(m_neighborRanks[1][2][1] != -1)
  {
    m_buffSendSize[1][2][1] = m_rangeX * m_rangeZ * 2;
    m_buffRecvSize[1][2][1] = m_rangeX * m_rangeZ * 2;
    m_buffSend[1][2][1] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeZ * 2);
    m_buffRecv[1][2][1] = (real*) malloc(max_num_grids* sizeof(real) * m_rangeX * m_rangeZ * 2);
  }

  if(m_neighborRanks[1][1][0] != -1)
  {
    m_buffSendSize[1][1][0] = m_rangeX * m_rangeY * 2;
    m_buffRecvSize[1][1][0] = m_rangeX * m_rangeY * 2;
    m_buffSend[1][1][0] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeY * 2);
    m_buffRecv[1][1][0] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeY * 2);
  }
  if(m_neighborRanks[1][1][2] != -1)
  {
    m_buffSendSize[1][1][2] = m_rangeX * m_rangeY * 2;
    m_buffRecvSize[1][1][2] = m_rangeX * m_rangeY * 2;
    m_buffSend[1][1][2] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeY * 2);
    m_buffRecv[1][1][2] = (real*) malloc(max_num_grids * sizeof(real) * m_rangeX * m_rangeY * 2);
  }
	
#endif
  return true;
};



void odc::parallel::Mpi::sendRecvBuffers(int i_numGrids, int i_dir)  
{
#ifdef AWP_USE_MPI      
  MPI_Request sendReq[3][3][3], recvReq[3][3][3];
  MPI_Status status;
  int doneReq[2][2]; // first index represents coord, second is 0 for recv 1 for send
  for(int i=0; i<2; i++)
    for(int j=0; j<2; j++)
      doneReq[i][j] = 0;
  int num_active_dirs = 0; // how many communications have yet to terminate

  int l_ind[3][2]; // stores the indices that correspond to direction i_dir;  first index is x,y,z, second is for down / up
  for(int i=0; i<3; i++)
    for(int j=0; j<2; j++)
      l_ind[i][j] = 1;

  l_ind[i_dir][0] = 0;
  l_ind[i_dir][1] = 2;

  for(int i=0; i<=1; i++)
  {
    if(m_neighborRanks[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]] != -1)
    {
      MPI_Irecv(m_buffRecv[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]], i_numGrids * m_buffRecvSize[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]],
	        AWP_MPI_REAL, m_neighborRanks[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]], 0, MPI_COMM_WORLD,
		&recvReq[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]]);
      MPI_Isend(m_buffSend[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]], i_numGrids * m_buffSendSize[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]],
		AWP_MPI_REAL, m_neighborRanks[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]], 0, MPI_COMM_WORLD,
		&sendReq[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]]);
      num_active_dirs+=2;
    }
  }
      
  while(num_active_dirs > 0)
  {
    for(int i=0; i<=1; i++)
    {
      if(m_neighborRanks[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]] != -1 && !doneReq[i][0])
      {
        MPI_Test(&recvReq[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]], &doneReq[i][0], &status);
        if(doneReq[i][0])
        {
          num_active_dirs--;
        }
      }
      if(m_neighborRanks[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]] != -1 && !doneReq[i][1])
      {
        MPI_Test(&sendReq[l_ind[0][i]][l_ind[1][i]][l_ind[2][i]], &doneReq[i][1], &status);
	if(doneReq[i][1])
	{
	  num_active_dirs--;
	}
      }
    }
  }
#endif	  
}
