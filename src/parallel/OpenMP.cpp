/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)

 @section DESCRIPTION
 OMP parallelization.
 
 @section LICENSE
 Copyright (c) 2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "OpenMP.h"

#include <omp.h>
#include <limits>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <deque>

static inline void makeAssignment(int x, int y, int z, int m_nWPX, int m_nWPY, int m_nWPZ, int indexVel, int l_nWP_oneKernel, odc::parallel::WorkPackage* m_workPackages,
				  int_pt startX, int_pt startY, int_pt startZ, int_pt endX, int_pt endY, int_pt endZ,
				  bool mpiNbrXLeft, bool mpiNbrXRight, bool mpiNbrYLeft, bool mpiNbrYRight, bool mpiNbrZLeft, bool mpiNbrZRight)
{
  // the stress WP are offset by the number of vel WPs plus one MPI WP
  int indexStress = indexVel + 1 + l_nWP_oneKernel;

  m_workPackages[indexVel].type    = odc::parallel::WorkPackageType::WP_VelUpdate;
  m_workPackages[indexStress].type = odc::parallel::WorkPackageType::WP_StressUpdate;
	
  m_workPackages[indexVel].start[0] = m_workPackages[indexStress].start[0] = startX;
  m_workPackages[indexVel].start[1] = m_workPackages[indexStress].start[1] = startY;
  m_workPackages[indexVel].start[2] = m_workPackages[indexStress].start[2] = startZ;

  m_workPackages[indexVel].end[0] = m_workPackages[indexStress].end[0] = endX;
  m_workPackages[indexVel].end[1] = m_workPackages[indexStress].end[1] = endY;
  m_workPackages[indexVel].end[2] = m_workPackages[indexStress].end[2] = endZ;

  // initially assume that every WorkPackage is innocent of being on free surface / x,y boundaries
  m_workPackages[indexVel].freeSurface = false;
  m_workPackages[indexVel].xMaxBdry = false;
  m_workPackages[indexVel].yMinBdry = false;

  
  for(int i=0; i<3; i++)
    for(int j=0; j<3; j++)
      for(int k=0; k<3; k++)
        m_workPackages[indexVel].mpiDir[i][j][k] = m_workPackages[indexStress].mpiDir[i][j][k] = false;
	
  bool onMpiBdry = false;
  if(x == 0 && mpiNbrXLeft)
  {
    onMpiBdry = true;
    m_workPackages[indexVel].mpiDir[0][1][1] = m_workPackages[indexStress].mpiDir[0][1][1] = true;
  }
  if(x == m_nWPX-1 && mpiNbrXRight)
  {
    onMpiBdry = true;
    m_workPackages[indexVel].mpiDir[2][1][1] = m_workPackages[indexStress].mpiDir[2][1][1] = true;
  }
  if(y == 0 && mpiNbrYLeft)
  {
    onMpiBdry = true;
    m_workPackages[indexVel].mpiDir[1][0][1] = m_workPackages[indexStress].mpiDir[1][0][1] = true;
  }
  if(y == m_nWPY-1 && mpiNbrYRight)
  {
    onMpiBdry = true;
    m_workPackages[indexVel].mpiDir[1][2][1] = m_workPackages[indexStress].mpiDir[1][2][1] = true;
  }
  if(z == 0 && mpiNbrZLeft)
  {
    onMpiBdry = true;
    m_workPackages[indexVel].mpiDir[1][1][0] = m_workPackages[indexStress].mpiDir[1][1][0] = true;
  }
  if(z == m_nWPZ-1 && mpiNbrZRight)
  {
    onMpiBdry = true;
    m_workPackages[indexVel].mpiDir[1][1][2] = m_workPackages[indexStress].mpiDir[1][1][2] = true;
  }

  if(z == m_nWPZ-1 && !mpiNbrZRight)
  {
    m_workPackages[indexStress].freeSurface = true;
  }

  if(y == 0 && !mpiNbrYLeft)
  {
    m_workPackages[indexStress].yMinBdry = true;
  }

  if(x == m_nWPX-1 && !mpiNbrXRight)
  {
    m_workPackages[indexStress].xMaxBdry = true;
  }  

  m_workPackages[indexVel].copyFromBuffer = m_workPackages[indexStress].copyFromBuffer = onMpiBdry;
  m_workPackages[indexVel].copyToBuffer = m_workPackages[indexStress].copyToBuffer = onMpiBdry;
}

odc::parallel::OpenMP::OpenMP( int_pt       i_nPtsX,
                               int_pt       i_nPtsY,
                               int_pt       i_nPtsZ,
                               int          i_nManageThreads,
                               int          i_nCompThreads,
                               PatchDecomp& i_ptchDec):
  m_nWgrpsAll(0), m_ptchDec(i_ptchDec), m_nPtsX(i_nPtsX), m_nPtsY(i_nPtsY), m_nPtsZ(i_nPtsZ) {


  m_trdNumAll = omp_get_thread_num();

  m_nThreadsAll = omp_get_num_threads();

  if(i_nManageThreads + i_nCompThreads != m_nThreadsAll)
  {
    std::cerr << "Error: number of threads requested does not equal number of threads provided by system." << std::endl;
  }  
  
  m_packageSizeX = WP_SIZE_X;
  m_packageSizeY = WP_SIZE_Y;
  m_packageSizeZ = WP_SIZE_Z;

  m_nWPX = (int) ((i_nPtsX + m_packageSizeX-1) / m_packageSizeX);
  m_nWPY = (int) ((i_nPtsY + m_packageSizeY-1) / m_packageSizeY);
  m_nWPZ = (int) ((i_nPtsZ + m_packageSizeZ-1) / m_packageSizeZ);

  // the number of work packages is given by those for the vel and stress updates and
  // two additional WPs for the MPI communications
  int l_nWP_oneKernel = m_nWPX * m_nWPY * m_nWPZ;
  m_nWP  = l_nWP_oneKernel * 2 + 2;

  m_workPackages  = (odc::parallel::WorkPackage*) malloc(m_nWP * sizeof(odc::parallel::WorkPackage));
  
  // first create the WPs for the velocity update
  int curTopX = 0, curBotX = 0, curTopY = 0, curBotY = 0, curTopZ = 0, curBotZ = 0;

  // left means in the negative direction, right means positive dir
  bool mpiNbrXLeft  = (odc::parallel::Mpi::m_neighborRanks[0][1][1] != -1);
  bool mpiNbrXRight = (odc::parallel::Mpi::m_neighborRanks[2][1][1] != -1);
  bool mpiNbrYLeft  = (odc::parallel::Mpi::m_neighborRanks[1][0][1] != -1);
  bool mpiNbrYRight = (odc::parallel::Mpi::m_neighborRanks[1][2][1] != -1);
  bool mpiNbrZLeft  = (odc::parallel::Mpi::m_neighborRanks[1][1][0] != -1);
  bool mpiNbrZRight = (odc::parallel::Mpi::m_neighborRanks[1][1][2] != -1);

  int indexVel = 0;

  for(int y=0; y<m_nWPY; y++)
    for(int z=0; z<m_nWPZ; z++)
    {
      int x = 0;
      int_pt startX = x*m_packageSizeX;
      int_pt endX   = (x+1)*m_packageSizeX;
      int_pt startY = y*m_packageSizeY;
      int_pt endY   = (y+1)*m_packageSizeY;
      int_pt startZ = z*m_packageSizeZ;
      int_pt endZ   = (z+1)*m_packageSizeZ;
      if(endX > i_nPtsX)
        endX = i_nPtsX;
      if(endY > i_nPtsY)
        endY = i_nPtsY;
      if(endZ > i_nPtsZ)
        endZ = i_nPtsZ;
      
      makeAssignment(x, y, z, m_nWPX, m_nWPY, m_nWPZ, indexVel, l_nWP_oneKernel, m_workPackages, startX, startY, startZ, endX, endY, endZ,
		       mpiNbrXLeft, mpiNbrXRight, mpiNbrYLeft, mpiNbrYRight, mpiNbrZLeft, mpiNbrZRight);
      indexVel++;
    }

  m_endOfStressBdryXZero = indexVel + l_nWP_oneKernel + 1 - 1;

  
  for(int y=0; y<m_nWPY; y++) 
    for(int z=0; z<m_nWPZ; z++)
    {
      int x = m_nWPX-1;
      int_pt startX = x*m_packageSizeX;
      int_pt endX   = (x+1)*m_packageSizeX;
      int_pt startY = y*m_packageSizeY;
      int_pt endY   = (y+1)*m_packageSizeY;
      int_pt startZ = z*m_packageSizeZ;
      int_pt endZ   = (z+1)*m_packageSizeZ;
      if(endX > i_nPtsX)
        endX = i_nPtsX;
      if(endY > i_nPtsY)
        endY = i_nPtsY;
      if(endZ > i_nPtsZ)
        endZ = i_nPtsZ;

      makeAssignment(x, y, z, m_nWPX, m_nWPY, m_nWPZ, indexVel, l_nWP_oneKernel, m_workPackages, startX, startY, startZ, endX, endY, endZ,
		       mpiNbrXLeft, mpiNbrXRight, mpiNbrYLeft, mpiNbrYRight, mpiNbrZLeft, mpiNbrZRight);
      indexVel++;
    }

  for(int x=1; x<m_nWPX-1; x++)
    for(int z=0; z<m_nWPZ; z++)
    {
      int y = 0;
      int_pt startX = x*m_packageSizeX;
      int_pt endX   = (x+1)*m_packageSizeX;
      int_pt startY = y*m_packageSizeY;
      int_pt endY   = (y+1)*m_packageSizeY;
      int_pt startZ = z*m_packageSizeZ;
      int_pt endZ   = (z+1)*m_packageSizeZ;
      if(endX > i_nPtsX)
        endX = i_nPtsX;
      if(endY > i_nPtsY)
        endY = i_nPtsY;
      if(endZ > i_nPtsZ)
        endZ = i_nPtsZ;
      
      makeAssignment(x, y, z, m_nWPX, m_nWPY, m_nWPZ, indexVel, l_nWP_oneKernel, m_workPackages, startX, startY, startZ, endX, endY, endZ,
		       mpiNbrXLeft, mpiNbrXRight, mpiNbrYLeft, mpiNbrYRight, mpiNbrZLeft, mpiNbrZRight);
      indexVel++;
    }
  
  for(int x=1; x<m_nWPX-1; x++)
    for(int z=0; z<m_nWPZ; z++)
    {
      int y = m_nWPY-1;
      int_pt startX = x*m_packageSizeX;
      int_pt endX   = (x+1)*m_packageSizeX;
      int_pt startY = y*m_packageSizeY;
      int_pt endY   = (y+1)*m_packageSizeY;
      int_pt startZ = z*m_packageSizeZ;
      int_pt endZ   = (z+1)*m_packageSizeZ;
      if(endX > i_nPtsX)
        endX = i_nPtsX;
      if(endY > i_nPtsY)
        endY = i_nPtsY;
      if(endZ > i_nPtsZ)
        endZ = i_nPtsZ;
      
      makeAssignment(x, y, z, m_nWPX, m_nWPY, m_nWPZ, indexVel, l_nWP_oneKernel, m_workPackages, startX, startY, startZ, endX, endY, endZ,
		       mpiNbrXLeft, mpiNbrXRight, mpiNbrYLeft, mpiNbrYRight, mpiNbrZLeft, mpiNbrZRight);
      indexVel++;
    }

  m_nBdry = indexVel;
  m_endOfVelBdry = indexVel-1;
  m_endOfStressBdry = indexVel + l_nWP_oneKernel + 1 - 1;

  int total_since_bdry = 0;

  for(int y=1; y<m_nWPY-1; y++)
    for(int z=0; z<m_nWPZ; z++)
    {
      int x = 1;
      int_pt startX = x*m_packageSizeX;
      int_pt endX   = (x+1)*m_packageSizeX;
      int_pt startY = y*m_packageSizeY;
      int_pt endY   = (y+1)*m_packageSizeY;
      int_pt startZ = z*m_packageSizeZ;
      int_pt endZ   = (z+1)*m_packageSizeZ;
      if(endX > i_nPtsX)
        endX = i_nPtsX;
      if(endY > i_nPtsY)
        endY = i_nPtsY;
      if(endZ > i_nPtsZ)
        endZ = i_nPtsZ;
      
      makeAssignment(x, y, z, m_nWPX, m_nWPY, m_nWPZ, indexVel, l_nWP_oneKernel, m_workPackages, startX, startY, startZ, endX, endY, endZ,
		       mpiNbrXLeft, mpiNbrXRight, mpiNbrYLeft, mpiNbrYRight, mpiNbrZLeft, mpiNbrZRight);
      indexVel++;
    }

  m_velMpiWP    = indexVel;
  m_stressMpiWP = indexVel + l_nWP_oneKernel + 1;
    
  m_workPackages[m_velMpiWP].type     = WP_MPI_Vel;
  m_workPackages[m_stressMpiWP].type  = WP_MPI_Stress;
  indexVel++;
  

  
  // Ideally we would like to place some work packages between the end of the boundary computation and
  // the mpi communication.  If there are too few work packages in the domain this isn't possible,
  // in which case just put the mpi communication right after the boundary is done (this basically
  // guarantees some stalling of cores)
  /*if(indexVel + i_nCompThreads * 3 > l_nWP_oneKernel + 1 - 1)
  {
    std::cout << "this is happening: " << indexVel  << ' ' << i_nCompThreads << ' ' << l_nWP_oneKernel << std::endl;
    
    m_velMpiWP    = indexVel;
    m_stressMpiWP = indexVel + l_nWP_oneKernel + 1;
    
    m_workPackages[m_velMpiWP].type     = WP_MPI_Vel;
    m_workPackages[m_stressMpiWP].type  = WP_MPI_Stress;
    indexVel++;

    // make the total_since_bdry large so that we don't try to place the MPI communication WPs again
    total_since_bdry = i_nCompThreads * 10;
    }*/


  //std::cout << "index before body: " << indexVel << std::endl;
  
  for(int x=2; x<m_nWPX-1; x++)
  {
    int_pt startX = x*m_packageSizeX;
    int_pt endX   = (x+1)*m_packageSizeX;
    if(endX > i_nPtsX)
      endX = i_nPtsX;
    
    for(int y=1; y<m_nWPY-1; y++)
    {
      int_pt startY = y*m_packageSizeY;
      int_pt endY   = (y+1)*m_packageSizeY;
      if(endY > i_nPtsY)
        endY = i_nPtsY;
      
      for(int z=0; z<m_nWPZ; z++)
      {
	int_pt startZ = z*m_packageSizeZ;
        int_pt endZ   = (z+1)*m_packageSizeZ;
        if(endZ > i_nPtsZ)
          endZ = i_nPtsZ;

	makeAssignment(x, y, z, m_nWPX, m_nWPY, m_nWPZ, indexVel, l_nWP_oneKernel, m_workPackages, startX, startY, startZ, endX, endY, endZ,
		       mpiNbrXLeft, mpiNbrXRight, mpiNbrYLeft, mpiNbrYRight, mpiNbrZLeft, mpiNbrZRight);
	indexVel++;
	total_since_bdry++;

/*	if(total_since_bdry == 2*i_nCompThreads)
	{
          m_velMpiWP    = indexVel;
          m_stressMpiWP = indexVel + l_nWP_oneKernel + 1;
	  
          m_workPackages[m_velMpiWP].type     = WP_MPI_Vel;
          m_workPackages[m_stressMpiWP].type  = WP_MPI_Stress;
          indexVel++;	  
	  }*/
      }
    }
  }

  // useful for debugging:
#if 0
  #pragma omp critical
  {
    std::cout << "WP layout: " << std::endl << std::endl;

    for(int i=0; i<m_nWP; i++)
    {
      std::cout << i << " ";
      if(m_workPackages[i].type == WP_VelUpdate)
	std::cout << "vel kernel " << ' ' << m_workPackages[i].start[0] / m_packageSizeX << ' ' << m_workPackages[i].start[1] / m_packageSizeY << ' ' << m_workPackages[i].start[2] / m_packageSizeZ << std::endl;
      else if(m_workPackages[i].type == WP_StressUpdate)
	std::cout << "stress kernel" << ' ' << m_workPackages[i].start[0] / m_packageSizeX << ' ' << m_workPackages[i].start[1] / m_packageSizeY << ' ' << m_workPackages[i].start[2] / m_packageSizeZ << std::endl;
      else if(m_workPackages[i].type == WP_MPI_Vel)
	std::cout << "vel mpi" << std::endl;
      else if(m_workPackages[i].type == WP_MPI_Stress)
	std::cout << "stress mpi" << std::endl;
      else
	std::cout << "?????" << std::endl;
      
    }
  }
#endif  
}

odc::parallel::OpenMP::~OpenMP()
{
  free(m_workPackages);
}

void odc::parallel::OpenMP::partDomain( int_pt   i_nParts,
                                        int_pt   i_offX,
                                        int_pt   i_offY,
                                        int_pt   i_offZ,
                                        int_pt   i_nPtsX,
                                        int_pt   i_nPtsY,
                                        int_pt   i_nPtsZ,
                                        int    (*o_domDec)[3],
                                        int_pt (*o_range)[3][2] ) {
  // number of domains in x-, y- and z-direction
  int l_nXdoms = 1;
  int l_nYdoms = 1;
  int l_nZdoms = 1;
  
  // find the primes smaller than i_nParts which we will
  // use to decompose i_nParts into prime powers
  std::vector<bool> is_prime(i_nParts+1,true);
  std::deque<int> primes;
  for(int i=2; i<=i_nParts; i++)
  {
    if(is_prime[i])
    {
      // if this prime divides into i_nParts, keep it for later
      if((i_nParts % i) == 0)
        primes.push_front(i);
      for(int j=2*i; j<=i_nParts; j+=i)
      {
        is_prime[j] = false;
      }
    }
  }
  
  
  // divide by multiples of 2
  int l_testDec = i_nParts;
  std::deque<int>::iterator cur_prime = primes.begin();

  while( l_testDec != 1 ) {
    if( (l_testDec % *cur_prime) != 0 )
      cur_prime++;

    l_testDec /= (*cur_prime);

    if( (i_nPtsX/l_nXdoms) >= (i_nPtsY/l_nYdoms) &&
        (i_nPtsX/l_nXdoms) >= (i_nPtsZ/l_nZdoms) )
      l_nXdoms *= (*cur_prime);
    else if( i_nPtsY/l_nYdoms >= (i_nPtsZ/l_nZdoms) )
      l_nYdoms *= (*cur_prime);
    else
      l_nZdoms *= (*cur_prime);
  }

  // assign partitions
  int l_part = 0;

  for( int l_xDom = 0; l_xDom < l_nXdoms; l_xDom++ ) {
    for( int l_yDom = 0; l_yDom < l_nYdoms; l_yDom++ ) {
      for( int l_zDom = 0; l_zDom < l_nZdoms; l_zDom++ ) {
        o_domDec[l_part][0] = l_xDom;
        o_domDec[l_part][1] = l_yDom;
        o_domDec[l_part][2] = l_zDom;
        l_part++;
      }
    }
  }

  // derive the start and end coords of the partitions
  for( int l_pa = 0; l_pa < i_nParts; l_pa++ ) {
    // get ids of this partition
    int l_xDom = o_domDec[l_pa][0];
    int l_yDom = o_domDec[l_pa][1];
    int l_zDom = o_domDec[l_pa][2];

    o_range[l_pa][0][0] = (i_nPtsX / l_nXdoms) *  l_xDom;
    o_range[l_pa][0][1] = (i_nPtsX / l_nXdoms) * (l_xDom + 1);
    o_range[l_pa][1][0] = (i_nPtsY / l_nYdoms) *  l_yDom;
    o_range[l_pa][1][1] = (i_nPtsY / l_nYdoms) * (l_yDom + 1);
    o_range[l_pa][2][0] = (i_nPtsZ / l_nZdoms) *  l_zDom;
    o_range[l_pa][2][1] = (i_nPtsZ / l_nZdoms) * (l_zDom + 1);

    // adjust upper bounds to cover all points
    if( l_xDom + 1 == l_nXdoms) o_range[l_pa][0][1] = i_nPtsX;
    if( l_yDom + 1 == l_nYdoms) o_range[l_pa][1][1] = i_nPtsY;
    if( l_zDom + 1 == l_nZdoms) o_range[l_pa][2][1] = i_nPtsZ;

    o_range[l_pa][0][0] += i_offX;
    o_range[l_pa][0][1] += i_offX;
    o_range[l_pa][1][0] += i_offY;
    o_range[l_pa][1][1] += i_offY;
    o_range[l_pa][2][0] += i_offZ;
    o_range[l_pa][2][1] += i_offZ;
  }
}

void odc::parallel::OpenMP::deriveLayout( int_pt    i_nPtsX,
                                          int_pt    i_nPtsY,
                                          int_pt    i_nPtsZ,
                                          int       i_nWgrpsComp,
                                          int      *i_nTdsPerWgrpComp) {

  int_pt numPatches = m_ptchDec.m_numPatches;
  int_pt numPatchesLeft = numPatches;
  
  // allocate memory to store patch info  
  m_nPtchsPerWgrpComp = (int_pt*)          malloc(sizeof(int_pt) * m_nWgrpsComp);
  m_wgrpPtchList      = (int_pt**)         malloc(sizeof(int_pt*) * m_nWgrpsComp);
  m_ptchToWgrpComp    = (int*)             malloc(sizeof(int) * numPatches);
  m_domDecCompTds     = (int(**)[3])       malloc( 3 * sizeof(int*) * numPatches);
  m_rangeCompTds      = (int_pt(**)[3][2]) malloc( 3 * 2 * sizeof(int_pt*) * numPatches);

  
  // compute number of patches in each comp workgroup
  for(int_pt i=0; i < m_nWgrpsComp; i++)
  {
    m_nPtchsPerWgrpComp[i] = numPatches / m_nWgrpsComp;
    numPatchesLeft -= m_nPtchsPerWgrpComp[i];
  }
  for(int_pt i=0; numPatchesLeft > 0; i++)
  {
    m_nPtchsPerWgrpComp[i]++;
    numPatchesLeft--;
    assert(i < m_nWgrpsComp);
  }  
    
  for(int_pt i=0; i < m_nWgrpsComp; i++)
  {
    m_wgrpPtchList[i] = (int_pt*) malloc(sizeof(int_pt) * m_nPtchsPerWgrpComp[i]);
  }
 

  int l_wgrp = 0;
  int_pt numAssigned = 0;

  for(int_pt x_ptch=0; x_ptch < m_ptchDec.m_numXPatches; x_ptch++)
  for(int_pt y_ptch=0; y_ptch < m_ptchDec.m_numYPatches; y_ptch++)
  for(int_pt z_ptch=0; z_ptch < m_ptchDec.m_numZPatches; z_ptch++)
  {
    int l_ptch = m_ptchDec.m_coordToId[x_ptch][y_ptch][z_ptch];
    m_ptchToWgrpComp[l_ptch] = l_wgrp;
    m_wgrpPtchList[l_wgrp][numAssigned] = l_ptch;

    numAssigned++;
    if(numAssigned == m_nPtchsPerWgrpComp[l_wgrp])
    {
      numAssigned = 0;
      l_wgrp++;
    }
  }
  
  for( int_pt l_ptch = 0; l_ptch < numPatches; l_ptch++ ) {
    int l_wg = m_ptchToWgrpComp[l_ptch];
    m_domDecCompTds[l_ptch] = (int(*)[3])       malloc( 3 *     sizeof(int*)    * m_nTdsPerWgrpComp[l_wg] );
    m_rangeCompTds[l_ptch]  = (int_pt(*)[3][2]) malloc( 3 * 2 * sizeof(int_pt*) * m_nTdsPerWgrpComp[l_wg] );    
  }    
    
  // iterate over patches and derive partitions
  for( int_pt l_ptch = 0; l_ptch < numPatches; l_ptch++ ) {
    int l_wg = m_ptchToWgrpComp[l_ptch];
    int_pt ptchSizeX = m_ptchDec.m_patches[l_ptch].size_x;
    int_pt ptchSizeY = m_ptchDec.m_patches[l_ptch].size_y;
    int_pt ptchSizeZ = m_ptchDec.m_patches[l_ptch].size_z;
    
    partDomain( i_nTdsPerWgrpComp[l_wg],
                0, 0, 0,
                ptchSizeX, ptchSizeY, ptchSizeZ,
                m_domDecCompTds[l_ptch], m_rangeCompTds[l_ptch]);
  }
  
}

void odc::parallel::OpenMP::getTrdExtent(int p_id, int_pt o_start[3], int_pt o_size[3])
{
  Patch* p = &m_ptchDec.m_patches[p_id];
  int_pt h = p->bdry_width;

  int trd_id = getThreadNumGrp();
  int_pt trd_ranges[3][2];
  getRangesTrd(p_id, trd_id, trd_ranges);

  bool trd_start[3];
  bool trd_end[3];
  for(int j=0; j<3; j++)
  {
    trd_start[j] = (trd_ranges[j][0] == 0);
    if(j==0)
      trd_end[j] = (trd_ranges[j][1] == p->size_x);
    if(j==1)
      trd_end[j] = (trd_ranges[j][1] == p->size_y);
    if(j==2)
      trd_end[j] = (trd_ranges[j][1] == p->size_z);            
  }

  for(int j=0; j<3; j++)
  {
    o_start[j] = trd_ranges[j][0];
    o_size[j]  = trd_ranges[j][1] - trd_ranges[j][0];
  }

  if(m_ptchDec.m_idToGridX[p_id] == 0 && trd_start[0])
  {
    o_start[0] += h;
    o_size[0] -= h;
  }
  if(m_ptchDec.m_idToGridY[p_id] == 0 && trd_start[1])
  {
    o_start[1] += h;
    o_size[1] -= h;
  }
  if(m_ptchDec.m_idToGridZ[p_id] == 0 && trd_start[2])
  {
    o_start[2] += h;
    o_size[2] -= h;
  }

  if(m_ptchDec.m_idToGridX[p_id] == m_ptchDec.m_numXPatches - 1 && trd_end[0])
  {
    o_size[0] -= h;
  }
  if(m_ptchDec.m_idToGridY[p_id] == m_ptchDec.m_numYPatches - 1 && trd_end[1])
  {
    o_size[1] -= h;
  }
  if(m_ptchDec.m_idToGridZ[p_id] == m_ptchDec.m_numZPatches - 1 && trd_end[2])
  {
    o_size[2] -= h;
  }
}

bool odc::parallel::OpenMP::isOnXMaxBdry(int i_ptch)
{
  Patch& p = m_ptchDec.m_patches[i_ptch];
  int trd_id = getThreadNumGrp();
  int_pt trd_ranges[3][2];
  getRangesTrd(i_ptch, trd_id, trd_ranges);

  // TODO(Josh): move this assert somewhere more appropriate, eg. init code
  // The purpose of this is to make sure that we can determine if a thread range
  // is at the end of a patch, without doing any boundary size computation gymnastics 
  assert(p.nx > 2*p.bdry_width);
  
  return ((m_ptchDec.m_idToGridX[i_ptch] == m_ptchDec.m_numXPatches-1)
          && (trd_ranges[0][1] >= p.nx)); 
}

bool odc::parallel::OpenMP::isOnYZeroBdry(int i_ptch)
{
  Patch& p = m_ptchDec.m_patches[i_ptch];
  int trd_id = getThreadNumGrp();
  int_pt trd_ranges[3][2];
  getRangesTrd(i_ptch, trd_id, trd_ranges);

  return ((m_ptchDec.m_idToGridY[i_ptch] == 0)
          && (trd_ranges[1][0] == 0)); 
}

bool odc::parallel::OpenMP::isOnZBdry(int i_ptch)
{
  Patch& p = m_ptchDec.m_patches[i_ptch];
  int trd_id = getThreadNumGrp();
  int_pt trd_ranges[3][2];
  getRangesTrd(i_ptch, trd_id, trd_ranges);

  // TODO(Josh): I think this should be the end of the domain instead? CHECK THIS!
  return ((m_ptchDec.m_idToGridZ[i_ptch] == 0)
          && (trd_ranges[2][0] == 0)); 
}


