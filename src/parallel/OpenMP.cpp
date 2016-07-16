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

odc::parallel::OpenMP::OpenMP( int_pt       i_nPtsX,
                               int_pt       i_nPtsY,
                               int_pt       i_nPtsZ,
                               int          i_nWgrps,
                               int         *i_tdsPerWgrp,
                               PatchDecomp& i_ptchDec):
m_nWgrpsAll(i_nWgrps), m_ptchDec(i_ptchDec) {
  if( !omp_in_parallel() )
    std::cerr << "Error: OpenMP constructor should be called in a parallel region." << std::endl;

  // copy threads per work group data
  m_nTdsPerWgrpAll = (int*) malloc( m_nWgrpsAll * sizeof( int ) );
  for( int l_wg = 0; l_wg < m_nWgrpsAll; l_wg++ ) {
    m_nTdsPerWgrpAll[l_wg] = i_tdsPerWgrp[l_wg];
  }

  m_trdNumAll = omp_get_thread_num();
  m_nThreadsAll = omp_get_num_threads();

  // check for valid number of threads for temporal shared memory parallelization
  int l_nThreads = 0;
  for( int l_gr = 0; l_gr < m_nWgrpsAll; l_gr++ ) {
    l_nThreads += std::abs( m_nTdsPerWgrpAll[l_gr] );
  }
  std::cout << "num threads: " << m_nThreadsAll << std::endl;
  assert( l_nThreads == m_nThreadsAll );

  // derive number of work groups participating in computation
  m_nWgrpsComp = 0;
  m_nThreadsComp = 0;
  for( int l_gr = 0; l_gr < m_nWgrpsAll; l_gr++ ) {
    if( m_nTdsPerWgrpAll[l_gr] > 0 ) {
      m_nWgrpsComp++;
      m_nThreadsComp += m_nTdsPerWgrpAll[l_gr];
    }
    else if( m_nTdsPerWgrpAll[l_gr] == 0 ) assert(false);
  }
  assert( m_nWgrpsComp > 0 );

  // store thread info for comp work groups only
  m_nTdsPerWgrpComp = (int*) malloc( m_nWgrpsComp * sizeof(int) );
  int l_compWgrp = 0;
  for( int l_gr = 0; l_gr < m_nWgrpsAll; l_gr++ ) {
    if( m_nTdsPerWgrpAll[l_gr] > 0 ) {
      m_nTdsPerWgrpComp[l_compWgrp] = m_nTdsPerWgrpAll[l_gr];
      l_compWgrp++;
    }
  }

  // derive charactericistics of this thread
  int l_prevThreads = 0;
  int l_touchedWgrpsComp = 0;
  for( int l_gr = 0; l_gr < m_nWgrpsAll; l_gr++ ) {
    if( m_trdNumAll >= l_prevThreads &&
        m_trdNumAll <  l_prevThreads + std::abs( m_nTdsPerWgrpAll[l_gr] ) ) {
      m_trdWgrpAll = l_gr;

      if( m_nTdsPerWgrpAll[l_gr] > 0 ) {
        // thread is part of a computational work group
        m_trdWgrpFun = l_touchedWgrpsComp;
        m_trdWgrpComp = true;
      }
      else {
        // thread is not part of a computational work group
        m_trdWgrpFun = l_gr - l_touchedWgrpsComp;
        m_trdWgrpComp = false;
      }

      // get thread id in the local work group
      m_trdNumGrp = m_trdNumAll - l_prevThreads;
    }

    // increased #touched wgrps if required
    if( m_nTdsPerWgrpAll[l_gr] > 0 ) l_touchedWgrpsComp++;

    // add #threads for this wgrp to counter
    l_prevThreads += std::abs( m_nTdsPerWgrpAll[l_gr] );
  }
  
  deriveLayout( i_nPtsX,
                i_nPtsY,
                i_nPtsZ,
                m_nWgrpsComp,
                m_nTdsPerWgrpComp);

}

odc::parallel::OpenMP::~OpenMP() {
  free( m_nTdsPerWgrpAll  );
  free( m_nTdsPerWgrpComp );
  free( m_nPtchsPerWgrpComp );
  free( m_ptchToWgrpComp  );

  int_pt numPatches = m_ptchDec.m_numPatches;
  for( int_pt l_ptch = 0; l_ptch < numPatches; l_ptch++ ) {
    free(m_domDecCompTds[l_ptch]);
    free(m_rangeCompTds[l_ptch]);
  }
  free(m_domDecCompTds);
  free(m_rangeCompTds);

  for( int l_wgrp = 0; l_wgrp < m_nWgrpsComp; l_wgrp++ ) {
    free(m_wgrpPtchList[l_wgrp]);
  }
  free(m_wgrpPtchList);
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



/*void odc::parallel::OpenMP::printLayoutPart( int      i_nParts,
                                             int    (*i_domDec)[3],
                                             int_pt (*i_range)[3][2]  ) {
  // get n-regions
  int l_nRegX = 0;
  int l_nRegY = 0;
  int l_nRegZ = 0;

  for( int l_part = 0; l_part < i_nParts; l_part++ ) {
    l_nRegX = std::max( i_domDec[l_part][0], l_nRegX );
    l_nRegY = std::max( i_domDec[l_part][1], l_nRegY );
    l_nRegZ = std::max( i_domDec[l_part][2], l_nRegZ );
  }
  l_nRegX+=1; l_nRegY+=1; l_nRegZ+=1;

  // print the thread layout
  for( int l_y = l_nRegY-1; l_y >= 0; l_y-- ) {
    for( unsigned int l_dim = 0; l_dim < 3; l_dim++ ) {
      for( int l_z = 0; l_z < l_nRegZ; l_z++ ) {
        for( int l_x = 0; l_x < l_nRegX; l_x++ ) {

          // find partition
          int l_part = -1;
          for( int l_do = 0; l_do < i_nParts; l_do++ ) {
            if( i_domDec[l_do][0] == l_x &&
                i_domDec[l_do][1] == l_y &&
                i_domDec[l_do][2] == l_z ) l_part = l_do;
          }
          if( l_dim == 0 ) std::cout << std::setw(3) << l_part << ": ";
          else  std::cout << std::setw(5) << " ";
          std::cout << std::setw(5) << i_range[l_part][l_dim][0] << " "
                    << std::setw(5) << i_range[l_part][l_dim][1] << " |";
        }

        std::cout << " -" << ( (l_dim==0) ? "x" : (l_dim==1 ?  "y" : "z") ) << "- ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  }*/

/*void odc::parallel::OpenMP::printLayout() {
  std::cout << "printing omp layout" << std::endl;
  std::cout << "#wgrps: " << m_nWgrpsAll << " #(comp wgrps:) " << m_nWgrpsComp << std::endl;

  std::cout << std::endl << "here's the decomposition into work groups" << std::endl;
  printLayoutPart( m_nWgrpsComp,
                   m_domDecCompWgrps,
                  m_rangeCompWgrps );

  std::cout << std::endl << "time for the layout of the individual workgroups into patches" << std::endl;

  for( int l_wg = 0; l_wg < m_nWgrpsComp; l_wg++ ) {
    std::cout << "wg: " << l_wg << std::endl;
    printLayoutPart( m_nPtchsPerWgrpComp[l_wg],
                   m_domDecCompPtchs[l_wg],
                   m_rangeCompPtchs[l_wg] );
  }

  std::cout << std::endl << "time for the layout of the individual patches into threads" << std::endl;

  for( int l_wg = 0; l_wg < m_nWgrpsComp; l_wg++ ) {
    for( int l_ptch = 0; l_ptch < m_nPtchsPerWgrpComp[l_wg]; l_ptch++) {
      std::cout << "wg: " << l_wg << "; ptch: " << l_ptch << std::endl;
      printLayoutPart( m_nTdsPerWgrpComp[l_wg],
                     m_domDecCompTds[l_wg][l_ptch],
                     m_rangeCompTds[l_wg][l_ptch]);
    }
  }
  
  
  }*/
