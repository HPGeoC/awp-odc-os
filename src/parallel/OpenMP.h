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

#ifndef OPENMP_H
#define OPENMP_H

// TODO(Josh): find the appropriate place for this
#define WP_SIZE_X 16
#define WP_SIZE_Y 16
#define WP_SIZE_Z 64


#include "data/PatchDecomp.hpp"
#include "parallel/Mpi.hpp"
#include "constants.hpp"

namespace odc {
  namespace parallel {
    class OpenMP;
    class WorkPackage;
    class OmpManager;

    enum WorkPackageType : unsigned short;
  }
}

class odc::parallel::OpenMP {

  
  
  //! store patch decomposition layout 
  PatchDecomp& m_ptchDec;
    
  /*
   * Unique data for every thread.
   */
  //! thread number w.r.t. to all work groups
  int m_trdNumAll;

  //! thread number in the respective local work group
  int m_trdNumGrp;

  //! work group of the thread
  int m_trdWgrpAll;

  // local id in the comp- or non-comp work groups
  int m_trdWgrpFun;

  //! true of work group of the thread is computational
  bool m_trdWgrpComp;

  /*
   * Unique but redundant data for all threads.
   */
  //! number of threads in all work groups
  int m_nThreadsAll;

  //! number of threads in all computational work groups
  int m_nThreadsComp;

  //! number of work groups
  const int m_nWgrpsAll;

  //! number of work groups participating in computation
  int m_nWgrpsComp;

  //! number of threads per work group (for all work groups)
  int* m_nTdsPerWgrpAll;

  //! number of threads per computatonal work group
  int* m_nTdsPerWgrpComp;

  //! layout of the domain decompostion w.r.t. to the threads in the comp work groups
  int (**m_domDecCompTds)[3];

  //! ranges of points covered by the computational threads
  int_pt (**m_rangeCompTds)[3][2];

  //! match patch with its owner workgroup  
  int* m_ptchToWgrpComp;

  //! number of patches owned by each computational workgroup
  int_pt* m_nPtchsPerWgrpComp;

  //! list of patches owned by each computational workgroup
  int_pt** m_wgrpPtchList;
  
  

  /**
   * Partitions a given 3D-hexahedral domain of points.
   *
   * @param i_parts number of partitions.
   * @param i_offX offset in x-direction.
   * @param i_offY offset in y-direction.
   * @param i_offZ offset in z-direction.
   * @param i_nPtsX number of pts in x-direction.
   * @param i_nPtsY number of pts in y-direction.
   * @param i_nPtsZ number of pts in z-direction.
   * @param o_domDec location of partition w.r.t. to total number of partitions for this decomp.
   * @param o_range range of pts covered by a partition. [*][]: x-, y-, z-dim; [][*]: min,(max+1)
   **/
  void partDomain( int_pt i_nParts,
		   int_pt i_offX,
		   int_pt i_offY,
		   int_pt i_offZ,
		   int_pt i_nPtsX,
		   int_pt i_nPtsY,
		   int_pt i_nPtsZ,
		   int    (*o_domDec)[3],
		   int_pt (*o_range)[3][2] );

  /**
   * Prints the layout of a partitioning.
   *
   * @param i_nParts number of partitions.
   * @param i_domDec decomposition into partitions.
   * @param i_range ranges covered by the partitions.
   **/
//    void printLayoutPart( int      i_nParts,
//                          int    (*i_domDec)[3],
//                          int_pt (*i_range)[3][2] );

  /**
   * Derives the computational layout.
   *
   * @param i_nPtsX number of points in x-direction for the entire MPI-partition.
   * @param i_nPtsY number of points in y-direction for the entire MPI-partition.
   * @param i_nPtsZ number of points in z-direction for the entire MPI-partition.
   * @param i_nWgrpsComp number of computational work groups.
   * @param i_nTdsPerWgrpComp #threads per computational work group.
   * @param o_domDecWgrps will be set to domain decomposion w.r.t. to the computational work groups.
   * @param o_rangeCompWgrps will be set to ranges covered by the comp. work groups.
   * @param o_domDecCompTds will bet set to domain dec. w.r.t. to the threads in the computational work groups.
   * @param o_rangeCompTds will bet set to ranges covered by the individual threads in the comp. worg grps.
   **/
  void deriveLayout(int_pt    i_nPtsX,
		    int_pt    i_nPtsY,
		    int_pt    i_nPtsZ,
		    int       i_nWgrpsComp,
		    int      *i_nTdsPerWgrpComp);


public:

  int_pt m_nPtsX;
  int_pt m_nPtsY;
  int_pt m_nPtsZ;
  
  
  //! array of work packages that must be completed each timestep
  odc::parallel::WorkPackage* m_workPackages;

  //! x size of work package region
  int m_packageSizeX;

  //! y size of work package region
  int m_packageSizeY;

  //! z size of work package region
  int m_packageSizeZ;
  
  //!
  int m_nWP;

  //!
  int m_nWPX;

  //!
  int m_nWPY;

  //!
  int m_nWPZ;

  //! number of work packages required to compute the x,y boundaries
  int m_nBdry;

  //! index of the end of last WP required for velocity boundary
  int m_endOfVelBdry;

  //! index of velocity mpi communication WP
  int m_velMpiWP;

  //! index of the end of last WP required for stress boundary
  int m_endOfStressBdry;

  //! index of the end of last WP required for stress boundary x=0;  need a barrier here depending on all vel being done
  int m_endOfStressBdryXZero;
  
  
  //! index of stress mpi communication WP
  int m_stressMpiWP;

  bool isComputationThread()
  {
    return (m_trdNumAll != 0);
  }

  
  /**
   * Initializes the OpenMP-parallelization for a calling thread.
   * Should be called from with a parallel region only.
   *
   * Work groups summarize the utilization of OMP threads.
   * Assignment of OMP threads to work groups is ascending.
   * Negative thread number relate to work groups not participating in computation.
   * Example:
   *  #wgrps          :  4
   *  #thread per wgrp:  4, -2, 8, -1
   *
   * Wgrp#1 participates in computations. Assigned OMP threads are 0,1,2,3.
   * Wgrp#2 does not participate in computation. Assigned OMP threads are 4,5
   * Wgrp#3 particaptes in computations. Assigned OMP threads are 6,7,8,9,10,11,12,13
   * Wgrp#4 does not participate in computations. Assigned OMP thread is 14.
   *
   * @param i_nPtsX number of points in x-direction.
   * @param i_nPtsY number of points in y-direction.
   * @param i_nPtsZ number of points in z-direction.
   * @param i_nWGrps number of work groups.
   * @param i_tdsPerWgrp #threads for every workgroup.
   **/
  OpenMP( int_pt       i_nPtsX,
	  int_pt       i_nPtsY,
	  int_pt       i_nPtsZ,
	  int          i_nManageThreads,
	  int          i_nCompThreads,
	  PatchDecomp& i_ptchDec);

  /**
   * Frees dynamic memory allocations.
   **/
  ~OpenMP();

  /**
   * Gets the thread number w.r.t. all thread.
   *
   * @return thread number.
   **/
  int getThreadNumAll() { return m_trdNumAll; };

  /**
   * Gets the local thread number in its working group.
   *
   * @return thread number.
   **/
  int getThreadNumGrp() { return (m_trdNumAll == 0) ? 0 : (m_trdNumAll-1); };

  /**
   * Gets the working group of the thread with respect to all groups.
   *
   * @return work group.
   **/
  int getWgrpAll() { return m_trdWgrpAll; };

  /**
   * Gets the functional working group of the thread.
   *
   * @return functional working group of the thread.
   **/
  int getWgrpFun() { return m_trdWgrpFun; };

  /**
   * Returns true if the calling thread is thread part of a computational work group
   *
   * @return true if comp thread, false otherwise.
   **/
  bool compThread() { return m_trdWgrpComp; };

  /**
   * Gets the number of computational threads.
   *
   * @return number of computational threads.
   **/
  int getNThreadsComp() { return m_nThreadsComp; };

  /**
   * Gets the x-, y- and z- ranges of the given computational wgrp with respect to the entire mesh.
   *
   * @param i_wGrpComp computaitoanl work group.
   * @param o_ranges will be set ranges of the work group.
   **/
//    void getRangesGrp( int    i_wGrpComp,
//                      int_pt o_ranges[3][2] ) {
//      for( int l_dim = 0; l_dim < 3; l_dim++ ) {
//        o_ranges[l_dim][0] = m_rangeCompWgrps[i_wGrpComp][l_dim][0];
//        o_ranges[l_dim][1] = m_rangeCompWgrps[i_wGrpComp][l_dim][1];
//      }
//    };

  /**
   * Gets the x-, y- and z- ranges of the given computational thread with respect to the groups local mesh.
   *
   * @param i_wGrpFun computaitional work group of the thread.
   * @param i_trdNumGrp local thread num of the thread w.r.t. to the group
   * @param o_ranges will be set ranges of the thread.
   **/
//    void getRangesPtch( int    i_wGrpFun,
//                       int    i_ptch,
//                       int_pt o_ranges[3][2] ) {
//      for( int l_dim = 0; l_dim < 3; l_dim++ ) {
//        o_ranges[l_dim][0] = m_rangeCompPtchs[i_wGrpFun][i_ptch][l_dim][0];
//        o_ranges[l_dim][1] = m_rangeCompPtchs[i_wGrpFun][i_ptch][l_dim][1];
//      }
//    };

  void getRangesTrd(
    int    i_ptch,
    int    i_trd,
    int_pt o_ranges[3][2] ) {
    for( int l_dim = 0; l_dim < 3; l_dim++ ) {
      o_ranges[l_dim][0] = m_rangeCompTds[i_ptch][i_trd][l_dim][0];
      o_ranges[l_dim][1] = m_rangeCompTds[i_ptch][i_trd][l_dim][1];
    }
  };


  void getTrdExtent(int i_ptch, int_pt o_start[3], int_pt o_size[3]);

  bool isOnXMaxBdry(int i_ptch);

  bool isOnYZeroBdry(int i_ptch);
  
  bool isOnZBdry(int i_ptch);

  int_pt maxNumPtchsPerWgrp()
  {
    return m_nPtchsPerWgrpComp[0];
  }

  bool participates(int_pt i_ptchIndex)
  {
    return (m_nPtchsPerWgrpComp[m_trdWgrpFun] > i_ptchIndex);
  }

  int_pt getPatchNumber(int_pt ptch)
  {
    return (m_wgrpPtchList[m_trdWgrpFun][ptch]);
  }

};

/**
 * Labels for the different types of work packages.  
 * Note(Josh): The long names make me squirm but that seems to be the style we are going with.
 **/
enum odc::parallel::WorkPackageType : unsigned short
{
  WP_VelUpdate,
  WP_StressUpdate,
  WP_FreeSurface_VelUpdate,
  WP_FreeSurface_StressUpdate,
  WP_MPI_Vel,
  WP_MPI_Stress    
};

/**
 * Mostly a placeholder class for when WorkPackages are more complex
 **/
class odc::parallel::WorkPackage
{
public:
  //! type of work package
  odc::parallel::WorkPackageType type;

  //! coordinates of start of region
  int_pt start[3];

  //! coordinates of end of region
  int_pt end[3];

  //! flag indicates to copy from MPI buffer before computation
  bool copyFromBuffer;

  //! flag indicates to copy from MPI buffer before computation
  bool copyToBuffer;

  //! array stores which directions MPI buffers exist in 
  bool mpiDir[3][3][3];

  //! flag indicates to do free surface updates on this block
  bool freeSurface;

  //! flag indicates whether this block is on x max boundary of entire comp domain
  bool xMaxBdry;

  //! flag indicates whether this block is on y min boundary of entire comp domain
  bool yMinBdry;
  
};

class odc::parallel::OmpManager
{
public:
  OmpManager(int nCompThreads, int nWP, odc::parallel::OpenMP& omp)
    : m_nCompThreads(nCompThreads), m_nWP(nWP), m_nextWP(0), m_omp(omp)
  {
    m_completedWP = (int*) malloc(nWP * sizeof(int));

    m_nBarriers = 4;
    m_indexToBarrier =    (int*) malloc(nWP * sizeof(int));
    m_barrierDependency = (int*) malloc(m_nBarriers * sizeof(int));
    m_barrierLeft =       (int*) malloc(m_nBarriers * sizeof(int));

    for(int i=0; i<nWP; i++)
      m_indexToBarrier[i] = -1;

    int barrier = m_omp.m_nBdry;

    m_indexToBarrier[m_omp.m_velMpiWP] = 0;
    m_barrierDependency[0] = m_omp.m_endOfVelBdry;
    m_barrierLeft[0] = m_barrierDependency[0] + 1;

    m_indexToBarrier[nWP/2] = 1;
    m_barrierDependency[1] = m_omp.m_velMpiWP;
    m_barrierLeft[1] = m_barrierDependency[1] + 1;

    m_indexToBarrier[m_omp.m_endOfStressBdryXZero+1] = 2;
    m_barrierDependency[2] = nWP/2 - 1;
    m_barrierLeft[2] = m_barrierDependency[2] + 1;
    
    m_indexToBarrier[m_omp.m_stressMpiWP] = 3;
    m_barrierDependency[3] = m_omp.m_endOfStressBdry;
    m_barrierLeft[3] = m_barrierDependency[3] + 1;

    // Useful for debugging 
#if 0
#pragma omp critical
{
    std::cout << "barriers: " << std::endl;
    for(int i=0; i<nWP; i++)
    {
      if(m_indexToBarrier[i] >= 0)
      {
	std::cout << i << ' ' << m_barrierDependency[m_indexToBarrier[i]] << std::endl;
      }
    }
}
#endif

  }

  ~OmpManager()
  {
    free(m_completedWP);
    free(m_indexToBarrier);
    free(m_barrierDependency);
    free(m_barrierLeft);
  }

  int m_nCompThreads;
  int m_nWP;
  int m_nextWP;
  
  int* m_completedWP;

  //! stores the number of custom barriers in one timestep in our dynamic OpenMP
  int m_nBarriers;
  
  //! for each WP stores either -1 to say no barrier here, or the index of the barrier in the arrays below
  int* m_indexToBarrier;

  //! for a barrier index, returns which WP index must be completed (*all* WPs before this index) before proceeding
  int* m_barrierDependency;

  //! for a barrier index, stores the number of WPs that are still outstanding before this barrier will open
  int* m_barrierLeft;

  odc::parallel::OpenMP& m_omp;
  
  
  /**
   * Sets the initial distribution of work packages.  Also resets data structures.
   *
   * @param i_nextWP an array of size [num_comp_threads][2] that stores wp assignments
   **/
  void initialWorkPackages(volatile int* i_nextWP)
  {
    // reset data structures
    m_nextWP = 0;
    for(int i=0; i<m_nWP; i++)
      m_completedWP[i] = 0;

    // for a barrier to open, the number of dependencies is the index of the dependency plus one
    for(int i=0; i<m_nBarriers; i++)
      m_barrierLeft[i] = m_barrierDependency[i]+1;


    // PPP: for initial assignments, make sure barriers aren't violated (only an issue when number of
    //      WPs is pathologically small, but should be careful)
    for(int i=0; i<2; i++)
    {
      for(int j=0; j<m_nCompThreads; j++)
      {
	if(m_nWP > m_nextWP)
  	  i_nextWP[2*j + i] = ++m_nextWP;
	else
	  i_nextWP[2*j + i] = m_nWP+1;
      }
    }
  }

  /**
   * Returns current number of tasks left.
   **/  
  int tasksLeft()
  {
    return m_nWP - m_nextWP;
  }

  /**
   * Check if the next task is behind a closed barrier, if not return that task and increment next task, otherwise return -1
   **/  
  int nextTask()
  {
    if(m_indexToBarrier[m_nextWP] == -1 || m_barrierLeft[m_indexToBarrier[m_nextWP]] == 0)
      return m_nextWP++;
    else
      return -1;
  }

  /**
   * Marks a work package as having been completed.
   *
   * @param i_wp is the (index+1) of completed work package
   **/
  void setDone(int i_wp)
  {
    // since i_wp is index+1 for signalling reasons, first recover index
    i_wp--;
    
    if(!m_completedWP[i_wp])
    {
      m_completedWP[i_wp] = 1;
      
      // if this WP is a dependency for any barriers, reduce the number of remaining dependencies for those barriers by one
      for(int i=0; i < m_nBarriers; i++)
      {
	if(i_wp <= m_barrierDependency[i])
	  m_barrierLeft[i]--;
      }
    }
  }  
};

#endif
