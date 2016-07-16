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

#include "data/PatchDecomp.hpp"
#include "constants.hpp"

namespace odc {
    namespace parallel {
        class OpenMP;
    }
}

class odc::parallel::OpenMP{
  //private:

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
            int          i_nWgrps,
            int         *i_tdsPerWgrp,
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
    int getThreadNumGrp() { return m_trdNumGrp; };

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

    /**
     * Prints the layout of work groups and all threads associated to the work groups.
     **/
//    void printLayout();
};

#endif
