/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Struct of Array (SoA) data representation.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SOA_HPP
#define SOA_HPP

#include "constants.hpp"
#include "common.hpp"
#include "io/OptionParser.h"
#include "Grid.hpp"

namespace odc {
    namespace data {
        class SoA;
    }
}

class odc::data::SoA {
public:
    /**
     * Meta data.
     **/
    // number of grid points
    int_pt m_numGridPoints;
    
    // number of x grid points
    int_pt m_numXGridPoints;
    
    // number of y grid points
    int_pt m_numYGridPoints;
    
    // number of z grid points
    int_pt m_numZGridPoints;
    
    /**
     * Solution data.
     **/
    // velocity in x-, y- and z-direction.
    Grid3D m_velocityX, m_velocityY, m_velocityZ;
    
    Grid3D m_stressXX, m_stressYY, m_stressZZ;
    Grid3D m_stressXY, m_stressXZ, m_stressYZ;
    
    // normal stress components
    //real *m_stressXX, *m_stressYY, *m_stressZZ;
    
    // shear stress components
    //real *m_stressXY, *m_stressXZ, *m_stressYZ;
    
    // attenuation
    Grid3D m_memXX, m_memYY, m_memZZ, m_memXY, m_memXZ, m_memYZ;
    
    
    /**
     * Initializes the meta data.
     *
     * @param i_options     Parsed command line options
     **/
    void initialize( int_pt i_numPointsX,
                     int_pt i_numPointsY,
                     int_pt i_numPointsZ ) {
        m_numXGridPoints = i_numPointsX;
        m_numYGridPoints = i_numPointsY;
        m_numZGridPoints = i_numPointsZ;
        
        m_numGridPoints = m_numXGridPoints * m_numYGridPoints * m_numZGridPoints;
    }
    
    /**
     * Derives the memory requirements.
     *
     * @return required size in GiB
     **/
    float getSize() {
        return ( m_numGridPoints * sizeof(real) * 15.0 ) / (1024 * 1024 * 1024);
    }
    
    
    /**
     * Allocates the memory.
     **/
    void allocate() {
        
        m_velocityX = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_velocityY = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_velocityZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        
        m_stressXX = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_stressYY = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_stressZZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        
        m_stressXY = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_stressXZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_stressYZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        
        m_memXX = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_memYY = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_memZZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_memXY = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_memXZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        m_memYZ = odc::data::Alloc3D(m_numXGridPoints, m_numYGridPoints, m_numZGridPoints, odc::constants::boundary);
        
    }
    
    
    
    /**
     * Finalizes the data structures.
     **/
    void finalize() {
        odc::data::Delloc3D(m_velocityX, odc::constants::boundary);
        odc::data::Delloc3D(m_velocityY, odc::constants::boundary);
        odc::data::Delloc3D(m_velocityZ, odc::constants::boundary);
        
        odc::data::Delloc3D(m_stressXX, odc::constants::boundary);
        odc::data::Delloc3D(m_stressYY, odc::constants::boundary);
        odc::data::Delloc3D(m_stressZZ, odc::constants::boundary);
        
        odc::data::Delloc3D(m_stressXY, odc::constants::boundary);
        odc::data::Delloc3D(m_stressXZ, odc::constants::boundary);
        odc::data::Delloc3D(m_stressYZ, odc::constants::boundary);
        
        odc::data::Delloc3D(m_memXX, odc::constants::boundary);
        odc::data::Delloc3D(m_memYY, odc::constants::boundary);
        odc::data::Delloc3D(m_memZZ, odc::constants::boundary);
        odc::data::Delloc3D(m_memXY, odc::constants::boundary);
        odc::data::Delloc3D(m_memXZ, odc::constants::boundary);
        odc::data::Delloc3D(m_memYZ, odc::constants::boundary);

    }
};

#endif
