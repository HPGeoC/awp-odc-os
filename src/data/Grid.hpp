/**
 @author Gautam Wilkins
 
 @section DESCRIPTION
 Main file.
 
 @section LICENSE
 Copyright (c) 2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef GRID_HPP
#define GRID_HPP

#include "constants.hpp"

namespace odc {
    namespace data {
        
        Grid3D Alloc3D(int_pt nx, int_pt ny, int_pt nz, int_pt boundary);
        
        Grid3Dww Alloc3Dww(int_pt nx, int_pt ny, int_pt nz, int_pt boundary);
        Grid1D Alloc1D(int_pt nx, int_pt boundary);
        PosInf Alloc1P(int_pt nx, int_pt boundary);
        
        void Delloc3D(Grid3D U, int_pt boundary);
        void Delloc3Dww(Grid3Dww U, int_pt boundary);
        void Delloc1D(Grid1D U, int_pt boundary);
        void Delloc1P(PosInf U, int_pt boundary);
        
        
        Grid3D Alloc3D(int_pt nx, int_pt ny, int_pt nz);
        Grid3Dww Alloc3Dww(int_pt nx, int_pt ny, int_pt nz);
        Grid1D Alloc1D(int_pt nx);
        PosInf Alloc1P(int_pt nx);
        
        void Delloc3D(Grid3D U);
        void Delloc3Dww(Grid3Dww U);
        void Delloc1D(Grid1D U);
        void Delloc1P(PosInf U);


#ifdef USING_YASK
        void CopyFromYASKGrid(Grid3D grid, RealvGridBase* yaskGrid,
                              int_pt xStart, int_pt yStart, int_pt zStart,
                              int_pt nx, int_pt ny, int_pt nz);
        void WriteToYASKGrid(Grid3D grid, RealvGridBase* yaskGrid,
                             int_pt xStart, int_pt yStart, int_pt zStart,
                             int_pt nx, int_pt ny, int_pt nz);
#endif
        
    }
}







#endif /* Grid_hpp */
