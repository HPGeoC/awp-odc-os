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


#ifndef AWP_ODC_OS_CERJAN_HPP
#define AWP_ODC_OS_CERJAN_HPP

#include "constants.hpp"
#include "io/OptionParser.h"
#include "data/SoA.hpp"

namespace odc {
    namespace data {
        class Cerjan;
    }
}


class odc::data::Cerjan {
    
public:
    Grid1D m_spongeCoeffX;
    Grid1D m_spongeCoeffY;
    Grid1D m_spongeCoeffZ;

    Cerjan() {};
    Cerjan(io::OptionParser i_options, SoA i_data);

    void initialize(io::OptionParser i_options, int_pt nx, int_pt ny, int_pt nz, int_pt bdry_width, int_pt *coords);
    void finalize();
    
private:
    void inicrj(float ARBC, int_pt *coords, int_pt nxt, int_pt nyt, int_pt nzt, int_pt NX, int_pt NY, int_pt ND, Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz);
    
    
    
    
    
};





#endif /* Cerjan_hpp */
