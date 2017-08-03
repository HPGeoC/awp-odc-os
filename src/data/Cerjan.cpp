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

#include "parallel/Mpi.hpp"
#include "Cerjan.hpp"
#include "Grid.hpp"
#include <cmath>
#include <cstdio>

odc::data::Cerjan::Cerjan(odc::io::OptionParser i_options, SoA i_data) {
    
    m_spongeCoeffX = Alloc1D(i_data.m_numXGridPoints, odc::constants::boundary);
    m_spongeCoeffY = Alloc1D(i_data.m_numYGridPoints, odc::constants::boundary);
    m_spongeCoeffZ = Alloc1D(i_data.m_numZGridPoints, odc::constants::boundary);
    
    for(int i=-odc::constants::boundary; i<i_data.m_numXGridPoints+odc::constants::boundary; i++) {
        m_spongeCoeffX[i]  = 1.0;
    }
    
    for(int j=-odc::constants::boundary; j<i_data.m_numYGridPoints+odc::constants::boundary; j++) {
        m_spongeCoeffY[j]  = 1.0;
    }
    
    for(int k=-odc::constants::boundary; k<i_data.m_numZGridPoints+odc::constants::boundary; k++) {
        m_spongeCoeffZ[k]  = 1.0;
    }
    
    int_pt coords[] = {odc::parallel::Mpi::m_startX,odc::parallel::Mpi::m_startY,odc::parallel::Mpi::m_startZ}; // in this legacy code, a single node is assumed
    inicrj(i_options.m_arbc, coords, i_data.m_numXGridPoints,
           i_data.m_numYGridPoints, i_data.m_numZGridPoints, i_data.m_numXGridPoints,
           i_data.m_numYGridPoints, i_options.m_nD, m_spongeCoeffX,
           m_spongeCoeffY, m_spongeCoeffZ);
    
}

void odc::data::Cerjan::initialize(odc::io::OptionParser i_options, int_pt nx, int_pt ny, int_pt nz,
                                   int_pt bdry_width, int_pt *coords) {
    m_spongeCoeffX = Alloc1D(nx + 2*bdry_width, odc::constants::boundary);
    m_spongeCoeffY = Alloc1D(ny + 2*bdry_width, odc::constants::boundary);
    m_spongeCoeffZ = Alloc1D(nz + 2*bdry_width, odc::constants::boundary);

    for(int i=-odc::constants::boundary; i < nx + 2*bdry_width + odc::constants::boundary; i++) {
        m_spongeCoeffX[i]  = 1.0;
    }
    
    for(int j=-odc::constants::boundary; j < ny + 2*bdry_width + odc::constants::boundary; j++) {
        m_spongeCoeffY[j]  = 1.0;
    }
    
    for(int k=-odc::constants::boundary; k < nz + 2*bdry_width + odc::constants::boundary; k++) {
        m_spongeCoeffZ[k]  = 1.0;
    }

    inicrj(i_options.m_arbc, coords, nx, ny, nz, i_options.m_nX, i_options.m_nY, i_options.m_nD, 
           m_spongeCoeffX + bdry_width, m_spongeCoeffY + bdry_width, m_spongeCoeffZ + bdry_width);
}


void odc::data::Cerjan::inicrj(float ARBC,                               // command line option .m_arbc
                               int_pt *coords,                              // coordinates of first element in _this_ Cerjan
                               int_pt nxt, int_pt nyt, int_pt nzt,       // dimensions of _this_ Cerjan object 
                               int_pt NX, int_pt NY,                     // x,y dimension of whole computational domain
                               int_pt ND,                                // width for Cerjan (default is 20)
                               Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz  // Cerjan arrays
                               ) {
    int nxp, nyp, nzp;
    int i,   j,   k;
    float alpha;
    alpha = sqrt(-log(ARBC))/ND;
    
    nxp   = coords[0] + 1;
    if(nxp <= ND)
    {
        for(i=0;i<ND;i++)
        {
            nxp        = i + 1;
            dcrjx[i] = dcrjx[i]*(exp(-((alpha*(ND-nxp+1))*(alpha*(ND-nxp+1)))));
        }
    }
    nxp   = coords[0] + 1;
    if( (nxp+nxt-1) >= (NX-ND+1))
    {
        for(i=nxt-ND;i<nxt;i++)
        {
            nxp        = i + NX - nxt + 1;
            dcrjx[i] = dcrjx[i]*(exp(-((alpha*(ND-(NX-nxp)))*(alpha*(ND-(NX-nxp))))));
        }
    }
    
    nyp   = coords[1] + 1;
    if(nyp <= ND)
    {
        for(j=0;j<ND;j++)
        {
            nyp        = j + 1;
            dcrjy[j] = dcrjy[j]*(exp(-((alpha*(ND-nyp+1))*(alpha*(ND-nyp+1)))));
        }
    }
    nyp   = coords[1] + 1;
    if((nyp+nyt-1) >= (NY-ND+1))
    {
        for(j=nyt-ND;j<nyt;j++)
        {
            nyp        = j + NY - nyt + 1;
            dcrjy[j] = dcrjy[j]*(exp(-((alpha*(ND-(NY-nyp)))*((alpha*(ND-(NY-nyp)))))));
        }
    }
    
    nzp = 1;
    if(nzp <= ND)
    {
        for(k=0;k<ND;k++)
        {
            nzp            = k + 1;
            dcrjz[k] = dcrjz[k]*(exp(-((alpha*(ND-nzp+1))*((alpha*(ND-nzp+1))))));
        }
    }
    return;
}


// TODO(Josh): looks like Cerjan is never freed?
void odc::data::Cerjan::finalize() {
   
}
