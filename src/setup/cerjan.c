/**
 @brief Calculates absorbing boundary coefficients using method from Cerjan et al. (1985)
 
 @section LICENSE
 Copyright (c) 2013-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <math.h>
#include <stdio.h>
#include "pmcl3d.h"

/**
 @brief Initializes sponge layer absorbing boundary conditions to reduce wave reflections at boundary (from Cerjan).
 
 @warning The number of nodes owned by each calling process (@c nxt) MUST be larger than the number of sponge
            layers (@c ND). This code assumes that if a calling process owns at least one sponge layer then it
            owns nodes up to the boundary. For example, if @c ND=10 and @nxt = 6, then we have the following situation at boundaries along the x domain:
                    proc 1         proc 2
                [- - - - - - | - - - -   - -| - - ...
                {   sponge layers      }
            This will cause an error in the code since it assumes that process 1 owns at least 10 nodes.
 
 Input parameter @c ND specifies the number of sponge layers where damping is applied (default is 20 layers). Every node
 that is within @c ND nodes of the domain boundary (excluding the surface, which is set as a free layer)
 has a multiplicative damping coefficient applied to the velocity and stress updates.
 
 
    @param ARBC             Coefficient for sponge layers (0.90-0.96)
    @param coords           @c int array (length 2) that stores the x and y coordinates of calling process in the Cartesian MPI topology
    @param nxt              Number of x nodes that calling process owns
    @param nyt              Number of y nodes that calling process owns
    @param nzt              Number of z nodes that calling process owns
    @param NX               Total number of nodes in x dimension
    @param NY               Total number of nodes in y dimension
    @param ND               Number of sponge layer nodes
    @param[out] dcrjx       Damping coefficients in x dimension (only set if calling process owns a block on the x boundary of the domain)
    @param[out] dxrjy       Damping coefficients in y dimension (only set if calling process owns a block on the y boundary of the domain)
    @param[out] dcrjz       Damping coefficients in z dimension (only set if calling process owns a block on the z boundary of the domain)
 
 */
void inicrj(float ARBC, int *coords, int nxt, int nyt, int nzt, int NX, int NY, int ND, Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz)
{
  int nxp, nyp, nzp;
  int i,   j,   k;
  float alpha;
  alpha = sqrt(-log(ARBC))/ND;

  nxp   = nxt*coords[0] + 1;
  if(nxp <= ND)
  {
     for(i=0;i<ND;i++)
     {
        nxp        = i + 1;
        dcrjx[i+2+4*LOOP] = dcrjx[i+2+4*LOOP]*(exp(-((alpha*(ND-nxp+1))*(alpha*(ND-nxp+1)))));
     } 
  }
  nxp   = nxt*coords[0] + 1;
  if( (nxp+nxt-1) >= (NX-ND+1))
  {
     for(i=nxt-ND;i<nxt;i++)
     {
        nxp        = i + NX - nxt + 1;
        dcrjx[i+2+4*LOOP] = dcrjx[i+2+4*LOOP]*(exp(-((alpha*(ND-(NX-nxp)))*(alpha*(ND-(NX-nxp))))));
     }
  }

  nyp   = nyt*coords[1] + 1;
  if(nyp <= ND)
  {
     for(j=0;j<ND;j++)
     {
        nyp        = j + 1;
        dcrjy[j+2+4*LOOP] = dcrjy[j+2+4*LOOP]*(exp(-((alpha*(ND-nyp+1))*(alpha*(ND-nyp+1)))));
     }
  }
  nyp   = nyt*coords[1] + 1;
  if((nyp+nyt-1) >= (NY-ND+1))
  {
     for(j=nyt-ND;j<nyt;j++)
     {
        nyp        = j + NY - nyt + 1;
        dcrjy[j+2+4*LOOP] = dcrjy[j+2+4*LOOP]*(exp(-((alpha*(ND-(NY-nyp)))*((alpha*(ND-(NY-nyp)))))));
     }
  }

  nzp = 1;
  if(nzp <= ND)
  {
     for(k=0;k<ND;k++)
     {
        nzp            = k + 1;
        dcrjz[k+ALIGN] = dcrjz[k+ALIGN]*(exp(-((alpha*(ND-nzp+1))*((alpha*(ND-nzp+1))))));
     }
  }
  return;
}  
