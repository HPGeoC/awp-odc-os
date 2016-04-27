/**
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
        dcrjx[i+2+4*loop] = dcrjx[i+2+4*loop]*(exp(-((alpha*(ND-nxp+1))*(alpha*(ND-nxp+1)))));
     }
  }
  nxp   = nxt*coords[0] + 1;
  if( (nxp+nxt-1) >= (NX-ND+1))
  {
     for(i=nxt-ND;i<nxt;i++)
     {
        nxp        = i + NX - nxt + 1;
        dcrjx[i+2+4*loop] = dcrjx[i+2+4*loop]*(exp(-((alpha*(ND-(NX-nxp)))*(alpha*(ND-(NX-nxp))))));
     }
  }

  nyp   = nyt*coords[1] + 1;
  if(nyp <= ND)
  {
     for(j=0;j<ND;j++)
     {
        nyp        = j + 1;
        dcrjy[j+2+4*loop] = dcrjy[j+2+4*loop]*(exp(-((alpha*(ND-nyp+1))*(alpha*(ND-nyp+1)))));
     }
  }
  nyp   = nyt*coords[1] + 1;
  if((nyp+nyt-1) >= (NY-ND+1))
  {
     for(j=nyt-ND;j<nyt;j++)
     {
        nyp        = j + NY - nyt + 1;
        dcrjy[j+2+4*loop] = dcrjy[j+2+4*loop]*(exp(-((alpha*(ND-(NY-nyp)))*((alpha*(ND-(NY-nyp)))))));
     }
  }

  nzp = 1;
  if(nzp <= ND)
  {
     for(k=0;k<ND;k++)
     {
        nzp            = k + 1;
        dcrjz[k+align] = dcrjz[k+align]*(exp(-((alpha*(ND-nzp+1))*((alpha*(ND-nzp+1))))));
     }
  }
  return;
}
