/**
@section LICENSE
Copyright (c) 2013-2016, Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "pmcl3d.h"

int writeCHK(char *chkfile, int ntiskp, float dt, float dh,
      int nxt, int nyt, int nzt,
      int nt, float arbc, int npc, int nve,
      float fl, float fh, float fp,
      float *vse, float *vpe, float *dde){

  FILE *fchk;

  fchk = fopen(chkfile,"w");
  fprintf(fchk,"STABILITY CRITERIA .5 > CMAX*DT/DX:\t%f\n",vpe[1]*dt/dh);
  fprintf(fchk,"# OF X,Y,Z NODES PER PROC:\t%d, %d, %d\n",nxt,nyt,nzt);
  fprintf(fchk,"# OF TIME STEPS:\t%d\n",nt);
  fprintf(fchk,"DISCRETIZATION IN SPACE:\t%f\n",dh);
  fprintf(fchk,"DISCRETIZATION IN TIME:\t%f\n",dt);
  fprintf(fchk,"PML REFLECTION COEFFICIENT:\t%f\n",arbc);
  fprintf(fchk,"HIGHEST P-VELOCITY ENCOUNTERED:\t%f\n",vpe[1]);
  fprintf(fchk,"LOWEST P-VELOCITY ENCOUNTERED:\t%f\n",vpe[0]);
  fprintf(fchk,"HIGHEST S-VELOCITY ENCOUNTERED:\t%f\n",vse[1]);
  fprintf(fchk,"LOWEST S-VELOCITY ENCOUNTERED:\t%f\n",vse[0]);
  fprintf(fchk,"HIGHEST DENSITY ENCOUNTERED:\t%f\n",dde[1]);
  fprintf(fchk,"LOWEST  DENSITY ENCOUNTERED:\t%f\n",dde[0]);
  fprintf(fchk,"SKIP OF SEISMOGRAMS IN TIME (LOOP COUNTER):\t%d\n",ntiskp);
  fprintf(fchk,"ABC CONDITION, PML=1 OR CERJAN=0:\t%d\n",npc);
  fprintf(fchk,"FD SCHEME, VISCO=1 OR ELASTIC=0:\t%d\n",nve);
  fprintf(fchk,"Q, FL,FP,FH:\t%f, %f, %f\n",fl,fp,fh);
  fclose(fchk);

return 0;
}
