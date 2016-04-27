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
#include "pmcl3d.h"

int read_src_ifault_2(int rank, int READ_STEP,
    char *INSRC, char *INSRC_I2,
    int maxdim, int *coords, int NZ,
    int nxt, int nyt, int nzt,
    int *NPSRC, int *SRCPROC,
    PosInf *psrc, Grid1D *axx, Grid1D *ayy, Grid1D *azz,
    Grid1D *axz, Grid1D *ayz, Grid1D *axy,
    int idx){

  FILE *f;
  char fname[150];
  int dummy[2], i, j;
  int nbx, nby;
  PosInf tpsrc = NULL;
  Grid1D taxx=NULL, tayy=NULL, tazz=NULL;
  Grid1D taxy=NULL, taxz=NULL, tayz=NULL;

  // First time entering this function
  if(idx == 1){
    sprintf(fname, "%s%07d", INSRC, rank);
    printf("SOURCE reading first time: %s\n",fname);
    f = fopen(fname, "rb");
    if(f == NULL){
      printf("SOURCE %d) no such file: %s\n",rank,fname);
      *SRCPROC = -1;
      *NPSRC = 0;
      return 0;
    }
    *SRCPROC = rank;
    nbx     = nxt*coords[0] + 1 - 2*loop;
    nby     = nyt*coords[1] + 1 - 2*loop;
    // not sure what happens if maxdim != 3
    fread(NPSRC, sizeof(int), 1, f);
    fread(dummy, sizeof(int), 2, f);

    printf("SOURCE I am, rank=%d npsrc=%d\n",rank,*NPSRC);

    tpsrc = Alloc1P((*NPSRC)*maxdim);
    fread(tpsrc, sizeof(int), (*NPSRC)*maxdim, f);
    // assuming nzt=NZ
    for(i=0; i<*NPSRC; i++){
      //tpsrc[i*maxdim] = (tpsrc[i*maxdim]-1)%nxt+1;
      //tpsrc[i*maxdim+1] = (tpsrc[i*maxdim+1]-1)%nyt+1;
      //tpsrc[i*maxdim+2] = NZ+1 - tpsrc[i*maxdim+2];
      tpsrc[i*maxdim]   = tpsrc[i*maxdim] - nbx - 1;
      tpsrc[i*maxdim+1] = tpsrc[i*maxdim+1] - nby - 1;
      tpsrc[i*maxdim+2] = NZ + 1 - tpsrc[i*maxdim+2];
    }
    *psrc = tpsrc;
    fclose(f);
  }
  if(*NPSRC > 0){
    sprintf(fname, "%s%07d_%03d", INSRC_I2, rank, idx);
    printf("SOURCE reading: %s\n",fname);
    f = fopen(fname, "rb");
    if(f == NULL){
      printf("ERROR! Rank=%d: Cannot open file: %s\n", rank, fname);
      return -1;
    }
    // fastest axis in partitioned source is npsrc, then read_step (see source.f)
    // fastest axis in axx is time
    Grid1D tmpta;
    tmpta = Alloc1D((*NPSRC)*READ_STEP);
    if(idx==1){
      taxx = Alloc1D((*NPSRC)*READ_STEP);
      tayy = Alloc1D((*NPSRC)*READ_STEP);
      tazz = Alloc1D((*NPSRC)*READ_STEP);
      taxy = Alloc1D((*NPSRC)*READ_STEP);
      taxz = Alloc1D((*NPSRC)*READ_STEP);
      tayz = Alloc1D((*NPSRC)*READ_STEP);
    }
    else{
      taxx = *axx;
      tayy = *ayy;
      tazz = *azz;
      taxy = *axy;
      taxz = *axz;
      tayz = *ayz;
    }
    fread(tmpta, sizeof(float), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        taxx[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(float), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        tayy[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(float), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        tazz[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(float), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        taxz[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(float), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        tayz[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(float), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        taxy[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fclose(f);
    *axx = taxx;
    *ayy = tayy;
    *azz = tazz;
    *axz = taxz;
    *ayz = tayz;
    *axy = taxy;
    Delloc1D(tmpta);
  }

return 0;
}

int inisource(int      rank,    int     IFAULT, int     NSRC,   int     READ_STEP, int     NST,     int     *SRCPROC, int    NZ,
              MPI_Comm MCW,     int     nxt,    int     nyt,    int     nzt,       int     *coords, int     maxdim,   int    *NPSRC,
              PosInf   *ptpsrc, Grid1D  *ptaxx, Grid1D  *ptayy, Grid1D  *ptazz,    Grid1D  *ptaxz,  Grid1D  *ptayz,   Grid1D *ptaxy, char *INSRC, char *INSRC_I2)
{
   int i, j, k, npsrc, srcproc, master=0;
   int nbx, nex, nby, ney, nbz, nez;
   PosInf tpsrc=NULL, tpsrcp =NULL;
   Grid1D taxx =NULL, tayy   =NULL, tazz =NULL, taxz =NULL, tayz =NULL, taxy =NULL;
   Grid1D taxxp=NULL, tayyp  =NULL, tazzp=NULL, taxzp=NULL, tayzp=NULL, taxyp=NULL;
   if(NSRC<1) return 0;

   npsrc   = 0;
   srcproc = -1;
// Indexing is based on 1: [1, nxt], etc. Include 1st layer ghost cells
   nbx     = nxt*coords[0] + 1 - 2*loop;
   nex     = nbx + nxt + 4*loop - 1;
   nby     = nyt*coords[1] + 1 - 2*loop;
   ney     = nby + nyt + 4*loop - 1;
   nbz     = 1;
   nez     = nzt;
   // IFAULT=1 has bug! READ_STEP does not work, it tries to read NST all at once - Efe
   if(IFAULT<=1)
   {
      tpsrc = Alloc1P(NSRC*maxdim);
      taxx  = Alloc1D(NSRC*READ_STEP);
      tayy  = Alloc1D(NSRC*READ_STEP);
      tazz  = Alloc1D(NSRC*READ_STEP);
      taxz  = Alloc1D(NSRC*READ_STEP);
      tayz  = Alloc1D(NSRC*READ_STEP);
      taxy  = Alloc1D(NSRC*READ_STEP);

      if(rank==master)
      {
      	 FILE   *file;
         int    tmpsrc[3];
         Grid1D tmpta;
         if(IFAULT == 1){
          file = fopen(INSRC,"rb");
          tmpta = Alloc1D(NST*6);
         }
         else if(IFAULT == 0) file = fopen(INSRC,"r");
         if(!file)
         {
            printf("can't open file %s", INSRC);
	    return 0;
         }
         if(IFAULT == 1){
          for(i=0;i<NSRC;i++)
          {
            if(fread(tmpsrc,sizeof(int),3,file) && fread(tmpta,sizeof(float),NST*6,file))
            {
               tpsrc[i*maxdim]   = tmpsrc[0];
               tpsrc[i*maxdim+1] = tmpsrc[1];
               tpsrc[i*maxdim+2] = NZ+1-tmpsrc[2];
               for(j=0;j<READ_STEP;j++)
               {
                  taxx[i*READ_STEP+j] = tmpta[j*6];
                  tayy[i*READ_STEP+j] = tmpta[j*6+1];
                  tazz[i*READ_STEP+j] = tmpta[j*6+2];
                  taxz[i*READ_STEP+j] = tmpta[j*6+3];
                  tayz[i*READ_STEP+j] = tmpta[j*6+4];
                  taxy[i*READ_STEP+j] = tmpta[j*6+5];
               }
            }
          }
          Delloc1D(tmpta);
         }
         else if(IFAULT == 0)
          for(i=0;i<NSRC;i++)
          {
            fscanf(file, " %d %d %d ",&tmpsrc[0], &tmpsrc[1], &tmpsrc[2]);
            tpsrc[i*maxdim]   = tmpsrc[0];
            tpsrc[i*maxdim+1] = tmpsrc[1];
            tpsrc[i*maxdim+2] = NZ+1-tmpsrc[2];
            //printf("SOURCE: %d,%d,%d\n",tpsrc[0],tpsrc[1],tpsrc[2]);
            for(j=0;j<READ_STEP;j++){
              fscanf(file, " %f %f %f %f %f %f ",
                &taxx[i*READ_STEP+j], &tayy[i*READ_STEP+j],
                &tazz[i*READ_STEP+j], &taxz[i*READ_STEP+j],
                &tayz[i*READ_STEP+j], &taxy[i*READ_STEP+j]);
              //printf("SOURCE VAL %d: %f,%f\n",j,taxx[j],tayy[j]);
            }
          }
         fclose(file);
      }
      MPI_Bcast(tpsrc, NSRC*maxdim,    MPI_INT,  master, MCW);
      MPI_Bcast(taxx,  NSRC*READ_STEP, MPI_REAL, master, MCW);
      MPI_Bcast(tayy,  NSRC*READ_STEP, MPI_REAL, master, MCW);
      MPI_Bcast(tazz,  NSRC*READ_STEP, MPI_REAL, master, MCW);
      MPI_Bcast(taxz,  NSRC*READ_STEP, MPI_REAL, master, MCW);
      MPI_Bcast(tayz,  NSRC*READ_STEP, MPI_REAL, master, MCW);
      MPI_Bcast(taxy,  NSRC*READ_STEP, MPI_REAL, master, MCW);
      for(i=0;i<NSRC;i++)
      {
          if( tpsrc[i*maxdim]   >= nbx && tpsrc[i*maxdim]   <= nex && tpsrc[i*maxdim+1] >= nby
           && tpsrc[i*maxdim+1] <= ney && tpsrc[i*maxdim+2] >= nbz && tpsrc[i*maxdim+2] <= nez)
          {
              srcproc = rank;
              npsrc ++;
          }
      }
      if(npsrc > 0)
      {
          tpsrcp = Alloc1P(npsrc*maxdim);
          taxxp  = Alloc1D(npsrc*READ_STEP);
          tayyp  = Alloc1D(npsrc*READ_STEP);
          tazzp  = Alloc1D(npsrc*READ_STEP);
          taxzp  = Alloc1D(npsrc*READ_STEP);
          tayzp  = Alloc1D(npsrc*READ_STEP);
          taxyp  = Alloc1D(npsrc*READ_STEP);
          k      = 0;
          for(i=0;i<NSRC;i++)
          {
              if( tpsrc[i*maxdim]   >= nbx && tpsrc[i*maxdim]   <= nex && tpsrc[i*maxdim+1] >= nby
               && tpsrc[i*maxdim+1] <= ney && tpsrc[i*maxdim+2] >= nbz && tpsrc[i*maxdim+2] <= nez)
               {
                 tpsrcp[k*maxdim]   = tpsrc[i*maxdim]   - nbx - 1;
                 tpsrcp[k*maxdim+1] = tpsrc[i*maxdim+1] - nby - 1;
                 tpsrcp[k*maxdim+2] = tpsrc[i*maxdim+2] - nbz + 1;
                 for(j=0;j<READ_STEP;j++)
                 {
                    taxxp[k*READ_STEP+j] = taxx[i*READ_STEP+j];
                    tayyp[k*READ_STEP+j] = tayy[i*READ_STEP+j];
                    tazzp[k*READ_STEP+j] = tazz[i*READ_STEP+j];
                    taxzp[k*READ_STEP+j] = taxz[i*READ_STEP+j];
                    tayzp[k*READ_STEP+j] = tayz[i*READ_STEP+j];
                    taxyp[k*READ_STEP+j] = taxy[i*READ_STEP+j];
                  }
                  k++;
               }
          }
      }
      Delloc1D(taxx);
      Delloc1D(tayy);
      Delloc1D(tazz);
      Delloc1D(taxz);
      Delloc1D(tayz);
      Delloc1D(taxy);
      Delloc1P(tpsrc);

      *SRCPROC = srcproc;
      *NPSRC   = npsrc;
      *ptpsrc  = tpsrcp;
      *ptaxx   = taxxp;
      *ptayy   = tayyp;
      *ptazz   = tazzp;
      *ptaxz   = taxzp;
      *ptayz   = tayzp;
      *ptaxy   = taxyp;
   }
   else if(IFAULT == 2){
      return read_src_ifault_2(rank, READ_STEP,
        INSRC, INSRC_I2,
        maxdim, coords, NZ,
        nxt, nyt, nzt,
        NPSRC, SRCPROC,
        ptpsrc, ptaxx, ptayy, ptazz,
        ptaxz, ptayz, ptaxy, 1);
   }
   return 0;
}

void addsrc(int i,      float DH,   float DT,   int NST,    int npsrc,  int READ_STEP, int dim, PosInf psrc,
            Grid1D axx, Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
            Grid3D xx,  Grid3D yy,  Grid3D zz,  Grid3D xy,  Grid3D yz,  Grid3D xz)
{
  float vtst;
  int idx, idy, idz, j;
  vtst = (float)DT/(DH*DH*DH);

  i   = i - 1;
  for(j=0;j<npsrc;j++)
  {
     idx = psrc[j*dim]   + 1 + 4*loop;
     idy = psrc[j*dim+1] + 1 + 4*loop;
     idz = psrc[j*dim+2] + align - 1;
     xx[idx][idy][idz] = xx[idx][idy][idz] - vtst*axx[j*READ_STEP+i];
     yy[idx][idy][idz] = yy[idx][idy][idz] - vtst*ayy[j*READ_STEP+i];
     zz[idx][idy][idz] = zz[idx][idy][idz] - vtst*azz[j*READ_STEP+i];
     xz[idx][idy][idz] = xz[idx][idy][idz] - vtst*axz[j*READ_STEP+i];
     yz[idx][idy][idz] = yz[idx][idy][idz] - vtst*ayz[j*READ_STEP+i];
     xy[idx][idy][idz] = xy[idx][idy][idz] - vtst*axy[j*READ_STEP+i];
/*
     printf("xx=%1.6g\n",xx[idx][idy][idz]);
     printf("yy=%1.6g\n",yy[idx][idy][idz]);
     printf("zz=%1.6g\n",zz[idx][idy][idz]);
     printf("xz=%1.6g\n",xz[idx][idy][idz]);
     printf("yz=%1.6g\n",yz[idx][idy][idz]);
     printf("xy=%1.6g\n",xy[idx][idy][idz]);
*/
  }
  return;
}
