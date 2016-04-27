/**
@section LICENSE
Copyright (c) 2013-2016, Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
********************************************************************************
* Grid3D.c                                                                     *
* programming in C language                                                    *
* 3D data structure                                                            *
********************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pmcl3d.h"

Grid3D Alloc3D(int nx, int ny, int nz)
{
   int i, j, k;
   Grid3D U = (Grid3D)malloc(sizeof(float**)*nx + sizeof(float *)*nx*ny +sizeof(float)*nx*ny*nz);

   if (!U){
       printf("Cannot allocate 3D float array\n");
       exit(-1);
   }
   for(i=0;i<nx;i++){
       U[i] = ((float**) U) + nx + i*ny;
    }

   float *Ustart = (float *) (U[nx-1] + ny);
   for(i=0;i<nx;i++)
       for(j=0;j<ny;j++)
           U[i][j] = Ustart + i*ny*nz + j*nz;

   for(i=0;i<nx;i++)
       for(j=0;j<ny;j++)
           for(k=0;k<nz;k++)
              U[i][j][k] = 0.0f;

   return U;
}


Grid1D Alloc1D(int nx)
{
   int i;
   Grid1D U = (Grid1D)malloc(sizeof(float)*nx);

   if (!U){
       printf("Cannot allocate 2D float array\n");
       exit(-1);
   }

   for(i=0;i<nx;i++)
       U[i] = 0.0f;

   return U;
}


PosInf Alloc1P(int nx)
{
   int i;
   PosInf U = (PosInf)malloc(sizeof(int)*nx);

   if (!U){
       printf("Cannot allocate 2D integer array\n");
       exit(-1);
   }

   for(i=0;i<nx;i++)
       U[i] = 0;

   return U;
}

void Delloc3D(Grid3D U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}

void Delloc1D(Grid1D U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}

void Delloc1P(PosInf U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}
