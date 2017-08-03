/** 
 @brief Defines functions to dynamically allocate and free memory for one and three dimensional grid data structures.
 
 @section LICENSE
 Copyright (c) 2013-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constants.hpp"

/**
 Allocates and returns pointer to three dimensional grid structure of @c float 's. All of the values
 are initialized to @c 0.0f. It is the responsibility of the caller to free the allocated memory when 
 done with a call to @c Delloc3D. Terminates with exit code -1 if unable to allocate the requested memory.
 
    @param nx   Size of x dimension
    @param ny   Size of y dimension
    @param nz   Size of z dimension
 
    @return Pointer to newly allocated @c Grid3D struct
 
    @warning Terminates with exit code -1 if unable to allocate requested memory.
 */
Grid3D Alloc3D(int_pt nx, int_pt ny, int_pt nz)
{
   int_pt i, j, k;
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


/**
 Allocates and returns pointer to three dimensional grid structure of @c int 's. All of the values are initialized to @c 0.
 It is the responsibility of the caller to free the allocated memory when done with a call to @c Delloc3Dww.
 Terminates with exit code -1 if unable to allocate the requested memory.
 
 @param nx  Size of x dimension
 @param ny  Size of y dimension
 @param nz  Size of z dimension
 
 @return Pointer to newly allocated @c Grid3Dww struct
 
 @warning Terminates with exit code -1 if unable to allocate requested memory.
 */
Grid3Dww Alloc3Dww(int_pt nx, int_pt ny, int_pt nz)
{
  int_pt i, j, k;
  Grid3Dww U = (Grid3Dww)malloc(sizeof(int**)*nx + sizeof(int *)*nx*ny +sizeof(int)*nx*ny*nz);

  if (!U){
    printf("Cannot allocate 3D int array\n");
    exit(-1);
  }
  for(i=0;i<nx;i++){
    U[i] = ((int**) U) + nx + i*ny;
  }

  int *Ustart = (int *) (U[nx-1] + ny);
  for(i=0;i<nx;i++)
    for(j=0;j<ny;j++)
      U[i][j] = Ustart + i*ny*nz + j*nz;

  for(i=0;i<nx;i++)
    for(j=0;j<ny;j++)
      for(k=0;k<nz;k++)
	U[i][j][k] = 0;

  return U;
}


/**
 Allocates and returns pointer to one dimensional grid structure of @c float 's. All of the values are initialized to 0.0f.
 It is the responsibility of the caller to free the allocated memory when done with a call to @c Delloc1D.
 Terminates with exit code -1 if unable to allocate the requested memory.
 
 @param nx  Number of @c int's in the grid
 
 @return Pointer to newly allocated @c Grid1D struct
 
 @warning Terminates with exit code -1 if unable to allocate requested memory.
 */
Grid1D Alloc1D(int_pt nx)
{
   int_pt i;
   Grid1D U = (Grid1D)malloc(sizeof(float)*nx);

   if (!U){
       printf("Cannot allocate 1D float array\n");
       exit(-1);
   }

   for(i=0;i<nx;i++)
       U[i] = 0.0f;

   return U;
}

/**
 Allocates and returns pointer to one dimensional grid structure of @c int's. All of the values are initialized to 0.
 It is the responsibility of the caller to free the allocated memory when done with a call to @c Delloc1P.
 Terminates with exit code -1 if unable to allocate the requested memory.
 
 @param nx  Number of @c int's in the grid
 
 @return Pointer to newly allocated @c Grid1P struct
 
 @warning Terminates with exit code -1 if unable to allocate requested memory.
 */
PosInf Alloc1P(int_pt nx)
{
   int_pt i;
   PosInf U = (PosInf)malloc(sizeof(int)*nx);

   if (!U){
       printf("Cannot allocate 1D integer array\n");
       exit(-1);
   }

   for(i=0;i<nx;i++)
       U[i] = 0;

   return U;
}

/**
 Deallocates memory that was allocated by a call to @c Alloc3D. The supplied pointer may be @c NULL.
 
 @param[in,out] U   Pointer allocated by a call to @c Alloc3D. @c U is set to @c NULL at the end of the function call.
 
 @warning If @c U is not either @c NULL or a pointer that was allocated from a call to @c Alloc3D then this will cause a memory leak.
 */
void Delloc3D(Grid3D U)
{
   if (U) 
   {
      free(U);
      U = NULL;
   }

   return;
}

/**
 Deallocates memory that was allocated by a call to @c Alloc3Dww. The supplied pointer may be @c NULL.
 
 @param[in,out] U   Pointer allocated by a call to @c Alloc3Dww. @c U is set to @c NULL at the end of the function call.
 
 @warning If @c U is not either @c NULL or a pointer that was allocated from a call to @c Alloc3Dww then this will cause a memory leak.
 */
void Delloc3Dww(Grid3Dww U)
{
  if (U)
    {
      free(U);
      U = NULL;
    }

  return;
}

/**
 Deallocates memory that was allocated by a call to @c Alloc1D. The supplied pointer may be @c NULL.
 
 @param[in,out] U   Pointer allocated by a call to @c Alloc1D. @c U is set to @c NULL at the end of the function call.
 
 @warning If @c U is not either @c NULL or a pointer that was allocated from a call to @c Alloc1D then this will cause a memory leak.
 */
void Delloc1D(Grid1D U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}

/**
 Deallocates memory that was allocated by a call to @c Alloc1P. The supplied pointer may be @c NULL.
 
 @param[in,out] U   Pointer allocated by a call to @c Alloc1P. @c U is set to @c NULL at the end of the function call.
 
 @warning If @c U is not either @c NULL or a pointer that was allocated from a call to @c Alloc1P then this will cause a memory leak.
 */
void Delloc1P(PosInf U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}
