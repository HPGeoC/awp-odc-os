/**
 @author Alexander Breuer (anbreuer AT ucsd.edu)
 
 @section DESCRIPTION
 Definition of compile time constants.
 
 @section LICENSE
 Copyright (c) 2015-2016, Regents of the University of California
 All rights reserved.
 
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP


// constants of shared code - start
#ifdef __RESTRICT
#define RESTRICT restrict
#else
#define RESTRICT
#endif

#ifdef USE_CUDA
#define LOOP 1
#define ALIGN 32
#endif

typedef float *RESTRICT *RESTRICT *RESTRICT Grid3D;
typedef int   *RESTRICT *RESTRICT *RESTRICT Grid3Dww;
typedef float *RESTRICT                     Grid1D;
typedef int   *RESTRICT                     PosInf;
// constants of shared code - end

// floating point precision
typedef float real;
#define AWP_MPI_REAL MPI_FLOAT

// int for grid points
typedef long long int_pt;
#define AWP_PT_FORMAT_STRING "lld"

// constants used in patch decomposition
#define PATCH_X 2048
#define PATCH_Y 2048
#define PATCH_Z 2048
#define BDRY_SIZE 8


#ifdef __cplusplus
namespace odc {
    namespace constants {
        static const int_pt boundary = 2;
    }
}
#endif


#endif
