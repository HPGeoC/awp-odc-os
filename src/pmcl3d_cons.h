/**
@section LICENSE
Copyright (c) 2013-2016, Regents of the University of California
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define BLOCK_SIZE_X 2
#define BLOCK_SIZE_Y 2
//#define BLOCK_SIZE_Z 128
#define BLOCK_SIZE_Z 256
#define awp_align 32
#define loop  1

#define Both  0
#define Left  1
#define Right 2
#define Front 3
#define Back  4

//HIGHEST ORDER OF FILTER is MAXFILT-1
#define MAXFILT 20


/*
 * Intercept CUDA errors. Usage: pass the CUDA
 * library function call to this macro.
 * For example, CUCHK(cudaMalloc(...));
 * This check will be disabled if the preprocessor macro NDEBUG is defined (same
 * macro that disables assert() )
 */
#ifndef NDEBUG
#define CUCHK(call) {                                                         \
  cudaError_t err = call;                                                     \
  if( cudaSuccess != err) {                                                   \
  fprintf(stderr, "CUDA error in %s:%i %s(): %s.\n",                          \
          __FILE__, __LINE__, __func__, cudaGetErrorString(err) );            \
  fflush(stderr);                                                             \
  exit(EXIT_FAILURE);                                                         \
  }                                                                           \
}
#else
#define CUCHK(call) {}
#endif

// intercept MPI errors. Same usage as for CUCHK
#ifndef NDEBUG
#define MPICHK(err) {                                                         \
 if (err != MPI_SUCCESS) {                                                    \
 char error_string[2048];                                                     \
 int length_of_error_string;                                                  \
 MPI_Error_string((err), error_string, &length_of_error_string);              \
 fprintf(stderr, "MPI error: %s:%i %s(): %s\n",                               \
         __FILE__, __LINE__, __func__, error_string);                         \
 MPI_Abort(MPI_COMM_WORLD, err);                                              \
 fflush(stderr);                                                              \
 exit(EXIT_FAILURE);                                                          \
}                                                                             \
}
#else
#define MPICHK(err) {}
#endif
