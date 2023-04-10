#ifdef __HIP
#include <hip/hip_runtime.h>
//#include <hip/hip_ext.h>
#define cudaBindTexture hipBindTexture
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaFuncCachePreferL1 hipFuncCachePreferL1
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyToSymbol hipMemcpyToSymbol
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemset hipMemset
#define cudaPrintfDisplay hipPrintfDisplay
#define cudaPrintfEnd hipPrintfEnd
#define cudaPrintfInit hipPrintfInit
#define cudaReadModeElementType hipReadModeElementType
#define cuda_runtime hip_runtime
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaThreadSynchronize hipDeviceSynchronize
#define cudaUnbindTexture hipUnbindTexture
// Perhaps newer version of AMD HIP has this implemented?
#define cudaFuncSetCacheConfig //Not available: hipFuncSetCacheConfig
#define cudaMemcpyAsync hipMemcpyAsync
// Register is the keyword available in CUDA but not HIP, so just remove it cause it throws a compiler error
#define register
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif
