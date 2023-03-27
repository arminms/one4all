#ifndef _ONE4ALL_DEVICE_HPP_
#define _ONE4ALL_DEVICE_HPP_

#if defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__)
#   define ONE4ALL_DEVICE_CODE __device__ __host__
// #elif defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)
// #   define ONE4ALL_DEVICE_CODE extern SYCL_EXTERNAL
#else
#   define ONE4ALL_DEVICE_CODE
#endif

#endif  //_ONE4ALL_DEVICE_HPP_