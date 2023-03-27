cmake_minimum_required(VERSION 3.21...3.26)

include(cmake/BuildType.cmake)

## life is about choices...
#
option(ONE4ALL_ENABLE_TESTS "Enable the unit tests ?" OFF)
option(ONE4ALL_ENABLE_BENCHMARKS "Enable benchmarks ?" OFF)

## finally our project...
#
project(
  one4all
  VERSION 1.00
  DESCRIPTION "A framework to streamline developing for CUDA, ROCm and oneAPI at the same time"
  LANGUAGES CXX
)

## setting target API
#
set(ONE4ALL_TARGET_API "stl" CACHE
  STRING "Choose target API ?")
set_property(CACHE ONE4ALL_TARGET_API PROPERTY STRINGS
  "cuda"
  "oneapi"
  "rocm"
  "stl"
)

## necessary modules
#
include(FetchContent)

## prevent in-source builds
#
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

## switch to the target API
#
if(${ONE4ALL_TARGET_API} STREQUAL oneapi)
  if("${CMAKE_CXX_COMPILER}" MATCHES ".*clang\\+\\+")
    set(SYCL_TARGETS "nvptx64-nvidia-cuda,spir64")
  else()
    set(SYCL_TARGETS "spir64")
  endif()
  find_package(IntelDPCPP REQUIRED)
  find_package(oneDPL REQUIRED)
elseif(${ONE4ALL_TARGET_API} STREQUAL cuda)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 11 REQUIRED)
  endif()
elseif(${ONE4ALL_TARGET_API} STREQUAL rocm)
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
  find_package(hip REQUIRED)
  set(HIP_TARGET  "hip::device" CACHE
      STRING "Choose HIP target ?")
  set_property(CACHE HIP_TARGET PROPERTY STRINGS
    "hip::host"
    "hip::device"
  )
  find_package(rocThrust REQUIRED)
elseif(${ONE4ALL_TARGET_API} STREQUAL stl)
  find_package(OpenMP REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
  find_package(TBB REQUIRED) # required for C++17 parallel algorithms
else()
  message(FATAL_ERROR "Wrong ONE4ALL_TARGET_API: ${ONE4ALL_TARGET_API}")
endif()

if(${ONE4ALL_ENABLE_TESTS})
  enable_testing()
  find_package(
    Catch2 3
    HINTS $ENV{HOME} /usr /usr/local /opt/local /opt
  )
  if(NOT Catch2_DIR)
      message(STATUS "Fetching Catch2 library...")
      FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.1.0
      )
      FetchContent_MakeAvailable(Catch2)
      list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
  endif()
  include(Catch)
  add_subdirectory(test)
endif()

if(${ONE4ALL_ENABLE_BENCHMARKS})
  find_package(benchmark
    CONFIG
    HINTS $ENV{HOME} /usr/local /opt/local /opt
  )
  if(NOT benchmark_DIR)
    message(STATUS "Fetching Google Benchmark library...")
    FetchContent_Declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.7.1
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL
      "Enable testing of the benchmark library."
    )
    FetchContent_MakeAvailable(benchmark)
  endif()
  add_subdirectory(perf)
endif()