[![Build and Test](https://github.com/arminms/one4all/actions/workflows/cmake.yml/badge.svg)](https://github.com/arminms/one4all/actions/workflows/cmake.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# one4all
A framework to streamline developing for CUDA, ROCm and oneAPI at the same time.
There is a recorded video about it on [SHARCNET YouTube Channel](https://sharcnet.ca):

- [CUDA, ROCm, oneAPI – All for One or One for All?](https://youtu.be/RYtdiOhrv0Q)

## Features
- Support four target APIs
    - CUDA
    - oneAPI
    - ROCm
    - STL Parallel Algorithms
- All the configurations are automatically done by [CMake](https://cmake.org/)
- Support unit testing with [Catch2](https://github.com/catchorg/Catch2)
- Support [Google Benchmark](https://github.com/google/benchmark)
- Two (kernel and Thrust/oneDPL) sample algorithms are already included

## Building from source
You need:
- C++ compiler supporting the C++17 standard (e.g. `gcc` 9.3)
- [CMake](https://cmake.org/) version 3.21 or higher.

And the following optional third-party libraries
* [Catch2](https://github.com/catchorg/Catch2) v3.1 or higher for unit testing
* [Google Benchmark](https://github.com/google/benchmark) for benchmarks

The `CMake` script configured in a way that if it cannot find the optional third-party libraries it tries to fetch and build them automatically. So, there is no need to do anything if they are missing but you need an internet connection for that to work.

On [the Alliance](https://alliancecan.ca/) clusters, you can activate the above environment by the following module command:
```
module load cmake googlebenchmark catch2
```

### Building C++17 parallel algorithm version
Parallel STL requires a [TBB](https://github.com/oneapi-src/oneTBB) version between 2018 to 2020 to work.
```shell
git clone https://github.com/arminms/one4all.git
cd one4all
cmake -S . -B build
cmake --build build -j
```
### Building CUDA version
Requires CUDA version 11 or higher.
```shell
git clone https://github.com/arminms/one4all.git
cd one4all
cmake -S . -B build-cuda -DONE4ALL_TARGET_API=cuda
cmake --build build-cuda -j
```
### Building ROCm version
Requires ROCm 5.4.3 or higher.
```shell
git clone https://github.com/arminms/one4all.git
cd one4all
CXX=hipcc cmake -S . -B build-rocm -DONE4ALL_TARGET_API=rocm
cmake --build build-rocm -j
```
### Building oneAPI version
Requires oneAPI 2023.0.0 or higher.
#### Building for OpenCL targets
```shell
git clone https://github.com/arminms/one4all.git
cd one4all
CXX=icpx cmake -S . -B build-oneapi -DONE4ALL_TARGET_API=oneapi
cmake --build build-oneapi -j
```
#### Building for OpenCL, NVIDIA and/or AMD GPUs
Requires [Codeplay](https://codeplay.com) plugins for [NVIDIA](https://developer.codeplay.com/products/oneapi/nvidia) and/or [AMD](https://developer.codeplay.com/products/oneapi/amd) GPUs installed.
```shell
git clone https://github.com/arminms/one4all.git
cd one4all
CXX=clang++ cmake -S . -B build-oneapi -DONE4ALL_TARGET_API=oneapi
cmake --build build-oneapi -j
```
## Running unit tests
```shell
cd build # or build-cuda / build-rocm / build-oneapi
ctest
```
To select target for oneAPI version, set `ONEAPI_DEVICE_SELECTOR` or `SYCL_DEVICE_FILTER` environment variable first:
```shell
# oneAPI 2023.1.0 or higher
ONEAPI_DEVICE_SELECTOR=[level_zero|opencl|cuda|hip|esimd_emulator|*][:cpu|gpu|fpga|*]

# older versions of oneAPI
SYCL_DEVICE_FILTER=[level_zero|opencl|cuda|hip|esimd_emulator|*][:cpu|gpu|acc|*]
```
You can find the complete syntax [here](https://intel.github.io/llvm-docs/EnvironmentVariables.html#oneapi-device-selector). Here is an example to run oneAPI version on NVIDIA GPUs:
```
ONEAPI_DEVICE_SELECTOR=cuda build-oneapi/test/unit_tests
```
## Running benchmarks
```shell
cd build  # or build-cuda / build-rocm / build-oneapi
perf/benchmarks --benchmark_counters_tabular=true
```
Selecting targets for oneAPI version is like unit tests described above.
### Benchmark results
Here are some updated benchmark results (more accurate than the preliminary results shown in the [YouTube video](https://youtu.be/RYtdiOhrv0Q) because of switching to
[cudaEvent*()](https://github.com/arminms/one4all/commit/a16c48eed536da55c91dada8aff3841e54dba55a) /
[hipEvent*()](https://github.com/arminms/one4all/commit/e02b76330f903ebe3225fac58e066734c5eeb203) /
[SYCL's queue profiling](https://github.com/arminms/one4all/pull/4/commits/586bb98621806cc842816ba2bdc3399c925ce363#diff-c6cbd687df542e27db5b7eb7615b11745b46a2dbe5f16667dd1a5f9855403c3f) for measuring performance) on [the Alliance](https://alliancecan.ca)'s clusters.
#### Parallel STL vs. oneAPI (higher is better)
Using `AMD EPYC 7543 x2 2.8 GHz (64C / 128T)`:
|API – Algorithm |`float`|`double`|
|:---|:---:|:---:|
|Parallel STL – `*`|1.00|1.00|
|oneAPI – `generate_table()`|0.46|0.27|
|oneAPI – `scale_table()`|1.61|1.36|
#### CUDA vs. oneAPI (higher is better)
Using `NVIDIA A100-SXM4-40GB`:
|API – Algorithm |`float`|`double`|
|:---|:---:|:---:|
|CUDA – `*`|1.00|1.00|
|oneAPI – `generate_table()`|1.00|1.02|
|oneAPI – `scale_table()`|1.01|1.02|
#### ROCm vs. oneAPI (higher is better)
Using `AMD Instinct MI210`:
|API – Algorithm |`float`|`double`|
|:---|:---:|:---:|
|ROCm – `*`|1.00|1.00|
|oneAPI – `generate_table()`|1.04|1.08|
|oneAPI – `scale_table()`|0.91|0.79|

## Using *one4all* for new projects
Select `fork` from the top right part of this page. You may choose a different name for your repository. In that case, you can also find/replace `one4all` with `<your-project>` in all files (case-sensitive) and `ONE4ALL_TARGET_API` with `<YOUR-PROJECT>_TARGET_API` in all `CMakeLists.txt` files. Finally, rename `include/one4all` folder to `include/<your-project>`.

You can add your new *algorithms* to `include/<your-project>/algorithm` 
along with *unit tests* and *benchmarks* in the corresponding `test/unit_test/unit_tests_*.cpp` and `perf/benchmark/benchmarks_*.cpp` files, respectively.

Later, if you decided to have a program, you can make a `src` folder and add the source code (e.g. `my_prog_*.cpp`) along with the following `CMakeLists.txt` into it:

```cmake
## defining target for my_prog
#
add_executable(my_prog
  my_prog_${<YOUR-PROJECT>_TARGET_API}.$<IF:$<STREQUAL:${<YOUR-PROJECT>_TARGET_API},cuda>,cu,cpp>
)

## defining link libraries for my_prog
#
target_link_libraries(my_prog PRIVATE
  ${PROJECT_NAME}::${<YOUR-PROJECT>_TARGET_API}
)

## installing my_prog
#
install(TARGETS my_prog RUNTIME DESTINATION CMAKE_INSTALL_BINDIR)
```

Don't forget to replace `<YOUR-PROJECT>` with the name of your project in the above file.

Finally, add `add_subdirectory(src)` at the end of the main [`CMakeLists.txt`](CMakeLists.txt) file.


<!-- ## Install oneAPI
```
ln -s ${PWD}/opt ~/opt
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19079/l_BaseKit_p_2023.0.0.25537_offline.sh
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19084/l_HPCKit_p_2023.0.0.25400_offline.sh
module load intel-opencl
sh ./l_BaseKit_p_2023.0.0.25537_offline.sh -a -c
sh ./l_HPCKit_p_2023.0.0.25400_offline.sh -a -c
sh oneapi-for-nvidia-gpus-2023.0.0-linux.sh --install-dir ~/opt/intel/oneapi
sh oneapi-for-amd-gpus-2023.0.0-linux.sh --install-dir ~/opt/intel/oneapi
patchelf --set-rpath /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/gcccore/9.3.0/lib64:/cvmfs/soft.computecanada.ca/gentoo/2020/lib64 /project/6004016/asobhani/opt/intel/oneapi/compiler/2023.0.0/linux/bin/sycl-ls
```

## Build on gra1339
```
CXX=/opt/rocm-5.4.0/bin/hipcc cmake -S . -B build -DTBB_INCLUDE_DIR=~/.local/include -DTBB_LIBRARY=~/.local/lib
```

```
. /opt/intel/oneapi/setvars.sh --include-intel-llvm
. ~/intel/oneapi/setvars.sh --include-intel-llvm
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

```
export PATH=/opt/rocm-4.5.2/bin:/home/centos/.local/bin:/opt/rh/devtoolset-11/root/usr/bin:/home/centos/.local/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/centos/.local/bin:/home/centos/bin
export LD_LIBRARY_PATH=~/.local/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=~/.local/lib64:/usr/lib:/usr/lib64:$LD_LIBRARY_PATH:~/.local/lib
SYCL_DEVICE_FILTER=level_zero|opencl|cuda|hip|esimd_emulator|*:cpu|gpu|acc|* ./unit_tests_oneapi

apptainer --rocm shell -C -B ~/armin/one4all ../cuda_rocm_oneapi_codeplay.sif
CXX=clang++ cmake -S . -B build-oneapi -DONE4ALL_TARGET_API=oneapi
cmake --build build-oneapi -j
SYCL_DEVICE_FILTER=hip:gpu build-oneapi/test/unit_tests
SYCL_DEVICE_FILTER=hip:gpu build-oneapi/perf/benchmarks --benchmark_counters_tabular=true
``` -->

