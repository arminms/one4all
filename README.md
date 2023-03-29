# one4all
<!-- A framework to streamline developing for CUDA, ROCm and oneAPI at the same time. -->
## Building from source
You need:
- C++ compiler supporting the C++17 standard (e.g. `gcc` 9.3)
- [CMake](https://cmake.org/) version 3.21 or higher.


### Building C++17 parallel algorithm version
```bash
git clone https://github.com/arminms/one4all.git
cd one4all
cmake -S . -B build
cmake --build build -j
```
### Building CUDA version
Requires CUDA version 11 or higher.
```bash
git clone https://github.com/arminms/one4all.git
cd one4all
cmake -S . -B build-cuda -DONE4ALL_TARGET_API=cuda
cmake --build build-cuda -j
```
### Building ROCm version
Only tested with ROCm 5.4.0.
```bash
git clone https://github.com/arminms/one4all.git
cd one4all
CXX=hipcc cmake -S . -B build-rocm -DONE4ALL_TARGET_API=rocm
cmake --build build-rocm -j
```
### Building oneAPI version
Only tested with oneAPI 2023.
```bash
git clone https://github.com/arminms/one4all.git
cd one4all
CXX=icpx cmake -S . -B build-oneapi -DONE4ALL_TARGET_API=oneapi
cmake --build build-oneapi -j
```
## Running unit tests
```bash
cd build # or build-cuda / build-rocm / build-oneapi
ctest
```
## Running benchmarks
```bash
cd build  # or build-cuda / build-rocm / build-oneapi
perf/benchmarks --benchmark_counters_tabular=true
```

<!-- wget -qO- "https://cmake.org/files/v3.26/cmake-3.26.0-rc4-linux-x86_64.tar.gz" | tar --strip-components=1 -xz -C ~/.local
```
## Install oneAPI
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

https://codeplay.com/portal/blogs/2022/12/16/bringing-nvidia-and-amd-support-to-oneapi.html

https://developer.codeplay.com/products/oneapi/nvidia/home/
https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia

https://developer.codeplay.com/products/oneapi/amd/home/
https://developer.codeplay.com/products/oneapi/amd/2023.0.0/guides/get-started-guide-amd.html

## Build on gra1339
```
CXX=/opt/rocm-5.4.0/bin/hipcc cmake -S . -B build -DTBB_INCLUDE_DIR=~/.local/include -DTBB_LIBRARY=~/.local/lib
```

```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```
 -->