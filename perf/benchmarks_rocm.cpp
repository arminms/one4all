#include <vector>
#include <sstream>
#include <iomanip>

#include <benchmark/benchmark.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

//----------------------------------------------------------------------------//
// generate_table() algortithm

template <class T>
void generate_table_rocm_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };
    thrust::device_vector<T> b(nr * nc), dr(r);

    for (auto _ : st)
    {   hipEventRecord(start);
        one4all::rocm::generate_table<pcg32>
        (   dr.begin()
        ,   b.begin()
        ,   nr
        ,   nc
        ,   seed_pi
        );
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    hipEventDestroy(start); hipEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(generate_table_rocm_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(generate_table_rocm_x8, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// scale_table() algorithm

template <class T>
void scale_table_rocm_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    hipEvent_t start, stop;
    hipEventCreate(&start); hipEventCreate(&stop);
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };
    thrust::device_vector<T> b(nr * nc), bs(nr * nc), dr(r);

    one4all::rocm::generate_table<pcg32>
    (   dr.begin()
    ,   b.begin()
    ,   nr
    ,   nc
    ,   seed_pi
    );

    for (auto _ : st)
    {   hipEventRecord(start);
        one4all::rocm::scale_table
        (   dr.begin()
        ,   b.begin()
        ,   bs.begin()
        ,   nr
        ,   nc
        ,   T(-1.0), T(1.0)
        );
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    hipEventDestroy(start); hipEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T) * 2) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(scale_table_rocm_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(scale_table_rocm_x8, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// main()

int main(int argc, char** argv)
{   benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    // adding GPU context
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    std::stringstream os;
    os << "\n  " << prop.name
       << "\n  L2 Cache: " << prop.l2CacheSize / 1024 << " KiB"
       << "\n  Number of SMs: x" << prop.multiProcessorCount
       << "\n  Peak Memory Bandwidth: "
       << std::fixed << std::setprecision(0)
       << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
       << " (GB/s)";
    benchmark::AddCustomContext("GPU", os.str());

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}