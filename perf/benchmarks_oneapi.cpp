#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include <benchmark/benchmark.h>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

//----------------------------------------------------------------------------//
// generate_table() algorithm

template <class T>
void generate_table_oneapi_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };
    sycl::buffer<T,1> br(r), b{sycl::range(nr * nc)};

    for (auto _ : st)
        one4all::oneapi::generate_table<pcg32>
        (   oneapi::dpl::begin(br)
        ,   oneapi::dpl::begin(b)
        ,   nr
        ,   nc
        ,   seed_pi
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(generate_table_oneapi_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(generate_table_oneapi_x8, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// scale_table() algorithm

template <class T>
void scale_table_oneapi_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };
    sycl::buffer<T> b(nr * nc), bs{nr * nc}, dr(r);

    one4all::oneapi::generate_table<pcg32>
    (   dpl::begin(dr)
    ,   dpl::begin(b)
    ,   nr
    ,   nc
    ,   seed_pi
    );

    for (auto _ : st)
    {   one4all::oneapi::scale_table
        (   dpl::begin(dr, sycl::read_only)
        ,   dpl::begin(b, sycl::read_only)
        ,   dpl::begin(bs, sycl::write_only, sycl::no_init)
        ,   nr
        ,   nc
        ,   T(-1.0), T(1.0)
        );
    }

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T) * 2) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(scale_table_oneapi_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(scale_table_oneapi_x8, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// main()

int main(int argc, char** argv)
{   benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    // adding oneAPI context
    sycl::queue q;
    std::stringstream os;
    os << "\n  Running on: "
       << q.get_device().get_info<sycl::info::device::name>()
       << "\n  Clock frequency: "
       << q.get_device().get_info<sycl::info::device::max_clock_frequency>()
       << "\n  Compute units: "
       << q.get_device().get_info<sycl::info::device::max_compute_units>();
    benchmark::AddCustomContext("oneAPI", os.str());

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}