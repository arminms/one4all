#include <execution>
#include <vector>
#include <algorithm>

#include <benchmark/benchmark.h>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

//----------------------------------------------------------------------------//
// generate_table() algortithm

template <class T>
void generate_table_rs_seq_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    std::vector<T> b(nr * nc), bs(nr * nc), r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };

    for (auto _ : st)
        one4all::generate_table_rs<pcg32>
        (   r.begin()
        ,   b.begin()
        ,   nr
        ,   nc
        ,   seed_pi
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(generate_table_rs_seq_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

template <class T>
void generate_table_bs_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    std::vector<T> b(nr * nc), bs(nr * nc), r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };

    for (auto _ : st)
        one4all::generate_table_bs<pcg32>
        (   r.begin()
        ,   b.begin()
        ,   nr
        ,   nc
        ,   seed_pi
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(generate_table_bs_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// scale_table() algorithm

template <class T>
void scale_table_seq_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    std::vector<T> b(nr * nc), bs(nr * nc), r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };

    one4all::generate_table<pcg32>
    (   r.begin()
    ,   b.begin()
    ,   nr
    ,   nc
    ,   seed_pi
    );

    for (auto _ : st)
        one4all::scale_table
        (   r.begin()
        ,   b.begin()
        ,   bs.begin()
        ,   nr
        ,   nc
        ,   T(-1.0), T(1.0)
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T) * 2) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(scale_table_seq_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

template <class T>
void scale_table_par_x8(benchmark::State& st)
{   size_t nr = size_t(st.range());
    size_t nc = 8;
    std::vector<T> b(nr * nc), bs(nr * nc), r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };

    one4all::generate_table<pcg32>
    (   r.begin()
    ,   b.begin()
    ,   nr
    ,   nc
    ,   seed_pi
    );

    for (auto _ : st)
        one4all::scale_table
        (   std::execution::par
        ,   r.begin()
        ,   b.begin()
        ,   bs.begin()
        ,   nr
        ,   nc
        ,   T(-1.0), T(1.0)
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (nr * nc * sizeof(T) * 2) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    |   benchmark::Counter::kAvgThreadsRate
    );
}

BENCHMARK_TEMPLATE(scale_table_par_x8, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// main()

BENCHMARK_MAIN();