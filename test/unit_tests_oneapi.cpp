// #include <fstream>
// #include <iomanip>

#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include <catch2/catch_all.hpp>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

TEST_CASE( "Device Info - oneAPI")
{
    sycl::queue q;
    WARN("Running on: " << q.get_device().get_info<sycl::info::device::name>());
}

TEMPLATE_TEST_CASE( "generate_table() x3 - oneAPI", "[oneAPI][10Kx3]", float, double )
{   typedef TestType T;
    const auto nr{10'000}, nc{3};
    sycl::queue q;
    sycl::usm_allocator<T,sycl::usm::alloc::shared> alloc(q);
    std::vector<T, decltype(alloc)> vrs(nr * nc, alloc);
    std::vector<T> r
    {   T(0), T( 1), T(  10)  // mins
    ,   T(1), T(10), T(1000)  // maxs
    };

    one4all::generate_table_rs<pcg32>
    (   std::begin(r)
    ,   std::begin(vrs)
    ,   nr
    ,   nc
    ,   seed_pi
    );

    SECTION("random_seeding")
    {
        CHECK( std::all_of
        (   dpl::counting_iterator<size_t>(0)
        ,   dpl::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( vrs[i] >= r[i % nc] && vrs[i] < r[ (i % nc) + nc] ); }
        ) );
    }

    SECTION("block_splitting")
    {
        std::vector<T, decltype(alloc)> vbs(nr * nc, alloc), dr(nc * 2, alloc);
        std::copy_n(std::begin(r), nc * 2, std::begin(dr));
        one4all::oneapi::generate_table<pcg32>
        (   std::begin(dr)
        ,   std::begin(vbs)
        ,   nr
        ,   nc
        ,   seed_pi
        );

        CHECK( std::all_of(
            dpl::counting_iterator<size_t>(0)
        ,   dpl::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( std::abs(vrs[i] - vbs[i]) < 0.0001 ); }
        ) );
    }
}

TEMPLATE_TEST_CASE( "scale_table() x8 - oneapi", "[1Kx8]", float, double )
{   typedef TestType T;
    const auto nr{1000}, nc{8};
    sycl::queue q;
    sycl::usm_allocator<T,sycl::usm::alloc::shared> alloc(q);
    std::vector<T, decltype(alloc)> b(nr * nc, alloc), dr(nc * 2, alloc);
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };

    std::copy_n(std::begin(r), nc * 2, std::begin(dr));
    one4all::oneapi::generate_table<pcg32>
    (   std::begin(dr)
    ,   std::begin(b)
    ,   nr
    ,   nc
    ,   seed_pi
    );

    std::vector<T, decltype(alloc)> bsr(nr * nc, alloc);
    std::ifstream file(ONE4ALL_TEST_DATA_PATH"/scale_table_ref.txt");
    std::copy
    (   std::istream_iterator<T>(file)
    ,   std::istream_iterator<T>()
    ,   std::begin(bsr)
    );

    std::vector<T, decltype(alloc)> bs(nr * nc, alloc);
    one4all::oneapi::scale_table
    (   std::begin(dr)
    ,   std::begin(b)
    ,   std::begin(bs)
    ,   nr
    ,   nc
    ,   T(-1.0), T(1.0)
    );

    auto begin = dpl::make_zip_iterator(std::begin(bsr), std::begin(bs));
    CHECK( std::all_of
    (   oneapi::dpl::execution::make_device_policy(q)
    ,   begin
    ,   begin + (nr * nc)
    ,   [] (auto t)
        { return ( std::abs(std::get<0>(t) - std::get<1>(t)) < 0.001 ); }
    )   );
}
