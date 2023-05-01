#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include <catch2/catch_all.hpp>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

TEST_CASE( "Device Info - oneAPI")
{   try
    {   sycl::queue q;
        // auto q = sycl::queue{usm_selector{}};
        WARN("Running on: " << q.get_device().get_info<sycl::info::device::name>());
    }
    catch(const sycl::exception& e)
    {   std::cerr << "Error: " << e.what() << std::endl;
    }
    REQUIRE(true);
}

TEMPLATE_TEST_CASE( "generate_table() x3 - oneAPI", "[oneAPI][10Kx3]", float, double )
{   typedef TestType T;
    const auto nr{10'000}, nc{3};
    sycl::queue q;
    std::vector<T> vrs(nr * nc), r
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
    {   sycl::buffer<T> dr(r), dvbs{sycl::range(nr * nc)};
        one4all::oneapi::generate_table<pcg32>
        (   dpl::begin(dr)
        ,   dpl::begin(dvbs)
        ,   nr
        ,   nc
        ,   seed_pi
        ,   q
        ).wait();

        sycl::host_accessor vbs{dvbs, sycl::read_only};

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
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };
    sycl::buffer<T> b{nr * nc}, dr(r);

    one4all::oneapi::generate_table<pcg32>
    (   dpl::begin(dr)
    ,   dpl::begin(b)
    ,   nr
    ,   nc
    ,   seed_pi
    ).wait();

    std::vector<T> bsr(nr * nc);
    std::ifstream file(ONE4ALL_TEST_DATA_PATH"/scale_table_ref.txt");
    std::copy
    (   std::istream_iterator<T>(file)
    ,   std::istream_iterator<T>()
    ,   std::begin(bsr)
    );

    sycl::buffer<T> bs{nr * nc}, dbsr(bsr);
    one4all::oneapi::scale_table
    (   dpl::begin(dr)
    ,   dpl::begin(b)
    ,   dpl::begin(bs)
    ,   nr
    ,   nc
    ,   T(-1.0), T(1.0)
    ,   q
    ).wait();

    auto begin = dpl::make_zip_iterator(dpl::begin(dbsr), dpl::begin(bs));
    CHECK( std::all_of
    (   dpl::execution::make_device_policy(q)
    ,   begin
    ,   begin + (nr * nc)
    ,   [] (auto t)
        { return ( std::abs(std::get<0>(t) - std::get<1>(t)) < 0.001 ); }
    )   );
}
