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
    std::vector<T> vrs(nr * nc);
    std::vector<T> r
    {   T(0), T( 1), T(  10)  // mins
    ,   T(1), T(10), T(1000)  // maxs
    };

    one4all::generate_table_rs<pcg32>
    (   r.begin()
    ,   vrs.begin()
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
        sycl::buffer<T,1> dvr(r), dvbs{sycl::range(nr * nc)};
        one4all::oneapi::generate_table<pcg32>
        (   oneapi::dpl::begin(dvr)
        ,   oneapi::dpl::begin(dvbs)
        ,   nr
        ,   nc
        ,   seed_pi
        );

        sycl::host_accessor vbs{dvbs, sycl::read_only};

        // std::ofstream out("vbs.txt");
        // for (size_t i = 0; i < nr * nc; ++i)
        // {
        //     if (0 == i % nc)
        //         out << std::endl;
        //     out << std::fixed
        //         << std::setw(10)
        //         << std::setprecision(3)
        //         << vbs[i];
        // }

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
    );

    std::vector<T> bsr(nr * nc);
    std::ifstream file(ONE4ALL_TEST_DATA_PATH"/scale_table_ref.txt");
    std::copy
    (   std::istream_iterator<T>(file)
    ,   std::istream_iterator<T>()
    ,   std::begin(bsr)
    );

    sycl::buffer<T> bs{nr * nc}, dbsr(bsr);
    one4all::oneapi::scale_table
    (   dpl::begin(dr, sycl::read_only, sycl::no_init)
    ,   dpl::begin(b, sycl::read_only, sycl::no_init)
    ,   dpl::begin(bs, sycl::write_only, sycl::no_init)
    ,   nr
    ,   nc
    ,   T(-1.0), T(1.0)
    );

    auto begin = dpl::make_zip_iterator(dpl::begin(dbsr), dpl::begin(bs));
    CHECK( std::all_of
    (   dpl::execution::dpcpp_default
    ,   begin
    ,   begin + (nr * nc)
    ,   [] (auto t)
        { return ( std::abs(std::get<0>(t) - std::get<1>(t)) < 0.001 ); }
    )   );
}
