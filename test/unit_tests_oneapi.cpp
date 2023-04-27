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
    T* vrs = sycl::malloc_device<T>(nr * nc, q);
    T* vbs = sycl::malloc_device<T>(nr * nc, q);
    T* r = sycl::malloc_device<T>(nc * 2, q);
    std::vector<T> vrsh(nr * nc), rh
    {   T(0), T( 1), T(  10)  // mins
    ,   T(1), T(10), T(1000)  // maxs
    };

    one4all::generate_table_rs<pcg32>
    (   std::begin(rh)
    ,   std::begin(vrsh)
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
            { return ( vrsh[i] >= rh[i % nc] && vrsh[i] < rh[ (i % nc) + nc] ); }
        ) );
    }

    SECTION("block_splitting")
    {   q.memcpy(r, rh.data(), nc * 2 * sizeof(T));
        one4all::oneapi::generate_table<pcg32>
        (   r
        ,   vbs
        ,   nr
        ,   nc
        ,   seed_pi
        ,   q
        ).wait();
        q.memcpy(vrs, vrsh.data(), nr * nc * sizeof(T));

        CHECK( std::all_of(
            oneapi::dpl::execution::make_device_policy(q)
        ,   dpl::counting_iterator<size_t>(0)
        ,   dpl::counting_iterator<size_t>(nr * nc)
        ,   [=] (size_t i)
            { return ( std::abs(vrs[i] - vbs[i]) < 0.0001 ); }
        ) );
    }

    sycl::free(r, q); sycl::free(vrs, q); sycl::free(vbs, q);
}

TEMPLATE_TEST_CASE( "scale_table() x8 - oneapi", "[1Kx8]", float, double )
{   typedef TestType T;
    const auto nr{1000}, nc{8};
    sycl::queue q;
    T* b = sycl::malloc_device<T>(nr * nc, q);
    T* r = sycl::malloc_device<T>(nc * 2, q);
    std::vector<T> rh
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };

    q.memcpy(r, rh.data(), nc * 2 * sizeof(T));
    one4all::oneapi::generate_table<pcg32>
    (   r
    ,   b
    ,   nr
    ,   nc
    ,   seed_pi
    ).wait();

    sycl::usm_allocator<T,sycl::usm::alloc::shared> alloc(q);
    std::vector<T, decltype(alloc)> bsr(nr * nc, alloc);
    std::ifstream file(ONE4ALL_TEST_DATA_PATH"/scale_table_ref.txt");
    std::copy
    (   std::istream_iterator<T>(file)
    ,   std::istream_iterator<T>()
    ,   std::begin(bsr)
    );

    T* bs = sycl::malloc_device<T>(nr * nc, q);
    one4all::oneapi::scale_table
    (   r
    ,   b
    ,   bs
    ,   nr
    ,   nc
    ,   T(-1.0), T(1.0)
    ,   q
    ).wait();

    auto begin = dpl::make_zip_iterator(std::begin(bsr), bs);
    CHECK( std::all_of
    (   oneapi::dpl::execution::make_device_policy(q)
    ,   begin
    ,   begin + (nr * nc)
    ,   [] (auto t)
        { return ( std::abs(std::get<0>(t) - std::get<1>(t)) < 0.001 ); }
    )   );

    sycl::free(r, q); sycl::free(b, q); sycl::free(bs, q);
}
