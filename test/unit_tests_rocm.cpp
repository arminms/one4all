#include <fstream>
#include <iomanip>

#include <catch2/catch_all.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

struct equal
{   template <typename Tuple>
    __host__ __device__
    bool operator()(Tuple t)
    {   return (std::abs(thrust::get<0>(t) - thrust::get<1>(t)) < 0.001);   }
};

struct equal_e_0_1
{   template <typename Tuple>
    __host__ __device__
    bool operator()(Tuple t)
    {   return (std::abs(thrust::get<0>(t) - thrust::get<1>(t)) < 0.1);   }
};

TEST_CASE( "Device Info - ROCm")
{   hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    WARN("Running on: " << prop.name);
}

TEMPLATE_TEST_CASE("generate_table() x3 - ROCm", "[ROCm][10Kx3]", float, double)
{   const auto nr{10'000}, nc{3};
    std::vector<TestType> vrs(nr * nc);
    std::vector<TestType> r
    {   TestType(0), TestType (1), TestType  (10)  // mins
    ,   TestType(1), TestType(10), TestType(1000)  // maxs
    };

    one4all::generate_table_rs<pcg32>
    (   r.begin()
    ,   vrs.begin()
    ,   nr
    ,   nc
    ,   seed_pi
    );

    SECTION("random_seeding")
    {   CHECK( std::all_of
        (   thrust::counting_iterator<size_t>(0)
        ,   thrust::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( vrs[i] >= r[i % nc] && vrs[i] < r[ (i % nc) + nc] ); }
        ) );
    }

    SECTION("block_splitting")
    {   thrust::device_vector<TestType> dvr(r), dvbs(nr * nc);
        one4all::rocm::generate_table<pcg32>
        (   dvr.begin()
        ,   dvbs.begin()
        ,   nr
        ,   nc
        ,   seed_pi
        );

        std::vector<TestType> vbs(nr * nc);
        thrust::copy(dvbs.begin(), dvbs.end(), vbs.begin());

        thrust::device_vector<TestType> dvrs(vrs);
        CHECK( thrust::all_of
        (   thrust::make_zip_iterator(thrust::make_tuple(dvrs.begin(), dvbs.begin()))
        ,   thrust::make_zip_iterator(thrust::make_tuple(dvrs.end(), dvbs.end()))
        ,   equal()
        ) );
    }
}

TEMPLATE_TEST_CASE( "scale_table() x8 - ROCm", "[ROCm][1Kx8]", float, double )
{   typedef TestType T;
    const auto nr{1000}, nc{8};
    std::vector<T> r
    {   T(-10), T(-5), T(-1), T(0), T(1), T( 5), T(10), T(15)  // mins
    ,   T( -5), T(-1), T( 0), T(1), T(5), T(10), T(15), T(20)  // maxs
    };
    thrust::device_vector<T> b(nr * nc), dr(r);

    one4all::rocm::generate_table<pcg32>
    (   dr.begin()
    ,   b.begin()
    ,   nr
    ,   nc
    ,   seed_pi
    );

    std::vector<T> bsr(nr * nc);
    std::ifstream file(ONE4ALL_TEST_DATA_PATH"/scale_table_ref.txt");
    std::copy
    (   std::istream_iterator<T>(file)
    ,   std::istream_iterator<T>()
    ,   bsr.begin()
    );
    thrust::device_vector<T> dbsr(bsr);

    thrust::device_vector<T> bs(nr * nc);
    one4all::rocm::scale_table
    (   dr.begin()
    ,   b.begin()
    ,   bs.begin()
    ,   nr
    ,   nc
    ,   T(-1.0), T(1.0)
    );

    CHECK( thrust::all_of
    (   thrust::make_zip_iterator(thrust::make_tuple(dbsr.begin(), bs.begin()))
    ,   thrust::make_zip_iterator(thrust::make_tuple(dbsr.end(), bs.end()))
    ,   equal()
    )   );
}
