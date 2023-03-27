#include <fstream>
#include <execution>

#include <catch2/catch_all.hpp>

#include <one4all/pcg/pcg_random.hpp>
#include <one4all/algorithm/generate_table.hpp>
#include <one4all/algorithm/scale_table.hpp>

const unsigned long seed_pi{3141592654};

TEMPLATE_TEST_CASE( "generate_table() x3 - STL", "[10Kx3]", float, double )
{   typedef TestType T;
    const auto nr{10'000}, nc{3};

    std::vector<T> vrs(nr * nc);
    std::vector<T> r
    {   T(0), T (1), T  (10)  // mins
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
    {   CHECK( std::all_of
        (   tbb::counting_iterator<size_t>(0)
        ,   tbb::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( vrs[i] >= r[i % nc] && vrs[i] < r[ (i % nc) + nc] ); }
        ) );
    }

    SECTION("block_splitting")
    {   std::vector<T> vbs(nr * nc);
        one4all::generate_table_bs<pcg32>
        // one4all::generate_table_bs<pcg32>
        (   r.begin()
        ,   vbs.begin()
        ,   nr
        ,   nc
        ,   seed_pi
        );

        CHECK( std::all_of
        (   std::execution::par
        ,   tbb::counting_iterator<size_t>(0)
        ,   tbb::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( std::abs(vrs[i] - vbs[i]) < 0.00001 ); }
        ) );
    }
}

TEMPLATE_TEST_CASE( "scale_table() x8 - STL", "[1Kx8]", float, double )
{   typedef TestType T;
    const auto nr{1000}, nc{8};
    std::vector<T> b(nr * nc), bsr(nr * nc), r
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

    std::ifstream file(ONE4ALL_TEST_DATA_PATH"/scale_table_ref.txt");
    std::copy
    (   std::istream_iterator<T>(file)
    ,   std::istream_iterator<T>()
    ,   bsr.begin()
    );

    SECTION("seq")
    {   std::vector<T> bs(nr * nc);
        one4all::scale_table
        (   r.begin()
        ,   b.begin()
        ,   bs.begin()
        ,   nr
        ,   nc
        ,   T(-1.0), T(1.0)
        );
        CHECK( std::all_of
        (   tbb::counting_iterator<size_t>(0)
        ,   tbb::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( std::abs(bsr[i] - bs[i]) < 0.00001 ); }
        )   );
    }

    SECTION("par")
    {   std::vector<T> bs(nr * nc);
        one4all::scale_table
        (   std::execution::par
        ,   r.begin()
        ,   b.begin()
        ,   bs.begin()
        ,   nr
        ,   nc
        ,   -1.0, 1.0
        );
        CHECK( std::all_of
        (   std::execution::par
        ,   tbb::counting_iterator<size_t>(0)
        ,   tbb::counting_iterator<size_t>(nr * nc)
        ,   [&] (size_t i)
            { return ( std::abs(bsr[i] - bs[i]) < 0.00001 ); }
        )   );
    }
}
