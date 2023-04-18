#ifndef _ONE4ALL_ALGORITHM_GENERATE_TABLE_HPP_
#define _ONE4ALL_ALGORITHM_GENERATE_TABLE_HPP_

#include <vector>

#include <one4all/trng/uniform_dist.hpp>

//----------------------------------------------------------------------------//
// random seeding technique

namespace one4all {

template
<   typename RngT
,   typename InputT
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
,   typename SeedT = unsigned long
>
inline void generate_table_rs
(   InputT in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   SeedT s = 0
)
{   typedef typename std::iterator_traits<OutputT>::value_type T;
    RngT r(s);
    std::vector<trng::uniform_dist<T>> u;
    u.reserve(nc);
    for (CSizeT i = 0; i < nc; ++i)
        u.emplace_back( in[i], in[nc + i]);
    for (RSizeT i = 0; i < nr * nc; ++i)
        out[i] = u[i % nc](r);
}

} // end one4all namespace

//----------------------------------------------------------------------------//
// block splitting technique (oneAPI)

#if defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)

#   include <oneapi/dpl/algorithm>
#   include <oneapi/dpl/iterator>

namespace one4all::oneapi {

template
<   typename RngT
,   typename InputT
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
,   typename SeedT = unsigned long
>
inline void generate_table
(   InputT in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   SeedT s = 0
)
{   typedef typename std::iterator_traits<OutputT>::value_type T;

    sycl::queue q;
    q.submit
    (   [&](sycl::handler& h)
        {
            const RSizeT threads_per_block{256};
            const RSizeT blocks_per_grid{nr / threads_per_block + 1};
            const RSizeT size{blocks_per_grid * threads_per_block};
            sycl::buffer buf_in = in.get_buffer();
            sycl::buffer buf_out = out.get_buffer();
            sycl::accessor ia(buf_in, h , sycl::read_only);
            sycl::accessor oa(buf_out, h , sycl::write_only, sycl::no_init);
            h.parallel_for
            (   sycl::nd_range<1>
                (   sycl::range<1>(blocks_per_grid * threads_per_block)
                ,   sycl::range<1>(threads_per_block)
                )
            ,   [=](sycl::nd_item<1> itm)
                {
                    RngT r(s);
                    auto idx
                    {(  itm.get_group(0)
                    *   itm.get_local_range(0)
                    *   nc )
                    +(  itm.get_local_id(0)
                    *   nc )
                    };
                    if (idx < nr * nc)
                    {   r.discard(idx);
                        for (CSizeT i = 0; i < nc; ++i)
                        {   trng::uniform_dist<T> u(ia[i], ia[i + nc]);
                            oa[idx + i] = u(r);
                        }
                    }
                }
            );
        }
    );
}

} // end one4all::oneapi namespace

//----------------------------------------------------------------------------//
// block splitting technique (CUDA)

#elif defined(__CUDACC__)

namespace one4all::cuda {

template
<   typename RngT
,   typename T
,   typename RSizeT
,   typename CSizeT
>
__global__ void block_splitting
(   T* in
,   T* out
,   RngT r
,   RSizeT nr
,   CSizeT nc
)
{   auto idx{(blockIdx.x * blockDim.x * nc) + (threadIdx.x * nc)};
    if (idx < nr * nc)
    {   r.discard(idx);
        for (CSizeT i = 0; i < nc; ++i)
        {   trng::uniform_dist<T> u(in[i], in[i + nc]);
            out[idx + i] = u(r); 
        }
    }
}

template
<   typename RngT
,   typename InputT
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
,   typename SeedT = unsigned long
>
inline void generate_table
(   InputT in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   SeedT s = 0
)
{   RngT r(s);
    const RSizeT threads_per_block{256};
    RSizeT blocks_per_grid{nr / threads_per_block + 1}; 
    block_splitting<<<blocks_per_grid, threads_per_block>>>
    (   thrust::raw_pointer_cast(&in[0])
    ,   thrust::raw_pointer_cast(&out[0])
    ,   r
    ,   nr
    ,   nc
    );
}

} // end one4all::cuda namespace

//----------------------------------------------------------------------------//
// block splitting technique (ROCm)

#elif defined(__HIP_PLATFORM_AMD__)

namespace one4all::rocm {

template
<   typename RngT
,   typename T
,   typename RSizeT
,   typename CSizeT
>
__global__ void block_splitting
(   T* in
,   T* out
,   RngT r
,   RSizeT nr
,   CSizeT nc
)
{   auto idx{(blockIdx.x * blockDim.x * nc) + (threadIdx.x * nc)};
    if (idx < nr * nc)
    {   r.discard(idx);
        for (CSizeT i = 0; i < nc; ++i)
        {   trng::uniform_dist<T> u(in[i], in[i + nc]);
            out[idx + i] = u(r); 
        }
    }
}

template
<   typename RngT
,   typename InputT
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
,   typename SeedT = unsigned long
>
inline void generate_table
(   InputT in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   SeedT s = 0
)
{   RngT r(s);
    const RSizeT threads_per_block{256};
    RSizeT blocks_per_grid{nr / threads_per_block + 1}; 
    block_splitting<<<blocks_per_grid, threads_per_block>>>
    (   thrust::raw_pointer_cast(&in[0])
    ,   thrust::raw_pointer_cast(&out[0])
    ,   r
    ,   nr
    ,   nc
    );
}

} // end one4all::rocm namespace

#else

#   include <algorithm>
#   include <thread>
#   include <omp.h>

namespace one4all {

//----------------------------------------------------------------------------//
// block splitting technique (OpenMP)

template
<   typename RngT
,   typename InputT
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
,   typename SeedT = unsigned long
>
inline void generate_table_bs
(   InputT in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   SeedT s = 0
)
{   typedef typename std::iterator_traits<OutputT>::value_type T;
    #pragma omp parallel
    {   RngT   r(s);
        auto   tidx{omp_get_thread_num()};
        auto   size{omp_get_num_threads()};
        RSizeT first{(tidx * nr / size) * nc};
        RSizeT  last{((tidx + 1) * nr / size) * nc};
        std::vector<trng::uniform_dist<T>> u;
        u.reserve(nc);
        for (CSizeT i = 0; i < nc; ++i)
            u.emplace_back(in[i], in[i + nc]);
        r.discard(first);
        for (RSizeT i{first}; i < last; ++i)
            out[i] = u[i % nc](r);
    }
}

//----------------------------------------------------------------------------//
// set block splitting as the default algorithm using a function alias

template
<   typename ...ExplicitArgs
,   typename... Args
>
inline void generate_table(Args&&... args)
{   generate_table_bs<ExplicitArgs...>(std::forward<Args>(args)...);   }

} // end one4all namespace

#endif  //__INTEL_LLVM_COMPILER && SYCL_LANGUAGE_VERSION

#endif  //_ONE4ALL_ALGORITHM_GENERATE_TABLE_HPP_