#ifndef _ONE4ALL_ALGORITHM_SCALE_TABLE_HPP_
#define _ONE4ALL_ALGORITHM_SCALE_TABLE_HPP_

//----------------------------------------------------------------------------//
// serial version

namespace one4all {

template
<   typename T
,   typename Input1T
,   typename Input2T
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
>
inline void scale_table
(   Input1T range
,   Input2T in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   T tmin
,   T tmax
)
{   auto min = range;
    auto max = range + nc;
    for (size_t i = 0; i < nr * nc; ++i)
    {   CSizeT idx = i % nc;
        *   ( out + i )
        =   ( *(in + i) - *(min + idx) )
        /   ( *(max + idx) - *(min + idx) )
        *   ( tmax - tmin )
        +   tmin
        ;
    }
}

} // end of one4all namespace

//----------------------------------------------------------------------------//
// oneAPI version

#if defined(__INTEL_LLVM_COMPILER) && defined(SYCL_LANGUAGE_VERSION)

#   include <oneapi/dpl/execution>
#   include <oneapi/dpl/algorithm>
#   include <oneapi/dpl/numeric>
#   include <oneapi/dpl/memory>
#   include <oneapi/dpl/iterator>

namespace one4all::oneapi {

template
<   typename T
,   typename Input1T
,   typename Input2T
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
>
inline void scale_table
(   Input1T range
,   Input2T in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   T tmin
,   T tmax
)
{
    sycl::queue q;
    q.submit
    (   [&](sycl::handler& h)
        {
            const RSizeT threads_per_block{256};
            const RSizeT blocks_per_grid{nr / threads_per_block + 1};
            const RSizeT size{blocks_per_grid * threads_per_block};
            sycl::buffer buf_range = range.get_buffer();
            sycl::buffer buf_in = in.get_buffer();
            sycl::buffer buf_out = out.get_buffer();
            sycl::accessor ra(buf_range, h , sycl::read_only);
            sycl::accessor ia(buf_in, h , sycl::read_only);
            sycl::accessor oa(buf_out, h , sycl::write_only);
            h.parallel_for
            (   sycl::nd_range<1>(sycl::range<1>(blocks_per_grid * threads_per_block), sycl::range<1>(threads_per_block))
            ,   [=](sycl::nd_item<1> itm)
                {   auto idx{(itm.get_group(0) * itm.get_local_range(0) * nc) + (itm.get_local_id(0) * nc)};
                    if (idx < nr * nc)
                    {   for (CSizeT i = 0; i < nc; ++i)
                        {   oa[idx + i]
                            =   ( ia[idx + i] - ra[i] )
                            /   ( ra[nc + i] - ra[i] )
                            *   ( tmax - tmin )
                            +   tmin;
                        }
                    }
                }
            );
        }
    ).wait();
}

} // end one4all::oneapi namespace

//----------------------------------------------------------------------------//
// CUDA version

#elif defined(__CUDACC__)

#include <thrust/iterator/counting_iterator.h>

namespace one4all::cuda {

template
<   typename T
,   typename Input1T
,   typename Input2T
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
>
inline void scale_table
(   Input1T range
,   Input2T in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   T tmin
,   T tmax
)
{   auto min = range;
    auto max = range + nc;
    thrust::for_each
    (   thrust::counting_iterator<RSizeT>(0)
    ,   thrust::counting_iterator<RSizeT>(nr * nc)
    ,   [=] __host__ __device__ (RSizeT i)
        {   CSizeT idx = i % nc;
            *   ( out + i )
            =   ( *(in + i) - *(min + idx) )
            /   ( *(max + idx) - *(min + idx) )
            *   ( tmax - tmin )
            +   tmin
            ;
        }
    );
}

} // end one4all::cuda namespace

//----------------------------------------------------------------------------//
// ROCm version

#elif defined(__HIP_PLATFORM_AMD__)

#include <thrust/iterator/counting_iterator.h>

namespace one4all::rocm {

template
<   typename T
,   typename Input1T
,   typename Input2T
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
>
inline void scale_table
(   Input1T range
,   Input2T in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   T tmin
,   T tmax
)
{   auto min = range;
    auto max = range + nc;
    thrust::for_each
    (   thrust::counting_iterator<RSizeT>(0)
    ,   thrust::counting_iterator<RSizeT>(nr * nc)
    ,   [=] __host__ __device__ (RSizeT i)
        {   CSizeT idx = i % nc;
            *   ( out + i )
            =   ( *(in + i) - *(min + idx) )
            /   ( *(max + idx) - *(min + idx) )
            *   ( tmax - tmin )
            +   tmin
            ;
        }
    );
}

} // end one4all::rocm namespace

//----------------------------------------------------------------------------//

#else

#   include <algorithm>
#   include <tbb/iterators.h>

namespace one4all {

//----------------------------------------------------------------------------//

template
<   typename T
,   typename ExecutionPolicyT
,   typename Input1T
,   typename Input2T
,   typename OutputT
,   typename RSizeT
,   typename CSizeT
>
inline void scale_table
(   ExecutionPolicyT&& policy
,   Input1T range
,   Input2T in
,   OutputT out
,   RSizeT nr
,   CSizeT nc
,   T tmin
,   T tmax
)
{   auto min = range;
    auto max = range + nc;
    std::for_each
    (   std::forward<ExecutionPolicyT>(policy)
    ,   tbb::counting_iterator<RSizeT>(0)
    ,   tbb::counting_iterator<RSizeT>(nr * nc)
    ,   [=] (RSizeT i)
        {   CSizeT idx = i % nc;
            *   ( out + i )
            =   ( *(in + i) - *(min + idx) )
            /   ( *(max + idx) - *(min + idx) )
            *   ( tmax - tmin )
            +   tmin
            ;
        }
    );
}

} // end of one4all namespace 

#endif  //__INTEL_LLVM_COMPILER && SYCL_LANGUAGE_VERSION

#endif  // _ONE4ALL_ALGORITHM_SCALE_TABLE_HPP_