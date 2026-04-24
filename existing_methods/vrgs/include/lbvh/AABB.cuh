#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH
#include <vector_types.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace lbvh {

struct Node {
    std::uint32_t parent_idx; // parent node
    std::uint32_t left_idx;   // index of left  child node
    std::uint32_t right_idx;  // index of right child node
};

template<typename T> struct vector_of;
template<> struct vector_of<float>  {using type = float4;};
template<> struct vector_of<double> {using type = double4;};

template<typename T>
using vector_of_t = typename vector_of<T>::type;


// AABB Bouding Box

template<typename T>
struct aabb
{
    typename vector_of<T>::type upper;
    typename vector_of<T>::type lower;
};

template<typename T>
__device__ __host__
inline bool intersects(const aabb<T>& lhs, const aabb<T>& rhs) noexcept
{
    if(lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x) {return false;}
    if(lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y) {return false;}
    if(lhs.upper.z < rhs.lower.z || rhs.upper.z < lhs.lower.z) {return false;}
    return true;
}

template<typename T>
__device__ __host__
inline aabb<T> merge(const aabb<T>& lhs, const aabb<T>& rhs) noexcept
{
    aabb<T> merged;
    merged.upper.x = max(lhs.upper.x, rhs.upper.x);
    merged.upper.y = max(lhs.upper.y, rhs.upper.y);
    merged.upper.z = max(lhs.upper.z, rhs.upper.z);
    merged.lower.x = min(lhs.lower.x, rhs.lower.x);
    merged.lower.y = min(lhs.lower.y, rhs.lower.y);
    merged.lower.z = min(lhs.lower.z, rhs.lower.z);
    return merged;
}

template<typename T>
__device__ __host__
inline typename vector_of<T>::type centroid(const aabb<T>& box) noexcept
{
    typename vector_of<T>::type c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

}

#endif