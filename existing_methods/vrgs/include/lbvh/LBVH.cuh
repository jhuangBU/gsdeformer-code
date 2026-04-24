#ifndef LBVH_LBVH_CUH
#define LBVH_LBVH_CUH

#include <cub/cub.cuh>
#include "CudaKernels.h"
#include "AABB.cuh"
#include "PointTriangleDistance.cuh"

namespace lbvh {

template<typename T>
__device__
inline T infinity() noexcept;

template<>
__device__
inline float  infinity<float >() noexcept {return CUDART_INF_F;}
template<>
__device__
inline double infinity<double>() noexcept {return CUDART_INF;}


__device__ __host__
inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
__device__ __host__
inline std::uint32_t morton_code(float4 xyz, float resolution = 1024.0f) noexcept {
    xyz.x = ::fminf(::fmaxf(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y = ::fminf(::fmaxf(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = ::fminf(::fmaxf(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

__device__ __host__
inline std::uint32_t morton_code(double4 xyz, double resolution = 1024.0) noexcept {
    xyz.x = ::fmin(::fmax(xyz.x * resolution, 0.0), resolution - 1.0);
    xyz.y = ::fmin(::fmax(xyz.y * resolution, 0.0), resolution - 1.0);
    xyz.z = ::fmin(::fmax(xyz.z * resolution, 0.0), resolution - 1.0);
    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

__device__
inline int common_upper_bits(const uint32_t lhs, const uint32_t rhs) noexcept
{
    return ::__clz(lhs ^ rhs);
}

__device__
inline int common_upper_bits(const uint64_t lhs, const uint64_t rhs) noexcept {
    return ::__clzll(lhs ^ rhs);
}

inline unsigned int next_pow_2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

__global__ void compute_tri_aabbs(const int num_faces, 
                                  const float *verts, 
                                  const int *faces,
                                  aabb<float> *aabbs,
                                  const float extended_dist = 0.0f
                                  ) {
    uint32_t f = threadIdx.x + blockDim.x * blockIdx.x;
    if (f < num_faces) {
        int v0 = faces[f*3+0];
		int v1 = faces[f*3+1];
		int v2 = faces[f*3+2];
		aabbs[f].lower.x = fminf(fminf(verts[v0*3+0], verts[v1*3+0]), verts[v2*3+0]) - extended_dist;
		aabbs[f].lower.y = fminf(fminf(verts[v0*3+1], verts[v1*3+1]), verts[v2*3+1]) - extended_dist;
		aabbs[f].lower.z = fminf(fminf(verts[v0*3+2], verts[v1*3+2]), verts[v2*3+2]) - extended_dist;

		aabbs[f].upper.x = fmaxf(fmaxf(verts[v0*3+0], verts[v1*3+0]), verts[v2*3+0]) + extended_dist;
		aabbs[f].upper.y = fmaxf(fmaxf(verts[v0*3+1], verts[v1*3+1]), verts[v2*3+1]) + extended_dist;
		aabbs[f].upper.z = fmaxf(fmaxf(verts[v0*3+2], verts[v1*3+2]), verts[v2*3+2]) + extended_dist;
    }
}

__global__ void query_collision_triangles(const int num_pairs,
                                          const float minimal_dist,
                                          const int2 *candidate_pairs,
                                          const float *verts, 
                                          const int *verts_group,
                                          const int *faces,
                                          int4 *exact_collision_pairs,
                                          int *total_exact_pairs,
                                          const int max_collision_pairs
                                          ) {
    uint32_t t = threadIdx.x + blockDim.x * blockIdx.x;
    if (t < num_pairs) {
        int f0 = candidate_pairs[t].x;
        int f1 = candidate_pairs[t].y;
        int v00 = faces[f0*3+0];
        int v01 = faces[f0*3+1];
        int v02 = faces[f0*3+2];
        int v10 = faces[f1*3+0];
        int v11 = faces[f1*3+1];
        int v12 = faces[f1*3+2];

        if (v00 == v10 || v00 == v11 || v00 == v12 ||
            v01 == v10 || v01 == v11 || v01 == v12 ||
            v02 == v10 || v02 == v11 || v02 == v12) return;
        
        // TODO(changyu): only consider obj-obj collision now...
        if (verts_group[v00] == verts_group[v10]) {
            return;
        }
        
        auto add_pairs = [&](int p, int p0, int p1, int p2) {
            if (point_triangle_distance(&verts[p*3], &verts[p0*3], &verts[p1*3], &verts[p2*3]) < minimal_dist) {
                int pair_idx = atomicAdd(total_exact_pairs, 1);
                if (pair_idx < max_collision_pairs) {
                    // Orient face to the point direction
                    float d1[3], d2[3], pp0[3], n[3];
                    for (int i = 0; i < 3; i++) {
                        d1[i] = verts[p1*3+i] - verts[p0*3+i];
                        d2[i] = verts[p2*3+i] - verts[p0*3+i];
                        pp0[i] = verts[p*3+i] - verts[p0*3+i];
                    }
                    cross_product3<float>(d1, d2, n);
                    if (dot<3, float>(n, pp0) < 0) {
                        const auto tmp = p1;
                        p1 = p2;
                        p2 = tmp;
                    }

                    exact_collision_pairs[pair_idx] = make_int4(p, p0, p1, p2);
                }
            }
        };
        
        add_pairs(v00, v10, v11, v12);
        add_pairs(v01, v10, v11, v12);
        add_pairs(v02, v10, v11, v12);
        add_pairs(v10, v00, v01, v02);
        add_pairs(v11, v00, v01, v02);
        add_pairs(v12, v00, v01, v02);
    }
}

template<class T>
__global__ void reset_aabb(size_t num_objects, const aabb<T> *tri_aabbs, aabb<T> *aabbs) {
    uint32_t t = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t num_internal_nodes = num_objects - 1;
    const size_t num_nodes = num_objects * 2 - 1;
    if (t < num_nodes) {
        if (t < num_internal_nodes) {
            aabbs[t].lower.x = infinity<T>();
            aabbs[t].lower.y = infinity<T>();
            aabbs[t].lower.z = infinity<T>();

            aabbs[t].upper.x = -infinity<T>();
            aabbs[t].upper.y = -infinity<T>();
            aabbs[t].upper.z = -infinity<T>();
        } else {
            aabbs[t] = tri_aabbs[t - num_internal_nodes];
        }
    }
}

template <class T, unsigned int blockSize>
__global__ void aabb_reduce(const aabb<T> *g_idata, aabb<T> *g_odata, const size_t n) {
    extern __shared__ aabb<T> sdata [];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    sdata[tid].lower.x = infinity<T>();
    sdata[tid].lower.y = infinity<T>();
    sdata[tid].lower.z = infinity<T>();
    sdata[tid].upper.x = -infinity<T>();
    sdata[tid].upper.y = -infinity<T>();
    sdata[tid].upper.z = -infinity<T>();

    while (i < n) { 
        sdata[tid] = merge(sdata[tid], g_idata[i]);
        if (i + blockSize < n) sdata[tid] = merge(sdata[tid], g_idata[i+blockSize]);;
        i += gridSize; 
    }
    __syncthreads();

    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = merge(sdata[tid], sdata[tid + 256]); } } __syncthreads();
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = merge(sdata[tid], sdata[tid + 128]); } } __syncthreads();
    if (blockSize >= 128) { if (tid < 64) { sdata[tid] = merge(sdata[tid], sdata[tid + 64]); } } __syncthreads();
    if (blockSize >= 64) { if (tid < 32) { sdata[tid] = merge(sdata[tid], sdata[tid + 32]); } } __syncthreads();
    if (blockSize >= 32) { if (tid < 16) { sdata[tid] = merge(sdata[tid], sdata[tid + 16]); } } __syncthreads();
    if (blockSize >= 16) { if (tid < 8) { sdata[tid] = merge(sdata[tid], sdata[tid + 8]); } } __syncthreads();
    if (blockSize >= 8) { if (tid < 4) { sdata[tid] = merge(sdata[tid], sdata[tid + 4]); } } __syncthreads();
    if (blockSize >= 4) { if (tid < 2) { sdata[tid] = merge(sdata[tid], sdata[tid + 2]); } } __syncthreads();
    if (blockSize >= 2) { if (tid < 1) { sdata[tid] = merge(sdata[tid], sdata[tid + 1]); } } __syncthreads();

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template<class T>
void launch_aabb_reduce(int blocks, int threads, const aabb<T> *g_idata, aabb<T> *g_odata, const size_t n) {
    size_t shared_size = sizeof(aabb<T>) * threads * ((threads <= 32) ? 2 : 1);
#define AABB(THREADS) case THREADS: aabb_reduce<T, THREADS> << < blocks, threads, shared_size >> > (g_idata, g_odata, n); break;
    switch(threads) {
        AABB(512)
        AABB(256)
        AABB(128)
        AABB(64)
        AABB(32)
        AABB(16)
        AABB(8)
        AABB(4)
        AABB(2)
        AABB(1)
    }
}

template<class T>
__global__ void compute_morton_and_indices(const size_t num_objects, const aabb<T> *whole_ptr, const aabb<T> *aabbs, uint64_t *morton, int *indices) {
    uint32_t t = threadIdx.x + blockDim.x * blockIdx.x;
    if (t < num_objects) {
        aabb<T> whole = whole_ptr[0];
        auto p = centroid(aabbs[t]);
        p.x -= whole.lower.x;
        p.y -= whole.lower.y;
        p.z -= whole.lower.z;
        p.x /= (whole.upper.x - whole.lower.x);
        p.y /= (whole.upper.y - whole.lower.y);
        p.z /= (whole.upper.z - whole.lower.z);
        morton[t] = uint64_t(morton_code(p)) << 32 | t;
        indices[t] = t;
    }
}

template<class T>
__global__ void get_sorted_tri_aabbs(const size_t num_objects, const int *sorted_indices, const aabb<T> *tri_aabbs, aabb<T> *sorted_tri_aabbs) {
    uint32_t t = threadIdx.x + blockDim.x * blockIdx.x;
    if (t < num_objects) {
        sorted_tri_aabbs[t] = tri_aabbs[sorted_indices[t]];
    }
}

template<class codeT>
__device__ inline uint2 determine_range(const codeT *node_code, const unsigned int num_leaves, unsigned int idx) {
    if(idx == 0) {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    const codeT self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx-1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx+1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves) {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min) {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0) {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min) {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0) {
        unsigned int tmp = idx;
        idx = jdx;
        jdx = tmp;
    }
    return make_uint2(idx, jdx);
}

template<class codeT>
__device__ inline unsigned int find_split(const codeT* node_code, const unsigned int num_leaves, const unsigned int first, const unsigned int last) noexcept {
    const codeT first_code = node_code[first];
    const codeT last_code  = node_code[last];
    if (first_code == last_code) {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last) {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node) {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}

template <class codeT>
__global__ void construct_internal_nodes(int num_objects, const codeT *morton, Node *nodes) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    int num_internal_nodes = num_objects - 1;
    if (idx < num_internal_nodes) {
        const uint2 ij  = determine_range(morton, num_objects, idx);
        const int gamma = find_split(morton, num_objects, ij.x, ij.y);

        nodes[idx].left_idx  = gamma;
        nodes[idx].right_idx = gamma + 1;
        if(min(ij.x, ij.y) == gamma) {
            nodes[idx].left_idx += num_objects - 1;
        }
        if(max(ij.x, ij.y) == gamma + 1) {
            nodes[idx].right_idx += num_objects - 1;
        }
        nodes[nodes[idx].left_idx].parent_idx  = idx;
        nodes[nodes[idx].right_idx].parent_idx = idx;
    }
}

template<class T>
__global__ void compute_internal_aabbs(int num_objects, const Node *nodes, aabb<T> *aabbs, int *flags) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    int num_internal_nodes = num_objects - 1;
    if (idx < num_objects) {
        idx += num_internal_nodes;
        unsigned int parent = nodes[idx].parent_idx;
        while(parent != 0xFFFFFFFF) { // means idx == 0
            const int old = atomicCAS(flags + parent, 0, 1);
            if(old == 0) {
                // this is the first thread entered here.
                // wait the other thread from the other child node.
                return;
            }
            assert(old == 1);
            // here, the flag has already been 1. it means that this
            // thread is the 2nd thread. merge AABB of both childlen.

            const auto lidx = nodes[parent].left_idx;
            const auto ridx = nodes[parent].right_idx;
            const auto lbox = aabbs[lidx];
            const auto rbox = aabbs[ridx];
            aabbs[parent] = merge(lbox, rbox);

            // look the next parent...
            parent = nodes[parent].parent_idx;
        }
    }
}

template<class T, int max_collision_per_node = 64, int block_size = 64> 
__global__ void query_collision_pairs(int num_targets, 
        const aabb<T> *target_aabbs, 
        const Node *bvh_nodes, 
        const aabb<T> *bvh_aabbs, 
        const int *bvh_indices, 
        int2 *collision_pairs,
        int *total_pairs,
        const unsigned int num_internal_nodes,
        const unsigned int max_collision_pairs) {
#define PAD 1
#define SHARED_ADDR(x) ((x) + ((x) >> 5) * PAD)
    __shared__ unsigned int s_in[block_size + (block_size >> 5) * PAD]; // padding to avoid bank conflict
    __shared__ unsigned int global_offset[1];
    unsigned int num_found = 0;
    unsigned int thread_pairs[max_collision_per_node];

    uint32_t target = threadIdx.x + blockDim.x * blockIdx.x;
    if (target < num_targets) {
        unsigned int stack[64]; // based on morton code bits
        unsigned int* stack_ptr = stack;
        *stack_ptr++ = 0xFFFFFFFF; // NULL node
        unsigned int node = 0;   // root node is always 0
        do {
            const unsigned int L_idx = bvh_nodes[node].left_idx;
            const unsigned int R_idx = bvh_nodes[node].right_idx;

            bool overlap_L = intersects(target_aabbs[target], bvh_aabbs[L_idx]);
            bool overlap_R = intersects(target_aabbs[target], bvh_aabbs[R_idx]);
            bool isleaf_L = (L_idx >= num_internal_nodes);
            bool isleaf_R = (R_idx >= num_internal_nodes);
            if(overlap_L && isleaf_L) {
                const auto leaf_idx = L_idx - num_internal_nodes;
                if (num_found < max_collision_per_node && 
                    target < bvh_indices[leaf_idx]) { // Count each dual-pair once & exclude self-self pair
                    thread_pairs[num_found++] = bvh_indices[leaf_idx];
                }
            }
            if(overlap_R && isleaf_R) {
                const auto leaf_idx = R_idx - num_internal_nodes;
                if (num_found < max_collision_per_node && 
                    target < bvh_indices[leaf_idx]) { // Count each dual-pair once & exclude self-self pair
                    thread_pairs[num_found++] = bvh_indices[leaf_idx];
                }
            }
            
            bool traverse_L = (overlap_L && !isleaf_L);
            bool traverse_R = (overlap_R && !isleaf_R);

            if (!traverse_L && !traverse_R)
                node = *--stack_ptr;
            else {
                node = (traverse_L) ? L_idx : R_idx;
                if (traverse_L && traverse_R)
                    *stack_ptr++ = R_idx;
            }
        } while (node != 0xFFFFFFFF);
    }

    unsigned int thid = threadIdx.x;
    unsigned int offset = 1;
    s_in[SHARED_ADDR(thid)] = num_found;

    for (unsigned int d = block_size >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;
            s_in[SHARED_ADDR(bi)] += s_in[SHARED_ADDR(ai)];
        }
        offset <<= 1;
    }

    if (thid == 0) { s_in[SHARED_ADDR(block_size - 1)] = 0; }

    for (unsigned int d = 1; d < block_size; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            unsigned int ai = offset * (2 * thid + 1) - 1;
            unsigned int bi = offset * (2 * thid + 2) - 1;
            const auto t = s_in[SHARED_ADDR(ai)];
            s_in[SHARED_ADDR(ai)] = s_in[SHARED_ADDR(bi)];
            s_in[SHARED_ADDR(bi)] += t;
        }
    }

    if (thid == 0) global_offset[0] = atomicAdd(total_pairs, s_in[SHARED_ADDR(block_size - 1)]);
    __syncthreads();

    if (num_found > 0) {
        unsigned int thread_offset = global_offset[0] + s_in[SHARED_ADDR(thid)] - num_found;
        unsigned int i = 0;
        while (thread_offset + i < max_collision_pairs && i < num_found) {
            collision_pairs[thread_offset + i] = make_int2(target, thread_pairs[i]);
            i++;
        }
    }
#undef SHARED_ADDR
#undef PAD
}

}

#endif