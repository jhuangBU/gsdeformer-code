#include "SimKernels.h"
#include "MathTools.h"
#include "cuda_rasterizer/forward.h"

namespace sim {

__global__ void initialize_covariance_cu(int num_gs, const float *scale, const float *rot, float *cov) {
    uint32_t gs = threadIdx.x + blockDim.x * blockIdx.x;
    if (gs < num_gs) {
        float S[9], R[9], L[9], LT[9];
        build_rotation(R, &rot[gs*4]);
        build_scaling(S, &scale[gs*3]);
        matmul<3, 3, 3, float>(R, S, L);
        transpose<3, 3, float>(L, LT);
        matmul<3, 3, 3, float>(L, LT, &cov[gs*9]);
    }
}

void initialize_covariance(int num_gs, const float *scale, const float *rot, float *cov, const int block_size) {
    initialize_covariance_cu << <(num_gs + block_size - 1) / block_size, block_size >> > (num_gs, scale, rot, cov);
}


__global__ void get_local_embeded_tets_cu(int num_gs, const float *pos, const float *cov, float *local_tet_x, float *local_tet_w,
    const int disable_LG_interp) {
    const float PI = 3.141592653;
    const float fac = 1.5f;
    uint32_t gs = threadIdx.x + blockDim.x * blockIdx.x;
    if (gs < num_gs) {
        float center[3] = {pos[gs*3+0], pos[gs*3+1], pos[gs*3+2]};
        if (disable_LG_interp) {
            for (int k = 0; k < 4; k++)
                for (int i = 0; i < 3; i++) {
                    local_tet_x[(gs*4+k)*3+i] = center[i];
                }
            local_tet_w[gs*3+0] = 1.f / 3.f;
            local_tet_w[gs*3+1] = 1.f / 3.f;
            local_tet_w[gs*3+2] = 1.f / 3.f;
        } else {
            float eval[3], evec[9];
            sym_eig3x3(&cov[gs*9], eval, evec);
            float sqrt_eval[3] = {sqrt(eval[0]), sqrt(eval[1]), sqrt(eval[2])};
            int mxV = eval[0] > max(eval[1], eval[2]) ? 0 : (eval[1] > max(eval[0], eval[2]) ? 1 : 2);
            int mnV = eval[0] < min(eval[1], eval[2]) ? 0 : (eval[1] < min(eval[0], eval[2]) ? 1 : 2);
            int mdV = 3 ^ (mxV ^ mnV);

            float x0[3];
            for (int i = 0; i < 3; i++) {
                local_tet_x[(gs*4+0)*3+i] = center[i] + sqrt_eval[mxV] * evec[i*3+mxV] * fac;
                x0[i] = center[i] - sqrt_eval[mxV] * evec[i*3+mxV];
            }
            float n[3] = {evec[0*3+mxV], evec[1*3+mxV], evec[2*3+mxV]};
            normalize<3, float>(n);
            float nT[3] = {evec[0*3+mdV], evec[1*3+mdV], evec[2*3+mdV]};
            normalize<3, float>(nT);
            float u[3];
            cross_product3<float>(n, nT, u);
            float v[3];
            cross_product3<float>(n, u, v);
            float r = sqrt_eval[mdV];

            for (int i = 0; i < 3; i++) {
                local_tet_x[(gs*4+1)*3+i] = x0[i] + r * (cosf(0) * u[i] + sinf(0) * v[i]) * fac;
                local_tet_x[(gs*4+2)*3+i] = x0[i] + r * (cosf(+2.f / 3.f * PI) * u[i] + sinf(+2.f / 3.f * PI) * v[i]) * fac;
                local_tet_x[(gs*4+3)*3+i] = x0[i] + r * (cosf(-2.f / 3.f * PI) * u[i] + sinf(-2.f / 3.f * PI) * v[i]) * fac;
            }

            float Ds_local[9] = {
                local_tet_x[(gs*4+0)*3+0] - local_tet_x[(gs*4+3)*3+0], local_tet_x[(gs*4+1)*3+0] - local_tet_x[(gs*4+3)*3+0], local_tet_x[(gs*4+2)*3+0] - local_tet_x[(gs*4+3)*3+0],
                local_tet_x[(gs*4+0)*3+1] - local_tet_x[(gs*4+3)*3+1], local_tet_x[(gs*4+1)*3+1] - local_tet_x[(gs*4+3)*3+1], local_tet_x[(gs*4+2)*3+1] - local_tet_x[(gs*4+3)*3+1],
                local_tet_x[(gs*4+0)*3+2] - local_tet_x[(gs*4+3)*3+2], local_tet_x[(gs*4+1)*3+2] - local_tet_x[(gs*4+3)*3+2], local_tet_x[(gs*4+2)*3+2] - local_tet_x[(gs*4+3)*3+2]
            };
            float Ds_local_inv[9];
            inverse3<float>(Ds_local, Ds_local_inv);
            float dxyz[3] = {center[0] - local_tet_x[(gs*4+3)*3+0], center[1] - local_tet_x[(gs*4+3)*3+1], center[2] - local_tet_x[(gs*4+3)*3+2]};
            matmul<3, 3, 1, float>(Ds_local_inv, dxyz, &local_tet_w[gs*3]);
        }
    }
}

void get_local_embeded_tets(int num_gs, const float *pos, const float *cov, 
    float *local_tet_x, float *local_tet_w, 
    const int disable_LG_interp, 
    const int block_size) {
    get_local_embeded_tets_cu << <(num_gs + block_size - 1) / block_size, block_size >> > (num_gs, pos, cov, local_tet_x, local_tet_w, disable_LG_interp);
}


__global__ void get_global_embeded_tet_cu(int num_gs, int gs_offset, const int t_id, const float *verts_X, const int *tets, const float *local_tet_x, int *global_tet_idx, float *global_tet_w) {
    uint32_t _ = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t local_gs = _ / 4;
    uint32_t c  = _ % 4;
    if (local_gs < num_gs) {
        int gs = local_gs + gs_offset;
        float Ds[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ds[j*3+i] = verts_X[tets[t_id*4+i]*3+j] - verts_X[tets[t_id*4+3]*3+j];
            }
        }
        float Ds_inv[9];
        inverse3<float>(Ds, Ds_inv);

        float dxyz[3] = {local_tet_x[(gs*4+c)*3+0] - verts_X[tets[t_id*4+3]*3+0],
                         local_tet_x[(gs*4+c)*3+1] - verts_X[tets[t_id*4+3]*3+1],
                         local_tet_x[(gs*4+c)*3+2] - verts_X[tets[t_id*4+3]*3+2]};
        float w[3];
        matmul<3, 3, 1, float>(Ds_inv, dxyz, w);
        if (
            w[0] >= 0. &&
            w[0] <= 1. &&
            w[1] >= 0. &&
            w[1] <= 1. &&
            w[2] >= 0. &&
            w[2] <= 1. &&
            w[0] + w[1] + w[2] <= 1.
        ) {
            global_tet_idx[gs*4+c] = t_id;
            global_tet_w[(gs*4+c)*3+0] = w[0];
            global_tet_w[(gs*4+c)*3+1] = w[1];
            global_tet_w[(gs*4+c)*3+2] = w[2];
        }
    }
}

void get_global_embeded_tet(int num_gs, int gs_offset, const int t_id, 
    const float *verts_X, const int *tets, const float *local_tet_x, 
    int *global_tet_idx, float *global_tet_w, const int block_size) {
    get_global_embeded_tet_cu << <((num_gs * 4) + block_size - 1) / block_size, block_size >> > (num_gs, gs_offset, t_id, verts_X, tets, local_tet_x, global_tet_idx, global_tet_w);
}

__device__ bool is_valid_embed(int idx, const int *global_tet_idx) {
    return global_tet_idx[idx*4+0] >= 0 &&
           global_tet_idx[idx*4+1] >= 0 &&
           global_tet_idx[idx*4+2] >= 0 &&
           global_tet_idx[idx*4+3] >= 0;
}

__global__ void deactivate_opacity_cu(int num_gs, int gs_offset, float *opacity, const int *global_tet_idx) {
    uint32_t local_gs = threadIdx.x + blockDim.x * blockIdx.x;
    if (local_gs < num_gs) {
        int gs = local_gs + gs_offset;
        if (!is_valid_embed(gs, global_tet_idx)) {
            opacity[gs] = 0.0;
        }
    }
}

void deactivate_opacity(int num_gs, int gs_offset, float *opacity, const int *global_tet_idx, const int block_size) {
    deactivate_opacity_cu << <(num_gs + block_size - 1) / block_size, block_size >> > (num_gs, gs_offset, opacity, global_tet_idx);
}

__global__ void apply_interpolation_cu(int num_gs,
                         float *pos, float *scale, float *rot, 
                         const float *cov, const int *tets, const float *verts_X, const float *verts_x,
                         const int *global_tet_idx, const float *global_tet_w, const float *local_tet_w,
                         const int disable_LG_interp, float *gs_local_rot) {
    uint32_t gs = threadIdx.x + blockDim.x * blockIdx.x;
    if (gs < num_gs) {
        if (is_valid_embed(gs, global_tet_idx)) {
            float F[9];
            if (disable_LG_interp) {
                float w[3] = {global_tet_w[(gs*4)*3+0], global_tet_w[(gs*4)*3+1], global_tet_w[(gs*4)*3+2]};
                int t_id = global_tet_idx[gs*4];
                float DS[9];
                float Ds[9];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        DS[j*3+i] = verts_X[tets[t_id*4+i]*3+j] - verts_X[tets[t_id*4+3]*3+j];
                        Ds[j*3+i] = verts_x[tets[t_id*4+i]*3+j] - verts_x[tets[t_id*4+3]*3+j];
                    }
                }
                float Ds_w[3];
                matmul<3, 3, 1, float>(Ds, w, Ds_w);
                for (int i = 0; i < 3; i++) {
                    pos[gs*3+i] = Ds_w[i] + verts_x[tets[t_id*4+3]*3+i];
                }
                float DS_inv[9];
                inverse3<float>(DS, DS_inv);
                matmul<3, 3, 3, float>(Ds, DS_inv, F);
            } else {
                float local_X[12], local_x[12];
                for (int c = 0; c < 4; c++) {
                    float w[3] = {global_tet_w[(gs*4+c)*3+0], global_tet_w[(gs*4+c)*3+1], global_tet_w[(gs*4+c)*3+2]};
                    int t_id = global_tet_idx[gs*4+c];
                    float global_DS[9];
                    float global_Ds[9];
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            global_DS[j*3+i] = verts_X[tets[t_id*4+i]*3+j] - verts_X[tets[t_id*4+3]*3+j];
                            global_Ds[j*3+i] = verts_x[tets[t_id*4+i]*3+j] - verts_x[tets[t_id*4+3]*3+j];
                        }
                    }
                    float g_DS_w[3], g_Ds_w[3];
                    matmul<3, 3, 1, float>(global_DS, w, g_DS_w);
                    matmul<3, 3, 1, float>(global_Ds, w, g_Ds_w);
                    for (int i = 0; i < 3; i++) {
                        local_X[c*3+i] = g_DS_w[i] + verts_X[tets[t_id*4+3]*3+i];
                        local_x[c*3+i] = g_Ds_w[i] + verts_x[tets[t_id*4+3]*3+i];
                    }
                }
                float local_DS[9], local_Ds[9];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        local_DS[j*3+i] = local_X[i*3+j]-local_X[3*3+j];
                        local_Ds[j*3+i] = local_x[i*3+j]-local_x[3*3+j];
                    }
                }
                float l_Ds_w[3], l_w[3] = {local_tet_w[gs*3+0], local_tet_w[gs*3+1], local_tet_w[gs*3+2]};
                matmul<3, 3, 1, float>(local_Ds, l_w, l_Ds_w);
                for (int i = 0; i < 3; i++) {
                    pos[gs*3+i] = l_Ds_w[i] + local_x[3*3+i];
                }

                float local_DS_inv[9];
                inverse3<float>(local_DS, local_DS_inv);
                matmul<3, 3, 3, float>(local_Ds, local_DS_inv, F);
            }

            float Ft[9];
            transpose<3, 3, float>(F, Ft);
            float tmpB[9], B[9];
            matmul<3, 3, 3, float>(F, &cov[gs*9], tmpB);
            matmul<3, 3, 3, float>(tmpB, Ft, B);
            float U[9], sigma[9], V[9];
            ssvd3x3<float>(B, U, sigma, V);
            
            // Transform to Rotation & Scaling
            build_quaternion(&rot[gs*4], U);
            scale[gs*3+0] = sqrt(max(sigma[0*3+0], 1e-8));
            scale[gs*3+1] = sqrt(max(sigma[1*3+1], 1e-8));
            scale[gs*3+2] = sqrt(max(sigma[2*3+2], 1e-8));

            matmul<3, 3, 3, float>(F, Ft, B);
            ssvd3x3<float>(B, U, sigma, V);
            float Vt[9];
            transpose<3, 3, float>(V, Vt);
            matmul<3, 3, 3, float>(U, Vt, B);
            float Bt[9];
            transpose<3, 3, float>(B, Bt);
            for (int i = 0; i < 9; i++) {
                gs_local_rot[gs * 9 + i] = Bt[i];
            }
        }
    }
}

void apply_interpolation(int num_gs, float *pos, float *scale, 
    float *rot, const float *cov, const int *tets, 
    const float *verts_X, const float *verts_x, const int *global_tet_idx, 
    const float *global_tet_w, const float *local_tet_w, 
    const int disable_LG_interp, float *gs_local_rot,
    const int block_size) {
    apply_interpolation_cu << <(num_gs + block_size - 1) / block_size, block_size >> > (
        num_gs, 
        pos, 
        scale, 
        rot, 
        cov, 
        tets, 
        verts_X, 
        verts_x, 
        global_tet_idx, 
        global_tet_w, 
        local_tet_w,
        disable_LG_interp,
        gs_local_rot);
}


__global__ void init_inv_mass_cu(int num_verts, const float *verts_X, const float *verts_m, float *verts_inv_m, int boundary) {
    uint32_t v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v < num_verts) {
        // Benchmark Chair use this
        // if (verts_m[v] == 0.f || verts_X[v * 3 + 2] >= 0.8026f) {
        // Fox Demo use this
        // if (verts_m[v] == 0.f || verts_X[v * 3 + 2] >= 1.3f) {
        // Bear Demo or Horse use this
        // if (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= (0.79f - 1.5f)) {
        // Basket Demo use this
        // if (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= -0.75f) {
        // Sofa Basket Demo use this
        // if (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= -0.1f) {
        // Toy Collection Demo use this
        // if (verts_m[v] == 0.f) {
        // JLS use this
        // if (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= 0.3f) {
        // Dance use this
        if (boundary == 0 && verts_m[v] == 0.f) {
            verts_inv_m[v] = 0.f;
        } else if (boundary == 1 && (verts_m[v] == 0.f || verts_X[v * 3 + 2] >= 1.3f)) { // Fox
            verts_inv_m[v] = 0.f;
        } else if (boundary == 2 && (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= (0.79f - 1.5f))) { // Bear
            verts_inv_m[v] = 0.f;
        } else if (boundary == 3 && (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= 0.3f)) { // JLS
            verts_inv_m[v] = 0.f;
        } else if (boundary == 4 && (verts_m[v] == 0.f || verts_X[v * 3 + 1] >= -0.1f)) { // revised basket
            verts_inv_m[v] = 0.f;
        } else if (boundary == 10 && (verts_m[v] == 0.f || verts_X[v * 3 + 2] <= -0.5f)) { // ficus
            verts_inv_m[v] = 0.f;
        } else if (boundary == 20 && (verts_m[v] == 0.f || verts_X[v * 3 + 0] <= 0.3f)) { // microphone
            verts_inv_m[v] = 0.f;
        } else if (boundary == 30 && (verts_m[v] == 0.f || verts_X[v * 3 + 2] >= 0.8026f)) { // chair
            verts_inv_m[v] = 0.f;
        }
        else {
            verts_inv_m[v] = 1.f / verts_m[v];
        }
    }
}

void init_inv_mass(int num_verts, const float *verts_X, const float *verts_m, float *verts_inv_m, int boundary, const int block_size) {
    init_inv_mass_cu << <(num_verts + block_size - 1) / block_size, block_size >> > (num_verts, verts_X, verts_m, verts_inv_m, boundary);
}


__global__ void init_FEM_bases_cu(int num_cells, const float *cells_density, const int *tets, const float *verts_X, float *verts_m, 
                    float *cells_DS_inv, float *cells_vol0) {
    uint32_t t_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (t_id < num_cells) {
        float DS[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                DS[j*3+i] = verts_X[tets[t_id*4+i]*3+j] - verts_X[tets[t_id*4+3]*3+j];
            }
        }
        inverse3<float>(DS, &cells_DS_inv[t_id*9]);
        float volume = abs(determinant3<float>(DS) / 6.0);
        cells_vol0[t_id] = volume;
        for (int i = 0; i < 4; ++i) {
            atomicAdd(verts_m + tets[t_id*4+i], 0.25 * cells_density[t_id] * volume);
        }
    }
}

void init_FEM_bases(int num_cells, const float *cells_density, const int *tets, const float *verts_X, float *verts_m, 
                    float *cells_DS_inv, float *cells_vol0, const int block_size) {
    init_FEM_bases_cu << <(num_cells + block_size - 1) / block_size, block_size >> > (num_cells, cells_density, tets, verts_X, verts_m, cells_DS_inv, cells_vol0);
}

__global__ void select_vertices_cu(int num_verts, const float controller_radius, const float *cpos, const int bit, 
                     const float *verts_x, int *verts_select) {
    uint32_t v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v < num_verts) {
        if (distance<3, float>(cpos, &verts_x[v*3]) <= controller_radius) {
            verts_select[v] |= bit;
        }
    }           
}

void select_vertices(int num_verts, const float controller_radius, const float *cpos, const int bit, 
                     const float *verts_x, int *verts_select, const int block_size) {
    select_vertices_cu << <(num_verts + block_size - 1) / block_size, block_size >> > (num_verts, controller_radius, cpos, bit, verts_x, verts_select);
}


__global__ void apply_external_force_cu(int num_verts, const float dt, const float gravity, const float damping_coeffient,
                          const float *verts_inv_m, const float *verts_x, float *verts_v, float *verts_new_x,
                          const float *lpos, const float *rpos, 
                          const float *lvel, const float *rvel,
                          const float *lrot, const float *rrot,
                          const int *verts_select, const int zup) {
    uint32_t v = threadIdx.x + blockDim.x * blockIdx.x;

    if (v < num_verts) {
        if (verts_inv_m[v] > 0) { // Gravity Force
            for (int i = 0; i < 3; i++) {
                verts_v[v*3+i] *= expf(-dt * damping_coeffient);
            }

            constexpr int LEFT_BIT = 1;
            constexpr int RIGHT_BIT = 2;
            // Normally 0.05f;
            // Sometimes 0.005f
            const float rot_factor = 0.05f; // NOTE(changyu): for smooth manipulation
            // Cannot select fixed verts
            if (!(verts_select[v] & (LEFT_BIT | RIGHT_BIT))) {
                verts_v[v*3+(zup?2:1)] += (zup ? 1.0 : -1.0) * gravity * dt;
            } else {
                for (int i = 0; i < 3; i++) {
                    verts_v[v*3+i] = 0.f;
                }
            }

            // Left controller
            if (verts_select[v] & LEFT_BIT) { // Controller Force
                float dpos[3] = {
                    verts_x[v*3+0] - lpos[0], 
                    verts_x[v*3+1] - lpos[1], 
                    verts_x[v*3+2] - lpos[2], 
                };
                float new_dpos[3];
                matmul<3, 3, 1>(lrot, dpos, new_dpos);
                for (int i = 0; i < 3; i++) {
                    verts_v[v*3+i] += (new_dpos[i] - dpos[i]) / dt * rot_factor + lvel[i];
                }
            }

            // Right controller
            if (verts_select[v] & RIGHT_BIT) { // Controller Force
                float dpos[3] = {
                    verts_x[v*3+0] - rpos[0], 
                    verts_x[v*3+1] - rpos[1], 
                    verts_x[v*3+2] - rpos[2], 
                };
                float new_dpos[3];
                matmul<3, 3, 1>(rrot, dpos, new_dpos);
                for (int i = 0; i < 3; i++) {
                    verts_v[v*3+i] += (new_dpos[i] - dpos[i]) / dt * rot_factor + rvel[i];
                }
            }
        } else {
            verts_v[v*3+0] = 0;
            verts_v[v*3+1] = 0;
            verts_v[v*3+2] = 0;
        }

        for (int i = 0; i < 3; i++) {
            verts_new_x[v*3+i] = verts_x[v*3+i] + verts_v[v*3+i] * dt;
        }
    }
}

void apply_external_force(int num_verts, const float dt, const float gravity, const float damping_coeffient,
                          const float *verts_inv_m, const float *verts_x, float *verts_v, 
                          float *verts_new_x, 
                          const float *lpos, const float *rpos, 
                          const float *lvel, const float *rvel,
                          const float *lrot, const float *rrot,
                          const int *verts_select, const int zup,
                          const int block_size) {
    apply_external_force_cu << <(num_verts + block_size - 1) / block_size, block_size >> > (num_verts, dt, gravity, damping_coeffient,
        verts_inv_m, verts_x, verts_v, verts_new_x,
        lpos, rpos, 
        lvel, rvel,
        lrot, rrot,
        verts_select, zup);
}


__global__ void pbd_post_solve_cu(int num_verts, const float *verts_dp, float *verts_new_x,
                                  const int *verts_select) {
    uint32_t v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v < num_verts) {
        if (verts_select[v]) {
            // No Update
        } else {
            for (int i = 0; i < 3; i++) {
                verts_new_x[v*3+i] += verts_dp[v*3+i];
            }
        }
    }
}

void pbd_post_solve(int num_verts, const float *verts_dp, float *verts_new_x,
                    const int *verts_select, const int block_size) {
    pbd_post_solve_cu << <(num_verts + block_size - 1) / block_size, block_size >> > (num_verts, verts_dp, verts_new_x, verts_select);
}


__global__ void pbd_advance_cu(int num_verts, const float dt, const float *verts_inv_m, 
    float *verts_v, float *verts_x, float *verts_new_x, 
    const int zup, const float ground_height) {
    uint32_t v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v < num_verts) {
        if (verts_inv_m[v] <= 0.0) {
            for (int i = 0; i < 3; i++) {
                verts_new_x[v*3+i] = verts_x[v*3+i];
            }
        } else {
            // Ground Collision
            if (zup) {
                if (verts_new_x[v*3+2] < ground_height) {
                    verts_new_x[v*3+2] = ground_height;
                }
            }  else {
                if (verts_new_x[v*3+1] > -ground_height) {
                    verts_new_x[v*3+1] = -ground_height;
                }
            }
            for (int i = 0; i < 3; i++) {
                verts_v[v*3+i] = (verts_new_x[v*3+i] - verts_x[v*3+i]) / dt;
            }
            for (int i = 0; i < 3; i++) {
                verts_x[v*3+i] = verts_new_x[v*3+i];
            }
        }
    }
}

void pbd_advance(int num_verts, const float dt, const float *verts_inv_m, 
    float *verts_v, float *verts_x, float *verts_new_x, 
    const int zup, const float ground_height, const int block_size) {
    pbd_advance_cu << <(num_verts + block_size - 1) / block_size, block_size >> > (num_verts, dt, verts_inv_m, verts_v, verts_x, verts_new_x, zup, ground_height);
}


__global__ void solve_FEM_constraints_cu(int num_cells, 
    const float dt, const float *cells_mu, const float *cells_lambda, const int *tets,
    const float *verts_new_x, const float *verts_inv_m, const float *cells_DS_inv, const float *cells_vol0,
    float *verts_dp, float *cells_multiplier, const int *rigid_group) {
    uint32_t t_id = threadIdx.x + blockDim.x * blockIdx.x;
    const float eps = 1e-6;
    if (t_id < num_cells && rigid_group[tets[t_id*4]] < 0) {
        float Ds[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                Ds[j*3+i] = verts_new_x[tets[t_id*4+i]*3+j] - verts_new_x[tets[t_id*4+3]*3+j];
            }
        }
        float F[9];
        matmul<3, 3, 3, float>(Ds, &cells_DS_inv[t_id*9], F);

        float E[9], trE = 0.0, sigma[9];
        // Green strain tensor
#define INVERSION_HANDLING
#ifndef INVERSION_HANDLING
        {
            float Ft[9];
            transpose<3, 3, float>(F, Ft);
            matmul<3, 3, 3, float>(Ft, F, E);
            for (int i = 0; i < 9; i++) {
                E[i] *= 0.5;
            }
            E[0*3+0] -= 0.5f;
            E[1*3+1] -= 0.5f;
            E[2*3+2] -= 0.5f;

            trE = E[0*3+0] + E[1*3+1] + E[2*3+2];
            float tmp[9];
            for (int i = 0; i < 9; i++) {
                tmp[i] = 2.0 * mu * E[i];
            }
            tmp[0*3+0] += cells_lambda[t_id] * trE;
            tmp[1*3+1] += cells_lambda[t_id] * trE;
            tmp[2*3+2] += cells_lambda[t_id] * trE;
            matmul<3, 3, 3, float>(F, tmp, sigma);
        }
#else
        {
            float U[9], sig[9], V[9];
            ssvd3x3<float>(F, U, sig, V);
            // Clamp small singular values
	        const float min_X_Val = 0.05f;
            const float max_X_Val = 1.7f;
            for (int i = 0; i < 3; i++) {
                sig[i*3+i] = min(max(min_X_Val, sig[i*3+i]), max_X_Val);
            }

            float E_diag[3] = {
                0.5f * (sig[0*3+0] * sig[0*3+0] - 1.0f),
                0.5f * (sig[1*3+1] * sig[1*3+1] - 1.0f),
                0.5f * (sig[2*3+2] * sig[2*3+2] - 1.0f)
            };
            trE = E_diag[0] + E_diag[1] + E_diag[2];
            sig[0*3+0] = 2.0f * cells_mu[t_id] * E_diag[0] + cells_lambda[t_id] * trE;
            sig[1*3+1] = 2.0f * cells_mu[t_id] * E_diag[1] + cells_lambda[t_id] * trE;
            sig[2*3+2] = 2.0f * cells_mu[t_id] * E_diag[2] + cells_lambda[t_id] * trE;

            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k) {
                    E[i*3+k] = 0;
                    sigma[i*3+k] = 0;
                    for (int j = 0; j < 3; ++j) {
                        E[i*3+k] += U[i*3+j] * E_diag[j] * V[k*3+j];
                        sigma[i*3+k] += U[i*3+j] * sig[j*3+j] * V[k*3+j];
                    }
            }
        }
#endif
        float psi = 0.0;
        for (int i = 0; i < 9; i++) {
            psi += E[i] * E[i];
        }
        psi = cells_mu[t_id] * psi + 0.5 * cells_lambda[t_id] * trE * trE;

        float *H = &Ds[0]; // reuse Ds since it is useless.
        for (int i = 0; i < 3; ++i)
            for (int k = 0; k < 3; ++k) {
                H[i*3+k] = 0;
                for (int j = 0; j < 3; ++j)
                    H[i*3+k] += sigma[i*3+j] * cells_DS_inv[t_id*9 + k*3+j];
        }
        for (int i = 0; i < 9; i++) H[i] *= cells_vol0[t_id];
        float sum_normGradC = 0.f;
        for (int i = 0; i < 3; i++) {
            sum_normGradC += verts_inv_m[tets[t_id*4+0]] * H[i*3+0] * H[i*3+0];
            sum_normGradC += verts_inv_m[tets[t_id*4+1]] * H[i*3+1] * H[i*3+1];
            sum_normGradC += verts_inv_m[tets[t_id*4+2]] * H[i*3+2] * H[i*3+2];
            sum_normGradC += verts_inv_m[tets[t_id*4+3]] * (H[i*3+0]+H[i*3+1]+H[i*3+2]) * (H[i*3+0]+H[i*3+1]+H[i*3+2]);
        }


        const float relaxation_factor = 0.1;
// #define XPBD
#ifdef XPBD
        // compute value U' which is the potential energy of the elastic solid U divided by Young's modulus E
	    // U = E * U'
        float U = cells_vol0[t_id] * psi;

        // By choosing the constraint function as sqrt(2 U'), the potential energy used in XPBD: 
        // U = 0.5 * alpha^-1 * C^2
        // gives us exactly the required potential energy of the elastic solid. 
        float C = sqrt(2.0 * U);

        float yE = cells_mu[t_id] * (3.0 * cells_lambda[t_id] + 2.0 * cells_mu[t_id]) / (cells_mu[t_id] + cells_lambda[t_id]); // Young's Modulus
        float alpha = 1.0 / (yE * dt * dt);
        // Note that grad C = 1/C grad U'
        sum_normGradC += C * C * alpha;
        float s = -(C * C + C * alpha * cells_multiplier[t_id]) / sum_normGradC;
        if (sum_normGradC > eps) { cells_multiplier[t_id] -= s; }
        s *= relaxation_factor;
#else
        float C = cells_vol0[t_id] * psi;
        float s = -C / sum_normGradC * relaxation_factor;
#endif
        
        if (sum_normGradC > eps) {
            for (int i = 0; i < 3; i++) {
                atomicAdd(verts_dp + tets[t_id*4+0]*3+i, s * verts_inv_m[tets[t_id*4+0]] * H[i*3+0]);
                atomicAdd(verts_dp + tets[t_id*4+1]*3+i, s * verts_inv_m[tets[t_id*4+1]] * H[i*3+1]);
                atomicAdd(verts_dp + tets[t_id*4+2]*3+i, s * verts_inv_m[tets[t_id*4+2]] * H[i*3+2]);
                atomicAdd(verts_dp + tets[t_id*4+3]*3+i, s * verts_inv_m[tets[t_id*4+3]] * (-H[i*3+0] - H[i*3+1] - H[i*3+2]));
            }
        }
    }
}

void solve_FEM_constraints(int num_cells, 
    const float dt, const float *cells_mu, const float *cells_lambda, const int *tets,
    const float *verts_new_x, const float *verts_inv_m, const float *cells_DS_inv, const float *cells_vol0,
    float *verts_dp, float *cells_multiplier, const int *rigid_group, const int block_size) {
    solve_FEM_constraints_cu << <(num_cells + block_size - 1) / block_size, block_size >> > (num_cells, dt, cells_mu, cells_lambda, tets,
        verts_new_x, verts_inv_m, cells_DS_inv, cells_vol0, verts_dp, cells_multiplier, rigid_group);
}

// https://github.com/InteractiveComputerGraphics/PositionBasedDynamics/blob/00db2e091a88a628099787cf6fada941e851dbfa/PositionBasedDynamics/PositionBasedDynamics.cpp#L291C11-L291C11
__global__ void solve_triangle_point_distance_constraint_cu(int num_pairs, 
    const float minimal_dist, const float collision_stiffness, const int4 *collision_pairs,
	const float *verts_new_x, const float *verts_inv_m, float *verts_dp) {
    uint32_t t_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (t_id < num_pairs) {
        int v  = collision_pairs[t_id].x;
        int v0 = collision_pairs[t_id].y;
        int v1 = collision_pairs[t_id].z;
        int v2 = collision_pairs[t_id].w;

        float p[3] = {verts_new_x[v*3+0], verts_new_x[v*3+1], verts_new_x[v*3+2]};
        float p0[3] = {verts_new_x[v0*3+0], verts_new_x[v0*3+1], verts_new_x[v0*3+2]};
        float p1[3] = {verts_new_x[v1*3+0], verts_new_x[v1*3+1], verts_new_x[v1*3+2]};
        float p2[3] = {verts_new_x[v2*3+0], verts_new_x[v2*3+1], verts_new_x[v2*3+2]};

        // find barycentric coordinates of closest point on triangle
        float b0 = 1.0f / 3.0f;		// for singular case
        float b1 = b0;
        float b2 = b0;

        float d1[3]  = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
        float d2[3]  = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        float pp0[3] = {p [0] - p0[0], p [1] - p0[1], p [2] - p0[2]};
        float a = dot<3, float>(d1, d1);
        float b = dot<3, float>(d2, d1);
        float c = dot<3, float>(pp0, d1);
        float d = b;
        float e = dot<3, float>(d2, d2);
        float f = dot<3, float>(pp0, d2);
        float det = a*e - b*d;

        if (det != 0.0) {
            float s = (c*e - b*f) / det;
            float t = (a*f - c*d) / det;
            b0 = 1.0f - s - t;		// inside triangle
            b1 = s;
            b2 = t;
            if (b0 < 0.0) {		// on edge 1-2
                float d[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
                float p_p1[3] = {p[0] - p1[0], p[1] - p1[1], p[2] - p1[2]};
                float d2 = dot<3, float>(d, d);
                float t = (d2 == 0.0f) ? 0.5f : dot<3, float>(d, p_p1) / d2;
                if (t < 0.0) t = 0.0;	// on point 1
                if (t > 1.0) t = 1.0;	// on point 2
                b0 = 0.0;
                b1 = (1.0f - t);
                b2 = t;
            }
            else if (b1 < 0.0) {	// on edge 2-0
                float d[3] = {p0[0] - p2[0], p0[1] - p2[1], p0[2] - p2[2]};
                float p_p2[3] = {p[0] - p2[0], p[1] - p2[1], p[2] - p2[2]};
                float d2 = dot<3, float>(d, d);
                float t = (d2 == 0.0f) ? 0.5f : dot<3, float>(d, p_p2) / d2;
                if (t < 0.0) t = 0.0;	// on point 2
                if (t > 1.0) t = 1.0; // on point 0
                b1 = 0.0;
                b2 = (1.0f - t);
                b0 = t;
            }
            else if (b2 < 0.0) {	// on edge 0-1
                float d[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
                float p_p0[3] = {p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
                float d2 = dot<3, float>(d, d);
                float t = (d2 == 0.0f) ? 0.5f : dot<3, float>(d, p_p0) / d2;
                if (t < 0.0) t = 0.0;	// on point 0
                if (t > 1.0) t = 1.0;	// on point 1
                b2 = 0.0;
                b0 = (1.0f - t);
                b1 = t;
            }
        }

        float q[3], n[3], fn[3];
        cross_product3<float>(d1, d2, fn);

        for (int i = 0; i < 3; i++) {
            q[i] = p0[i] * b0 + p1[i] * b1 + p2[i] * b2;
            n[i] = p[i] - q[i];
        }
        float dist = norm<3, float>(n);
        normalize<3, float>(n);
        /*if (dot<3, float>(fn, pp0) < 0) {
            dist *= -1;
            n[0] *= -1;
            n[1] *= -1;
            n[2] *= -1;
        }*/
        float C = dist - minimal_dist;
        
        if (C < 0.0) {
            float s = verts_inv_m[v] + verts_inv_m[v0] * b0*b0 + verts_inv_m[v1] * b1*b1 + verts_inv_m[v2] * b2*b2;
            const float relaxation_factor = 1.0;

            if (s > 0.0) {
                s = C / s * collision_stiffness * relaxation_factor;

                for (int i = 0; i < 3; i++) {
                    atomicAdd(verts_dp + v *3 + i, -s * verts_inv_m[v]  * n[i]); // grad
                    atomicAdd(verts_dp + v0*3 + i, -s * verts_inv_m[v0] * (-n[i] * b0)); // grad0
                    atomicAdd(verts_dp + v1*3 + i, -s * verts_inv_m[v1] * (-n[i] * b1)); // grad1
                    atomicAdd(verts_dp + v2*3 + i, -s * verts_inv_m[v2] * (-n[i] * b2)); // grad2
                }
            }
        }
    }
}

void solve_triangle_point_distance_constraint(int num_pairs, 
    const float minimal_dist, const float collision_stiffness, const int4 *collision_pairs,
	const float *verts_new_x, const float *verts_inv_m, float *verts_dp, const int block_size) {
    if (num_pairs == 0) {
        return ;
    }

    solve_triangle_point_distance_constraint_cu << <(num_pairs + block_size - 1) / block_size, block_size >> > (
        num_pairs, minimal_dist, collision_stiffness, 
        collision_pairs, verts_new_x, verts_inv_m, verts_dp);
}

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ void eval_sh(float *final, const float* shs, const glm::vec3 dir)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
    const int deg = 3;
	glm::vec3* sh = ((glm::vec3*)shs);
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	final[0] = glm::max(result.x, 0.0f);
    final[1] = glm::max(result.y, 0.0f);
    final[2] = glm::max(result.z, 0.0f);
}

__global__ void convert_SH_cu(const int num_gs, const float *viewpoint_cam, const float *shs,
    const float *pos, const float *local_rot, const float *init_rot,
    float *pre_color) {
    uint32_t gs = threadIdx.x + blockDim.x * blockIdx.x;
    if (gs < num_gs) {
        float dir_pp[3], new_dir_pp[3];
        for (int i = 0; i < 3; i++) {
            dir_pp[i] = pos[gs * 3 + i] - viewpoint_cam[i];
        }
        float overall_rot[9];
        matmul<3, 3, 3, float>(&local_rot[gs*9], &init_rot[gs*9], overall_rot);
        if (abs(determinant3<float>(overall_rot)) > 1e-5) {
            matmul<3, 3, 1, float>(overall_rot, dir_pp, new_dir_pp);
        } else {
            for (int i = 0; i < 3; i++) new_dir_pp[i] = dir_pp[i];
        }
        
        normalize<3, float>(new_dir_pp);
        int shs_size = (3 + 1) * (3 + 1) * 3;
        glm::vec3 dir;
        dir.x = new_dir_pp[0];
        dir.y = new_dir_pp[1];
        dir.z = new_dir_pp[2];
        eval_sh(&pre_color[gs*3], &shs[gs * shs_size], dir);
    }
}

void convert_SH(const int num_gs, const float *viewpoint_cam, const float *shs,
    const float *pos, const float *local_rot, const float *init_rot,
    float *pre_color) {
    int block_size = 256;
    convert_SH_cu << <(num_gs + block_size - 1) / block_size, block_size >> > (
        num_gs, viewpoint_cam, shs, pos, local_rot, init_rot, pre_color);
}

__global__ void enforce_quasi_boundary_condition_cu(const int num_face, 
    float *pos, const int *face,
    const float *quasi_pos, const int *quasi_face,
    int *verts_select) {
    uint32_t f = threadIdx.x + blockDim.x * blockIdx.x;
    if (f < num_face) {
        if (face[f] == quasi_face[f]) {
            int old_value = atomicAdd(&verts_select[face[f]], 1);
            if (old_value == 0) {
                for (int i = 0; i < 3; i++) {
                    pos[face[f]*3+i] = quasi_pos[quasi_face[f]*3+i];
                }
            }
        }
    }
}

void enforce_quasi_boundary_condition(const int num_face, 
    float *pos, const int *face,
    const float *quasi_pos, const int *quasi_face,
    int *verts_select) {
    int block_size = 256;
    enforce_quasi_boundary_condition_cu << < (num_face + block_size - 1) / block_size, block_size >> > (
        num_face, pos, face, quasi_pos, quasi_face, verts_select);
}

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__global__ void init_rigid_cu(const int num_verts, double *rigid_m, double *rigid_cm0, const int *rigid_group, const float *verts_X, const float *verts_m) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_verts && rigid_group[idx] >= 0) {
        int gr = rigid_group[idx];
        for (int i = 0; i < 3; i++) {
            atomicAdd(rigid_cm0 + gr*3+i, double(verts_m[idx] * verts_X[idx*3+i]));
        }
        atomicAdd(rigid_m + gr, double(verts_m[idx]));
    }
}

void init_rigid(const int num_verts, double *rigid_m, double *rigid_cm0, const int *rigid_group, const float *verts_X, const float *verts_m) {
    int block_size = 256;
    init_rigid_cu  << < (num_verts + block_size - 1) / block_size, block_size >> > (
        num_verts, rigid_m, rigid_cm0, rigid_group, verts_X, verts_m);
}

__global__ void solve_rigid_init_cm_cu(const int num_verts, double *rigid_cm, const int *rigid_group, const float *verts_x, const float *verts_m) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_verts && rigid_group[idx] >= 0) {
        int gr = rigid_group[idx];
        for (int i = 0; i < 3; i++) {
            atomicAdd(rigid_cm + gr*3+i, double(verts_m[idx] * verts_x[idx*3+i]));
        }
    }
}

__global__ void solve_rigid_compute_A(const int num_verts, 
    double *rigid_A, 
    const int *rigid_group,
    const double *rigid_m,
    const double *rigid_cm, 
    const double *rigid_cm0, 
    const float *verts_x, 
    const float *verts_X, 
    const float *verts_m) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_verts && rigid_group[idx] >= 0) {
        int gr = rigid_group[idx];
        double q_l[3], p_l[3], A[9];
        for (int i = 0; i < 3; i++) {
            q_l[i] = verts_X[idx*3+i] - (rigid_cm0[gr*3+i] / rigid_m[gr]);
            p_l[i] = verts_x[idx*3+i] - (rigid_cm [gr*3+i] / rigid_m[gr]);
        }
        outer_product<3, double>(p_l, q_l, A);
        for (int i = 0; i < 9; i++) {
            atomicAdd(rigid_A + gr * 9 + i, A[i] * verts_m[idx]);
        }
    }
}

__global__ void solve_rigid_compute_R (const int num_groups, double *rigid_R, const double *rigid_A) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_groups) {
        float A[9], R[9];
        for (int i = 0; i < 9; i++) A[i] = rigid_A[idx*9+i];
        float U[9], sigma[9], V[9];
        svd3x3<float>(A, U, sigma, V);
        float Vt[9];
        transpose<3, 3, float>(V, Vt);
        matmul<3, 3, 3, float>(U, Vt, R);
        for (int i = 0; i < 9; i++) rigid_R[idx*9+i] = R[i];
    }
}

__global__ void solve_rigid_update_x(const int num_verts, const int *rigid_group, const double* rigid_R, float *verts_x, const float *verts_X, const double *rigid_cm, const double *rigid_cm0, const double *rigid_m) {
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_verts && rigid_group[idx] >= 0) {
        int gr = rigid_group[idx];
        double rest_vec[3];
        for (int i = 0; i < 3; i++) {
            rest_vec[i] = verts_X[idx*3+i] - (rigid_cm0[gr*3+i] / rigid_m[gr]);
        }
        double cur_vec[3];
        matmul<3, 3, 1, double>(&rigid_R[gr*9], rest_vec, cur_vec);
        for (int i = 0; i < 3; i++) {
            float target = (rigid_cm[gr*3+i] / rigid_m[gr]) + cur_vec[i];
            verts_x[idx*3+i] += (target - verts_x[idx*3+i]) * 1.0;
        }
    }
}

void solve_rigid(const int num_verts, const int num_groups, 
            const int *rigid_group,
            const double* rigid_m, const double *rigid_cm0, double *rigid_cm,
			double *rigid_A, double *rigid_R,
			float *verts_X, float *verts_x, float *verts_m,
			const int block_size) {
    cudaMemset(rigid_cm, 0, sizeof(double) * 3 * num_groups);
    cudaMemset(rigid_A, 0, sizeof(double) * 9 * num_groups);
    cudaMemset(rigid_R, 0, sizeof(double) * 9 * num_groups);

    solve_rigid_init_cm_cu << < (num_verts + block_size - 1) / block_size, block_size >> > (num_verts, rigid_cm, rigid_group, verts_x, verts_m);
    cudaDeviceSynchronize();
    solve_rigid_compute_A << < (num_verts + block_size - 1) / block_size, block_size >> > (num_verts, rigid_A, rigid_group, rigid_m, rigid_cm, rigid_cm0, verts_x, verts_X, verts_m);
    solve_rigid_compute_R << < (num_groups + block_size - 1) / block_size, block_size >> > (num_groups, rigid_R, rigid_A);
    cudaDeviceSynchronize();
    solve_rigid_update_x << < (num_verts + block_size - 1) / block_size, block_size >> > (num_verts, rigid_group, rigid_R, verts_x, verts_X, rigid_cm, rigid_cm0, rigid_m);
}

}