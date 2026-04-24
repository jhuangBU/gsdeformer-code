#ifndef SIM_KERNELS_H
#define SIM_KERNELS_H

#include <string>
#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>

#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define SIM_API __declspec(dllexport)
    #else
        #define SIM_API __declspec(dllimport)
    #endif
#else
    #define SIM_API
#endif

namespace sim {

// Simulation Interpolation Initialization
SIM_API void initialize_covariance(int num_gs, const float *scale, const float *rot, 
    float *cov, const int block_size = 256);

SIM_API void get_local_embeded_tets(int num_gs, const float *pos, const float *cov, 
    float *local_tet_x, float *local_tet_w, 
    const int disable_LG_interp, 
    const int block_size = 256);

SIM_API void get_global_embeded_tet(int num_gs, int gs_offset, const int t_id, 
    const float *verts_X, const int *tet, const float *local_tet_x, 
    int *global_tet_idx, float *global_tet_w, const int block_size = 256);

void deactivate_opacity(int num_gs, int gs_offset, float *opacity, 
    const int *global_tet_idx, const int block_size = 256);

SIM_API void apply_interpolation(int num_gs, float *pos, float *scale, 
    float *rot, const float *cov, const int *tets, 
    const float *verts_X, const float *verts_x, const int *global_tet_idx, 
    const float *global_tet_w, const float *local_tet_w, 
    const int disable_LG_interp, float *gs_local_rot,
    const int block_size = 256);

// PBD-based Simulation
void init_FEM_bases(int num_cells, const float *cells_density, 
    const int *tets, const float *verts_X, float *verts_m, 
    float *cells_DS_inv, float *cells_vol0, const int block_size = 256);
void init_inv_mass(int num_verts, const float *verts_X, 
    const float *verts_m, float *verts_inv_m, int boundary, const int block_size = 256);

void select_vertices(int num_verts, const float controller_radius, 
    const float *cpos, const int bit, 
    const float *verts_x, int *verts_select, const int block_size = 256);

void apply_external_force(int num_verts, const float dt, const float gravity, const float damping_coeffient,
    const float *verts_inv_m, const float *verts_x, float *verts_v, float *verts_new_x,
    const float *lpos, const float *rpos, 
    const float *lvel, const float *rvel,
    const float *lrot, const float *rrot,
    const int *verts_select, const int zup, const int block_size = 256);

void pbd_post_solve(int num_verts, const float *verts_dp, float *verts_new_x,
    const int *verts_select, const int block_size = 256);

void pbd_advance(int num_verts, const float dt, const float *verts_inv_m, 
    float *verts_v, float *verts_x, float *verts_new_x,  
    const int zup, const float ground_height, const int block_size = 256);

void solve_FEM_constraints(int num_cells, 
    const float dt, const float *cells_mu, const float *cells_lambda, const int *tets,
    const float *verts_new_x, const float *verts_inv_m, const float *cells_DS_inv, const float *cells_vol0,
    float *verts_dp, float *cells_multiplier, const int *rigid_group, const int block_size = 256);

void solve_triangle_point_distance_constraint(int num_pairs, 
    const float minimal_dist, const float collision_stiffness, const int4 *collision_pairs,
	const float *verts_new_x, const float *verts_inv_m, float *verts_dp_d, const int block_size = 256);

void convert_SH(const int num_gs, const float *viewpoint_cam, const float *shs,
    const float *pos, const float *local_rot, const float *init_rot,
    float *pre_color);

void enforce_quasi_boundary_condition(const int num_face, 
    float *pos, const int *face,
    const float *quasi_pos, const int *quasi_face,
    int *verts_select);

void init_rigid(const int num_verts, double *rigid_m, double *rigid_cm0, const int *rigid_group, const float *verts_X, const float *verts_m);
void solve_rigid(const int num_verts, const int num_groups, 
            const int *rigid_group,
            const double* rigid_m, const double *rigid_cm0, double *rigid_cm,
			double *rigid_A, double *rigid_R,
			float *verts_X, float *verts_x, float *verts_m,
			const int block_size = 256);
}

#endif