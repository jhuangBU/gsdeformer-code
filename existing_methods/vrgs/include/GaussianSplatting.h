#pragma once

#include <Eigen/Eigen>
#include <fstream>
#include <chrono>
#include <iostream>
#include <ratio>
#include "lbvh/AABB.cuh"

#define FLOG(x) sim_log.open(global_path + "log.txt", std::ios::app); x; sim_log.close();
#define FDUMP(fname, x); sim_log.open(global_path + fname); x; sim_log.close();
#define FLOG_FLUSH() sim_log.open(global_path + "log.txt"); sim_log.close();

typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> Vector3f;
typedef	Eigen::Matrix<int, 3, 1, Eigen::DontAlign> Vector3i;
typedef	Eigen::Matrix<float, 4, 4, Eigen::DontAlign, 4, 4> Matrix4f;
typedef	Eigen::Matrix<float, 3, 3, Eigen::DontAlign, 3, 3> Matrix3f;

typedef Vector3f Pos;
template<int D> struct SHs { float shs[(D + 1) * (D + 1) * 3]; };
struct Scale { float scale[3]; };
struct Rot { float rot[4]; };

//Gaussian Splatting data structure
template<int D>
struct RichPoint
{
	Pos pos;
	float n[3];
	SHs<D> shs;
	float opacity;
	Scale scale;
	Rot rot;
	// float seg[3];
};

class GaussianSplatting {
public:
	~GaussianSplatting() {

#define DEL_DEVICE_PTR(ptr) if (ptr != nullptr) { cudaFree((void*)ptr); ptr = nullptr; }
#define DEL_HOST_PTR(ptr) if (ptr != nullptr) { free((void*)ptr); ptr = nullptr; }

		DEL_DEVICE_PTR(pos_d);
		DEL_DEVICE_PTR(rot_d);
		DEL_DEVICE_PTR(scale_d);
		DEL_DEVICE_PTR(opacity_d);
		DEL_DEVICE_PTR(shs_d);
		DEL_DEVICE_PTR(pre_color_d);
		DEL_DEVICE_PTR(init_rot_d);
		DEL_DEVICE_PTR(local_rot_d);
		DEL_DEVICE_PTR(rect_d);
		
		DEL_DEVICE_PTR(geomPtr);
		DEL_DEVICE_PTR(binningPtr);
		DEL_DEVICE_PTR(imgPtr);
		
		DEL_DEVICE_PTR(view_d);
		DEL_DEVICE_PTR(proj_d);
		DEL_DEVICE_PTR(proj_inv_d);
		DEL_DEVICE_PTR(cam_pos_d);
		DEL_DEVICE_PTR(background_d);

		DEL_DEVICE_PTR(edges_ind_d);
		DEL_DEVICE_PTR(faces_ind_d);
		DEL_DEVICE_PTR(cells_ind_d);

		DEL_DEVICE_PTR(verts_X_d);
		DEL_DEVICE_PTR(verts_x_d);
		DEL_DEVICE_PTR(verts_v_d);
		DEL_DEVICE_PTR(verts_f_d);
		DEL_DEVICE_PTR(verts_m_d);
		DEL_DEVICE_PTR(verts_inv_m_d);
		DEL_DEVICE_PTR(verts_new_x_d);
		DEL_DEVICE_PTR(verts_dp_d);
		DEL_DEVICE_PTR(verts_selected_d);
		DEL_DEVICE_PTR(verts_group_d);
		DEL_DEVICE_PTR(rigid_verts_group_d);
		DEL_DEVICE_PTR(cells_multiplier_d);
		DEL_DEVICE_PTR(cells_DS_inv_d);
		DEL_DEVICE_PTR(cells_vol0_d);
		DEL_DEVICE_PTR(cells_density_d);
		DEL_DEVICE_PTR(cells_mu_d);
		DEL_DEVICE_PTR(cells_lambda_d);

		DEL_DEVICE_PTR(rigid_m_d);
		DEL_DEVICE_PTR(rigid_cm0_d);
		DEL_DEVICE_PTR(rigid_cm_d);
		DEL_DEVICE_PTR(rigid_A_d);
		DEL_DEVICE_PTR(rigid_R_d);

		DEL_DEVICE_PTR(cov_d);
		DEL_DEVICE_PTR(local_tet_x);
		DEL_DEVICE_PTR(local_tet_w);
		DEL_DEVICE_PTR(global_tet_idx);
		DEL_DEVICE_PTR(global_tet_w);

		DEL_DEVICE_PTR(tri_aabbs);
		DEL_HOST_PTR(partial_aabb_h);
		DEL_DEVICE_PTR(partial_aabb_d);
		DEL_DEVICE_PTR(aabbs);
		DEL_DEVICE_PTR(morton_code);
		DEL_DEVICE_PTR(indices);
		DEL_DEVICE_PTR(sorted_morton_code);
		DEL_DEVICE_PTR(sorted_indices);
		DEL_DEVICE_PTR(sorted_tri_aabbs);
		DEL_DEVICE_PTR(nodes);
		DEL_DEVICE_PTR(flags);

		DEL_DEVICE_PTR(collision_pairs);
		DEL_DEVICE_PTR(total_pairs_d);

		DEL_DEVICE_PTR(exact_collision_pairs);
		DEL_DEVICE_PTR(total_exact_pairs_d);

		DEL_DEVICE_PTR(sort_buffer);

		DEL_DEVICE_PTR(sm_image_d);
		DEL_DEVICE_PTR(sm_depth_d);
		DEL_DEVICE_PTR(sm_alpha_d);
		DEL_DEVICE_PTR(lighting_view_d);
		DEL_DEVICE_PTR(lighting_proj_d);
		DEL_DEVICE_PTR(lighting_cam_pos_d);

		DEL_DEVICE_PTR(left.controller_pos_d);
		DEL_DEVICE_PTR(left.controller_vel_d);
		DEL_DEVICE_PTR(left.controller_ang_vel_d);
		DEL_DEVICE_PTR(right.controller_pos_d);
		DEL_DEVICE_PTR(right.controller_vel_d);
		DEL_DEVICE_PTR(right.controller_ang_vel_d);

		DEL_DEVICE_PTR(quasi_verts_d);
		DEL_DEVICE_PTR(quasi_faces_ind_d);
	}
	//TODO: create a parameters
	int _sh_degree = 3; //used when learning 3 is the default value
	bool _fastCulling = true;
	float _scalingModifier = 1.0;
	
	//The scene limit
	Vector3f _scenemin, _scenemax;
	
	//Fix data (the model for cuda)
	float* pos_d = nullptr;
	float* rot_d = nullptr;
	float* scale_d = nullptr;
	float* opacity_d = nullptr;
	float* shs_d = nullptr;
	int* rect_d = nullptr;
	float* pre_color_d = nullptr;
	float* init_rot_d = nullptr;
	float* local_rot_d = nullptr;

	size_t allocdGeom = 0, allocdBinning = 0, allocdImg = 0;
	void* geomPtr = nullptr, * binningPtr = nullptr, * imgPtr = nullptr;
	std::function<char* (size_t N)> geomBufferFunc, binningBufferFunc, imgBufferFunc;

	//Changing data (the pov)
	float* view_d = nullptr;
	float* proj_d = nullptr;
	float* proj_inv_d = nullptr; // Used for shadow map sampling
	float* cam_pos_d = nullptr;
	float* background_d = nullptr;

	// Global Tracking Data
	std::string global_path;
	std::ofstream sim_log;
	int global_step = 0;

	// Simulation Data
	float frame_dt = 7.5e-3;
	float dt = 1e-3f + 1e-5f;
	float gravity = -4.0f;
	float damping_coeffient = 5.0f;
	int rest_iter = 25;
	
	// Collision Handling Parameters
	float collision_stiffness = 0.1f;
	float minimal_dist = 5e-3f;
	int collision_dection_iter_interval = 100;

	// Cage Mesh Data
	int total_verts;
	int total_edges;
	int total_faces;
	int total_cells;
	int* edges_ind_d = nullptr;
	int* faces_ind_d = nullptr;
	int* cells_ind_d = nullptr;

	float* verts_X_d = nullptr;
	float* verts_x_d = nullptr;
	float* verts_v_d = nullptr;
	float* verts_f_d = nullptr;
	float* verts_m_d = nullptr;
	float* verts_inv_m_d = nullptr;
	float* verts_new_x_d = nullptr;
	float* verts_dp_d = nullptr;
	int*   verts_selected_d = nullptr;
	int*   verts_group_d = nullptr;
	int*   rigid_verts_group_d = nullptr;
	float* cells_multiplier_d = nullptr;
	float* cells_DS_inv_d = nullptr;
	float* cells_vol0_d = nullptr;
	float* cells_density_d = nullptr;
	float* cells_mu_d = nullptr;
	float* cells_lambda_d = nullptr;

	double *rigid_m_d = nullptr;
	double *rigid_cm0_d = nullptr;
	double *rigid_cm_d = nullptr;
	double *rigid_A_d = nullptr;
	double *rigid_R_d = nullptr;

	// Interpolation Data
	int total_gs;
	float* cov_d = nullptr; // [num_gs] -> 3x3
	float* local_tet_x = nullptr; // [num_gs, 4] -> 3
	float* local_tet_w = nullptr; // [num_gs] -> 3
	int* global_tet_idx = nullptr; // [num_gs, 4] -> 1
	float* global_tet_w = nullptr; // [num_gs, 4] -> 3

	// Collision Data
	// LBVH Data
	lbvh::aabb<float> aabb_global;
	lbvh::aabb<float> *tri_aabbs = nullptr;
	lbvh::aabb<float> *partial_aabb_h = nullptr;
    lbvh::aabb<float> *partial_aabb_d = nullptr;
    lbvh::aabb<float> *aabbs = nullptr;
    uint64_t *morton_code = nullptr;
    int *indices = nullptr;
    uint64_t *sorted_morton_code = nullptr;
    int *sorted_indices = nullptr;
    lbvh::aabb<float> *sorted_tri_aabbs = nullptr;
    lbvh::Node *nodes = nullptr;
	int *flags = nullptr;
	
	// Board Phase Culling Data
	const int max_collision_pairs = 1000000;
	int2 *collision_pairs = nullptr;
	int *total_pairs_d = nullptr;
	int total_pairs_h = 0;

	// Narrow Phase Detection Data
	int4 *exact_collision_pairs = nullptr;
	int *total_exact_pairs_d = nullptr;
	int total_exact_pairs_h = 0;

	unsigned int *sort_buffer = nullptr;
	size_t sort_buffer_size = 0;

	// Lighting Data
	bool sm_buffer_initialized;
	int last_lighting_H = 0;
	int last_lighting_W = 0;
	float* sm_image_d = nullptr;
	float* sm_depth_d = nullptr;
	float* sm_alpha_d = nullptr;

	bool use_shadow_map = false;
	float shadow_eps = 5e-2f;
	float shadow_factor = 0.5f;
	int lighting_H = 0;
	int lighting_W = 0;
	Matrix4f lighting_view_mat;
	Matrix4f lighting_proj_view_mat;
	Vector3f lighting_pos;
	float lighting_fovy;
	float* lighting_view_d = nullptr;
	float* lighting_proj_d = nullptr;
	float* lighting_cam_pos_d = nullptr;

	// Controller Data (from Unity)
	struct VRController {
		Vector3f last_controller_pos;
		Vector3f controller_pos;
		Matrix3f last_controller_rot;
		Matrix3f controller_rot;
		int last_last_triggered = false;
		int last_triggered = false;
		int triggered = false;

		float* controller_pos_d = nullptr;
		float* controller_vel_d = nullptr;
		float* controller_ang_vel_d = nullptr;
	};
	VRController left, right;

	// Misc Parameters
	float controller_radius = 1000.0f;
	int zup = 0;
	float ground_height = 0.f;

	// Experimental Options
	int disable_LG_interp = 0;
	int enable_export_frames = false;
    int max_export_sequence = 100;

	// Quasi-static Update
	float* quasi_verts_d = nullptr;
	int* quasi_faces_ind_d = nullptr;

	int quasi_static = 0;
	int max_quasi_sequence = 0;

	int boundary = 0;

public:
	// Cpu version of the datas
	struct GSObject {
		// Per-object Simulation Parameters
		std::vector<float> density;
		std::vector<float> lambda;
		std::vector<float> mu;

		// GS particles
		int num_gs;
		int gs_offset;
		std::vector<Pos> pos;
		std::vector<Rot> rot;
		std::vector<Scale> scale;
		std::vector<float> opacity;
		std::vector<SHs<3>> shs;
		std::vector<Matrix3f> init_rot;

		// Mesh
		int num_verts;
		int num_edges;
		int num_faces;
		int num_cells;
		int verts_offset;
		int edges_offset;
		int faces_offset;
		int cells_offset;
		std::vector<Pos> mesh_verts;
		std::vector<int> mesh_verts_group;
		std::vector<int> mesh_edges;
		std::vector<int> mesh_faces;
		std::vector<int> mesh_cells;

		bool is_background;
		bool is_rigid;
		std::vector<int> rigid_group;
	};
	std::vector<GSObject> gs_objects;

	// Configurable Scene Parameters (from Unity)
	float global_scale = 1.0f;
	std::vector<float> global_offset;
	std::vector<float> object_offsets;
	const int material_property_size = 9;
	std::vector<float> object_materials;

	Eigen::Quaternion<float> global_rotation_q;
	Eigen::Matrix<float, 3, 3> global_rotation;

	// Profiling Data
#define BENCHMARK 114514
#define PROFILING
	long long collision_detection_time = 0;
	long long fem_solve_time = 0;
	long long collision_solve_time = 0;
	long long xpbd_time = 0;
	long long embedding_time = 0;
	long long left_time = 0;
	long long right_time = 0;
	long long shadow_time = 0;
	bool rendering_right = false;

#ifdef PROFILING
#define PROFILE(var, func) \
	{ \
	CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
	long long ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); \
	func; \
	CUDA_SAFE_CALL(cudaDeviceSynchronize()); \
	long long after_ts = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); \
	var += (after_ts - ts); \
	}
#else
#define PROFILE(var, func) func;
#endif
public:
	// Rendering Funcs
	void Load(const char* filepath);
	void RenderShadowMap();
	void RenderImage(float* image_d, float* depth_d, float* alpha_d, Matrix4f view_mat, Matrix4f proj_mat, Vector3f position, float fovy, int width, int height);
	void GetSceneSize(float* scene_min, float* scene_max);

	// Simulation Funcs
	void Update();
	void Save();

	// Collision Part
	void GetGlobalAABB();
	void ConstructBVH();
	void BoardPhaseCulling(const float minimal_dist);
	void NarrowPhaseDetection(const float minimal_dist);
	void CollisionDetection(const float minimal_dist);

	// Backend - Unity Communication
	void SetSimParams(float frame_dt, 
					  float dt, 
					  float gravity, 
					  float damping_coeffient, 
					  int rest_iter,
					  float collision_stiffness, 
					  float minimal_dist, 
					  float collision_detection_iter_interval,
					  float shadow_eps,
					  float shadow_factor,
					  int zup, 
					  float ground_height, 
					  float global_scale, 
					  float *global_offset, 
					  float *global_rotation, 
					  int num_objects, 
					  float *object_offsets,
					  float *object_materials,
					  int disable_LG_interp,
					  int quasi_static,
					  int max_quasi_sequence,
					  int enable_export_frames,
            		  int max_export_sequence,
					  int boundary);
	void SetController(
		float controller_radius,
		float *lpos, float *lrot, int ltriggered,
		float *rpos, float *rrot, int rtriggered);
	void GetControllerVel(
		float *lpos, float *lvel, float *lang_vel_d, 
		float *rpos, float *rvel, float *rang_vel_d, 
		float dt);
};
