#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include "SimKernels.h"
#include "lbvh/LBVH.cuh"
#include "RadixSort.cuh"
#include <cuda_runtime.h>
#include <cuda_rasterizer/rasterizer.h>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <filesystem>
#include <random>

using namespace std;

inline float sigmoid(const float m1) { return 1.0f / (1.0f + exp(-m1)); }
inline float inverse_sigmoid(const float m1) { return log(m1 / (1.0f - m1)); }

inline std::function<char* (size_t N)> resizeFunctional(void** ptr, size_t& S) {
	auto lambda = [ptr, &S](size_t N) {
		if (N > S)
		{
			if (*ptr)
				CUDA_SAFE_CALL(cudaFree(*ptr));
			CUDA_SAFE_CALL(cudaMalloc(ptr, 2 * N));
			S = 2 * N;
		}
		return reinterpret_cast<char*>(*ptr);
	};
	return lambda;
}

// Save PLY
void savePly(const char* filename,
	const std::vector<Pos>& pos,
	const std::vector<SHs<3>>& shs,
	const std::vector<float>& opacities,
	const std::vector<Scale>& scales,
	const std::vector<Rot>& rot)
{
	// Read all Gaussians at once (AoS)
	int count = 0;
	for (int i = 0; i < pos.size(); i++) {
		if (opacities[i] > 1e-5f) count++;
	}
	std::vector<RichPoint<3>> points(count);

	// Output number of Gaussians contained
	std::ofstream outfile(filename, std::ios_base::binary);

	outfile << "ply\nformat binary_little_endian 1.0\nelement vertex " << count << "\n";

	std::string props1[] = { "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
	std::string props2[] = { "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3" };

	for (auto s : props1)
		outfile << "property float " << s << std::endl;
	for (int i = 0; i < 45; i++)
		outfile << "property float f_rest_" << i << std::endl;
	for (auto s : props2)
		outfile << "property float " << s << std::endl;
	outfile << "end_header" << std::endl;

	count = 0;
	for (int i = 0; i < pos.size(); i++)
	{
		if (opacities[i] <= 1e-5f) continue;

		points[count].pos = pos[i];
		points[count].rot = rot[i];
		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			points[count].scale.scale[j] = log(scales[i].scale[j]);
		// Activate alpha
		points[count].opacity = inverse_sigmoid(opacities[i]);
		points[count].shs.shs[0] = shs[i].shs[0];
		points[count].shs.shs[1] = shs[i].shs[1];
		points[count].shs.shs[2] = shs[i].shs[2];
		for (int j = 1; j < 16; j++)
		{
			points[count].shs.shs[(j - 1) + 3] = shs[i].shs[j * 3 + 0];
			points[count].shs.shs[(j - 1) + 18] = shs[i].shs[j * 3 + 1];
			points[count].shs.shs[(j - 1) + 33] = shs[i].shs[j * 3 + 2];
		}
		count++;
	}
	outfile.write((char*)points.data(), sizeof(RichPoint<3>) * points.size());
}

// Load the Gaussians from the given file.
template<int D>
int loadPly(std::string filename,
	std::vector<Pos>& pos,
	std::vector<SHs<3>>& shs,
	std::vector<float>& opacities,
	std::vector<Scale>& scales,
	std::vector<Rot>& rot,
	Vector3f& minn,
	Vector3f& maxx)
{
	std::ifstream infile(filename, std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error((stringstream() << "Unable to find model's PLY file, attempted:\n" << filename).str());

	// "Parse" header (it has to be a specific format anyway)
	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	// Read all Gaussians at once (AoS)
	std::vector<RichPoint<D>> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPoint<D>));

	// Resize our SoA data
	pos.resize(count);
	shs.resize(count);
	scales.resize(count);
	rot.resize(count);
	opacities.resize(count);

	// Gaussians are done training, they won't move anymore. Arrange
	// them according to 3D Morton order. This means better cache
	// behavior for reading Gaussians that end up in the same tile 
	// (close in 3D --> close in 2D).
	for (int i = 0; i < count; i++)
	{
		maxx = maxx.cwiseMax(points[i].pos);
		minn = minn.cwiseMin(points[i].pos);
	}
	std::vector<std::pair<uint64_t, int>> mapp(count);
	for (int i = 0; i < count; i++)
	{
		Vector3f rel = (points[i].pos - minn).array() / (maxx - minn).array();
		Vector3f scaled = ((float((1 << 21) - 1)) * rel);
		Vector3i xyz = scaled.cast<int>();

		uint64_t code = 0;
		for (int i = 0; i < 21; i++) {
			code |= ((uint64_t(xyz.x() & (1 << i))) << (2 * i + 0));
			code |= ((uint64_t(xyz.y() & (1 << i))) << (2 * i + 1));
			code |= ((uint64_t(xyz.z() & (1 << i))) << (2 * i + 2));
		}

		mapp[i].first = code;
		mapp[i].second = i;
	}
	auto sorter = [](const std::pair < uint64_t, int>& a, const std::pair < uint64_t, int>& b) {
		return a.first < b.first;
	};
	std::sort(mapp.begin(), mapp.end(), sorter);

	// Move data from AoS to SoA
	int SH_N = (D + 1) * (D + 1);
	for (int k = 0; k < count; k++)
	{
		int i = mapp[k].second;
		pos[k] = points[i].pos;

		// Normalize quaternion
		float length2 = 0;
		for (int j = 0; j < 4; j++)
			length2 += points[i].rot.rot[j] * points[i].rot.rot[j];
		float length = sqrt(length2);
		for (int j = 0; j < 4; j++)
			rot[k].rot[j] = points[i].rot.rot[j] / length;

		// Exponentiate scale
		for (int j = 0; j < 3; j++)
			scales[k].scale[j] = exp(points[i].scale.scale[j]);

		// Activate alpha
		opacities[k] = sigmoid(points[i].opacity);

		shs[k].shs[0] = points[i].shs.shs[0];
		shs[k].shs[1] = points[i].shs.shs[1];
		shs[k].shs[2] = points[i].shs.shs[2];
		for (int j = 1; j < SH_N; j++)
		{
			shs[k].shs[j * 3 + 0] = points[i].shs.shs[(j - 1) + 3];
			shs[k].shs[j * 3 + 1] = points[i].shs.shs[(j - 1) + SH_N + 2];
			shs[k].shs[j * 3 + 2] = points[i].shs.shs[(j - 1) + 2 * SH_N + 1];
		}
	}

	bool random_cutoff = false;
	float random_scale = 0.05;
	std::default_random_engine engine;
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
	if (random_cutoff) {
		std::vector<Pos> pos_N;
		std::vector<SHs<3>> shs_N;
		std::vector<float> opacities_N;
		std::vector<Scale> scales_N;
		std::vector<Rot> rot_N;
		for (int i = 0; i < count; i++) {
			if (distribution(engine) < random_scale) {
				pos_N.push_back(pos[i]);
				shs_N.push_back(shs[i]);
				opacities_N.push_back(opacities[i]);
				scales_N.push_back(scales[i]);
				rot_N.push_back(rot[i]);
			}
		}
		count = pos_N.size();
		pos = pos_N;
		shs = shs_N;
		opacities = opacities_N;
		scales = scales_N;
		rot = rot_N;
	}

	return count;
}

// Load the Quasi-static sequence face mesh
void loadObj(std::string filename,
	std::vector<Pos>& verts,
	std::vector<int>& faces) {
	verts.clear();
	faces.clear();

	std::ifstream file(filename);
	std::string line;
	while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string type;
        lineStream >> type;

        if (type == "v") {
            Pos pos;
            lineStream >> pos[0] >> pos[1] >> pos[2];
            verts.push_back(pos);
        } else if (type == "f") {
            int a, b, c;
            lineStream >> a >> b >> c;
            faces.push_back(a - 1);
			faces.push_back(b - 1);
			faces.push_back(c - 1);
        }
    }

    file.close();
}

// Load the Cage mesh from the given file.
void loadTet(std::string filename,
	std::vector<Pos>& verts,
	std::vector<int>& edges,
	std::vector<int>& faces,
	std::vector<int>& cells) {

	std::ifstream file(filename);
	int numVerts, numCells;
    file >> numVerts >> numCells;

	verts.resize(numVerts);
	cells.resize(numCells * 4);

	for (int i = 0; i < numVerts; i++) {
		file >> verts[i][0] >> verts[i][1] >> verts[i][2];
	}

	std::set<std::pair<int, int>> edges_set;
	auto add_edge = [&](int x, int y) {
		const auto &edge = std::make_pair(min(x, y), max(x, y));
		if (edges_set.find(edge) == edges_set.end()) {
			edges_set.insert(edge);
		}
	};

	std::map<std::tuple<int, int, int>, std::tuple<int, int, int>> face_set;
	auto add_face = [&](int x, int y, int z) {
		int i0 = min(min(x, y), z);
		int i2 = max(max(x, y), z);
		int i1 = (x ^ y ^ z) ^ i0 ^ i2;
		const auto &face_key = std::make_tuple(i0, i1, i2);
		const auto &face_value = std::make_tuple(x, y, z);
		if (face_set.find(face_key) == face_set.end()) {
			face_set.insert(std::make_pair(face_key, face_value));
		}
		else {
			// If a face appears twice, it is shared by two tets and not a surface face.
			face_set.erase(face_key);
		}
	};

	for (int i = 0; i < numCells; i++) {
		file >> cells[i*4+0] >> cells[i*4+1] >> cells[i*4+2] >> cells[i*4+3];
		add_edge(cells[i*4+0], cells[i*4+1]);
		add_edge(cells[i*4+0], cells[i*4+2]);
		add_edge(cells[i*4+0], cells[i*4+3]);
		add_edge(cells[i*4+1], cells[i*4+2]);
		add_edge(cells[i*4+1], cells[i*4+3]);
		add_edge(cells[i*4+2], cells[i*4+3]);

		add_face(cells[i*4+0], cells[i*4+2], cells[i*4+1]);
		add_face(cells[i*4+0], cells[i*4+3], cells[i*4+2]);
		add_face(cells[i*4+0], cells[i*4+1], cells[i*4+3]);
		add_face(cells[i*4+1], cells[i*4+2], cells[i*4+3]);
	}

	edges.clear();
	for (const auto &edge : edges_set) {
		edges.push_back(edge.first);
		edges.push_back(edge.second);
	}

	faces.clear();
	for (const auto &face : face_set) {
		faces.push_back(std::get<0>(face.second));
		faces.push_back(std::get<1>(face.second));
		faces.push_back(std::get<2>(face.second));
	}

	file.close();
}

void GaussianSplatting::GetGlobalAABB() {
    const int max_threads = 256;
    int threads = (total_faces < max_threads*2) ? lbvh::next_pow_2((total_faces + 1)/ 2) : max_threads;
    int blocks = ::min((int(total_faces) + (threads * 2 - 1)) / (threads * 2), 64);

    lbvh::launch_aabb_reduce(blocks, threads, tri_aabbs, partial_aabb_d, total_faces);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaMemcpy(partial_aabb_h, partial_aabb_d, blocks * sizeof(lbvh::aabb<float>), cudaMemcpyDeviceToHost);
    for (int i = 1; i < blocks; ++i) {
        partial_aabb_h[0] = merge(partial_aabb_h[0], partial_aabb_h[i]);
    }

    aabb_global = partial_aabb_h[0];
    cudaMemcpy(partial_aabb_d, partial_aabb_h, sizeof(lbvh::aabb<float>), cudaMemcpyHostToDevice);
}

void GaussianSplatting::ConstructBVH() {
// #define DEBUG_SORT
// #define DEBUG_NODE
// #define DEBUG_AABB

	lbvh::compute_tri_aabbs << < (total_faces + 255) / 256, 256>> >(total_faces, verts_x_d, faces_ind_d, tri_aabbs);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	const size_t num_objects = total_faces;
    const size_t num_internal_nodes = num_objects - 1;
    const size_t num_nodes = num_objects * 2 - 1;

    GetGlobalAABB();

	// Sort AABB & index based on Morton code
    // https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
	{
		lbvh::compute_morton_and_indices << < (num_objects + 255) / 256, 256 >> > (num_objects, partial_aabb_d, tri_aabbs, morton_code, indices);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef DEBUG_SORT
		uint64_t *morton_code_cpu = (uint64_t*)malloc(sizeof(uint64_t) * num_objects);
		int *indices_cpu = (int*)malloc(sizeof(int) * num_objects);
		lbvh::aabb<float> *tri_aabbs_cpu = (lbvh::aabb<float>*)malloc(sizeof(lbvh::aabb<float>) * num_objects);
		CUDA_SAFE_CALL(cudaMemcpy(morton_code_cpu, morton_code, num_objects * sizeof(uint64_t), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(indices_cpu, indices, num_objects * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tri_aabbs_cpu, tri_aabbs, num_objects * sizeof(lbvh::aabb<float>), cudaMemcpyDeviceToHost));

		if (global_step == 0) {
			FLOG(for (int i=0; i<5000;i++) {
			sim_log << "morton: " << i << " " << indices_cpu[i] << " " << morton_code_cpu[i] << " ";
			sim_log << "(" << tri_aabbs_cpu[i].lower.x << " " << tri_aabbs_cpu[i].lower.y << " " << tri_aabbs_cpu[i].lower.z << ") ";
			sim_log << "(" << tri_aabbs_cpu[i].upper.x << " " << tri_aabbs_cpu[i].upper.y << " " << tri_aabbs_cpu[i].upper.z << ")\n";
		});}
#endif

		radix_sort(sorted_morton_code, morton_code, sorted_indices, indices, sort_buffer, sort_buffer_size, num_objects);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		lbvh::get_sorted_tri_aabbs << < (num_objects + 255) / 256, 256 >> >(num_objects, sorted_indices, tri_aabbs, sorted_tri_aabbs);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		
#ifdef DEBUG_SORT
		CUDA_SAFE_CALL(cudaMemcpy(morton_code_cpu, sorted_morton_code, num_objects * sizeof(uint64_t), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(indices_cpu, sorted_indices, num_objects * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(tri_aabbs_cpu, sorted_tri_aabbs, num_objects * sizeof(lbvh::aabb<float>), cudaMemcpyDeviceToHost));
		if (global_step == 0) { 
			FLOG(for (int i=0; i<5000;i++) {
				sim_log << "sorted morton: " << i << " " 
						<< indices_cpu[i] << " " 
						<< morton_code_cpu[i] << " ";
				sim_log << "(" << tri_aabbs_cpu[i].lower.x 
						<< " " << tri_aabbs_cpu[i].lower.y 
						<< " " << tri_aabbs_cpu[i].lower.z << ") "
						<< "(" << tri_aabbs_cpu[i].upper.x 
						<< " " << tri_aabbs_cpu[i].upper.y 
						<< " " << tri_aabbs_cpu[i].upper.z << ")\n";
			});
		}

		free((void*)morton_code_cpu);
		free((void*)indices_cpu);
		free((void*)tri_aabbs_cpu);
#endif
	}

	lbvh::reset_aabb << < (num_nodes + 255) / 256, 256  >> > (num_objects, sorted_tri_aabbs, aabbs);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemset(nodes, 0xFF, sizeof(lbvh::Node) * num_nodes));
	lbvh::construct_internal_nodes << < (num_objects - 1 + 255) / 256, 256  >> > (num_objects, sorted_morton_code, nodes);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef DEBUG_NODE
	if (global_step == 0) {
		lbvh::Node *nodes_cpu = (lbvh::Node*)malloc(sizeof(lbvh::Node) * num_nodes);
		CUDA_SAFE_CALL(cudaMemcpy(nodes_cpu, nodes, num_nodes * sizeof(lbvh::Node), cudaMemcpyDeviceToHost));
		FLOG(for (int i = 0; i < num_nodes; i++) {
			sim_log << "node: " << i << " " << nodes_cpu[i].parent_idx << " " << nodes_cpu[i].left_idx << " " << nodes_cpu[i].right_idx << "\n" ;
		});
		free((void*)nodes_cpu);
	}
#endif

	CUDA_SAFE_CALL(cudaMemset(flags, 0, sizeof(int) * num_internal_nodes));
	lbvh::compute_internal_aabbs << < (num_objects + 255) / 256, 256  >> > (num_objects, nodes, aabbs, flags);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef DEBUG_AABB
	if (global_step == 0) {
		int *flags_cpu = (int*)malloc(sizeof(int) * num_internal_nodes);
		lbvh::aabb<float> *aabbs_cpu = (lbvh::aabb<float>*)malloc(sizeof(lbvh::aabb<float>) * num_nodes);
		CUDA_SAFE_CALL(cudaMemcpy(flags_cpu, flags, sizeof(int) * num_internal_nodes, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(aabbs_cpu, aabbs, sizeof(lbvh::aabb<float>) * num_nodes, cudaMemcpyDeviceToHost));
		FLOG(for (int i = 0; i < num_internal_nodes; i++) {
			sim_log << i << " flag: " << flags_cpu[i] << " aabb: ";
			sim_log << "(" << aabbs_cpu[i].lower.x << " " << aabbs_cpu[i].lower.y << " " << aabbs_cpu[i].lower.z << ") ";
			sim_log << "(" << aabbs_cpu[i].upper.x << " " << aabbs_cpu[i].upper.y << " " << aabbs_cpu[i].upper.z << ")\n";
		});
	}
#endif

	/*FLOG(
	sim_log << "(" << aabb_global.lower.x << " " << aabb_global.lower.y << " " << aabb_global.lower.z << ") ";
	sim_log << "(" << aabb_global.upper.x << " " << aabb_global.upper.y << " " << aabb_global.upper.z << ")\n";
	);*/
}

void GaussianSplatting::BoardPhaseCulling(const float minimal_dist) {
	CUDA_SAFE_CALL(cudaMemset(total_pairs_d, 0, sizeof(int)));
	// The previous tri_aabbs was disrupted by radix sort and requires to be re-computed.
	// And should be extended by minimal distance.
	constexpr float sqrt3 = 1.73205080757f;
	lbvh::compute_tri_aabbs << < (total_faces + 255) / 256, 256>> >(total_faces, 
		verts_x_d, 
		faces_ind_d, 
		tri_aabbs, 
		minimal_dist * sqrt3);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	constexpr unsigned int culling_block_size = 128;
	lbvh::query_collision_pairs <float, 64, culling_block_size>
		<< < (total_faces + culling_block_size - 1) / culling_block_size, culling_block_size  >> > 
		(total_faces,
		tri_aabbs,
        nodes,
        aabbs, 
        sorted_indices,
        collision_pairs,
        total_pairs_d,
        total_faces - 1,
        max_collision_pairs);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(&total_pairs_h, total_pairs_d, sizeof(int), cudaMemcpyDeviceToHost));

	// If the number of pairs exceeds the buffer size, clamp it
	total_pairs_h = ::min(total_pairs_h, max_collision_pairs);

// #define DEBUG_CULLING
// #define DEBUG_CULLING_EXPORT

#ifdef DEBUG_CULLING
	FLOG(sim_log << "total_faces: " << total_faces << " "
				 << "total_pairs: " << total_pairs_h << "\n";);

	int2 *collision_pairs_cpu = (int2*)malloc(sizeof(int2) * total_pairs_h);

	CUDA_SAFE_CALL(cudaMemcpy(collision_pairs_cpu, collision_pairs, sizeof(int2) * total_pairs_h, cudaMemcpyDeviceToHost));
	FLOG(for(int i = 0; i < total_pairs_h; i++) {
		sim_log << collision_pairs_cpu[i].x << " " 
				<< collision_pairs_cpu[i].y << "\n";
	});

	free((void*)collision_pairs_cpu);
#endif

#ifdef DEBUG_CULLING_EXPORT
	if (global_step == 0) {
		int2 *collision_pairs_cpu = (int2*)malloc(sizeof(int2) * total_pairs_h);
		CUDA_SAFE_CALL(cudaMemcpy(collision_pairs_cpu, collision_pairs, sizeof(int2) * total_pairs_h, cudaMemcpyDeviceToHost));
		int *faces_ind_cpu = (int*)malloc(sizeof(int) * 3 * total_faces);
		CUDA_SAFE_CALL(cudaMemcpy(faces_ind_cpu, faces_ind_d, sizeof(int) * 3 * total_faces, cudaMemcpyDeviceToHost));
		int *verts_group_cpu = (int*)malloc(sizeof(int) * total_verts);
		CUDA_SAFE_CALL(cudaMemcpy(verts_group_cpu, verts_group_d, sizeof(int) * total_verts, cudaMemcpyDeviceToHost));

		std::vector<int> verts_in_collision(total_verts, 0);
		int total_pairs = 0;
		for(int i = 0; i < total_pairs_h; i++) {
			if (verts_group_cpu[faces_ind_cpu[collision_pairs_cpu[i].x * 3 + 0]] !=	
				verts_group_cpu[faces_ind_cpu[collision_pairs_cpu[i].y * 3 + 0]]) {
				verts_in_collision[faces_ind_cpu[collision_pairs_cpu[i].x * 3 + 0]] = 1;
				verts_in_collision[faces_ind_cpu[collision_pairs_cpu[i].x * 3 + 1]] = 1;
				verts_in_collision[faces_ind_cpu[collision_pairs_cpu[i].x * 3 + 2]] = 1;
				verts_in_collision[faces_ind_cpu[collision_pairs_cpu[i].y * 3 + 0]] = 1;
				verts_in_collision[faces_ind_cpu[collision_pairs_cpu[i].y * 3 + 1]] = 1;
				verts_in_collision[faces_ind_cpu[collision_pairs_cpu[i].y * 3 + 2]] = 1;
				total_pairs++;
			}
		}
		FLOG(sim_log << "total_pairs: " << total_pairs << "\n"; );

		free((void*)collision_pairs_cpu);
		free((void*)faces_ind_cpu);
		free((void*)verts_group_cpu);
		FDUMP("culling_collision.obj",
		for (int i = 0; i < gs_objects.size(); i++) {
			for (int k = 0; k < gs_objects[i].num_verts; k++) {
				int r = 1.0;
				int gb = verts_in_collision[k + gs_objects[i].verts_offset] ? 0.0 : 1.0;
				sim_log << "v " << gs_objects[i].mesh_verts[k][0] << " " 
								<< gs_objects[i].mesh_verts[k][1] << " " 
								<< gs_objects[i].mesh_verts[k][2] << " " << r << " " << gb << " " << gb << "\n";
			}
		}

		for (int i = 0; i < gs_objects.size(); i++) {
			for (int k = 0; k < gs_objects[i].num_faces; k++) {
				sim_log << "f " << gs_objects[i].mesh_faces[k*3+0]+1 
						 << " " << gs_objects[i].mesh_faces[k*3+1]+1 
						 << " " << gs_objects[i].mesh_faces[k*3+2]+1 << "\n";
			}
		});
	}
#endif
}

void GaussianSplatting::NarrowPhaseDetection(const float minimal_dist) {
	CUDA_SAFE_CALL(cudaMemset(total_exact_pairs_d, 0, sizeof(int)));
	lbvh::query_collision_triangles << < (total_pairs_h + 255) / 256, 256  >> > 
		(total_pairs_h,
		minimal_dist,
        collision_pairs,
        verts_x_d,
		verts_group_d,
        faces_ind_d,
        exact_collision_pairs,
        total_exact_pairs_d,
		max_collision_pairs);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(&total_exact_pairs_h, total_exact_pairs_d, sizeof(int), cudaMemcpyDeviceToHost));

	// If the number of pairs exceeds the buffer size, clamp it
	total_exact_pairs_h = ::min(total_exact_pairs_h, max_collision_pairs);

// #define DEBUG_DETECTION
// #define DEBUG_EXPORT

#ifdef DEBUG_DETECTION
	FLOG(sim_log << "num_pairs(board): " << total_pairs_h 
				 << " num_pairs(narrow): " << total_exact_pairs_h << "\n";);

	if (global_step == 0) {
		int4 *collision_pairs_cpu = (int4*)malloc(sizeof(int4) * total_exact_pairs_h);
		
		CUDA_SAFE_CALL(cudaMemcpy(collision_pairs_cpu, exact_collision_pairs, sizeof(int4) * total_exact_pairs_h, cudaMemcpyDeviceToHost));
		FLOG(for(int i = 0; i < total_exact_pairs_h; i++) {
			sim_log << collision_pairs_cpu[i].x << " " 
					<< collision_pairs_cpu[i].y << " "
					<< collision_pairs_cpu[i].z << " " 
					<< collision_pairs_cpu[i].w << "\n";
		});
		
		free((void*)collision_pairs_cpu);
	}
#endif

#ifdef DEBUG_EXPORT
	if (global_step == 0) {
		int4 *collision_pairs_cpu = (int4*)malloc(sizeof(int4) * total_exact_pairs_h);
		CUDA_SAFE_CALL(cudaMemcpy(collision_pairs_cpu, exact_collision_pairs, sizeof(int4) * total_exact_pairs_h, cudaMemcpyDeviceToHost));
		int *faces_ind_cpu = (int*)malloc(sizeof(int) * 3 * total_faces);
		CUDA_SAFE_CALL(cudaMemcpy(faces_ind_cpu, faces_ind_d, sizeof(int) * 3 * total_faces, cudaMemcpyDeviceToHost));
		float *verts_x_cpu = (float*)malloc(sizeof(float) * 3 * total_verts);
		CUDA_SAFE_CALL(cudaMemcpy(verts_x_cpu, verts_x_d, sizeof(float) * 3 * total_verts, cudaMemcpyDeviceToHost));

		std::vector<int> verts_in_collision(total_verts, 0);
		for(int i = 0; i < total_exact_pairs_h; i++) {
			verts_in_collision[collision_pairs_cpu[i].x] = 1;
			verts_in_collision[collision_pairs_cpu[i].y] = 1;
			verts_in_collision[collision_pairs_cpu[i].z] = 1;
			verts_in_collision[collision_pairs_cpu[i].w] = 1;
		}

		FDUMP("collision.obj",
		for (int i = 0; i < total_verts; i++) {
			int r = 1.0;
			int gb = verts_in_collision[i] ? 0.0 : 1.0;
			sim_log << "v " << verts_x_cpu[i*3+0] << " " 
							<< verts_x_cpu[i*3+1] << " " 
							<< verts_x_cpu[i*3+2] << " " 
							<< r << " " << gb << " " << gb << "\n";
		}

		for (int i = 0; i < total_faces; i++) {
			sim_log << "f " << faces_ind_cpu[i*3+0]+1 
					 << " " << faces_ind_cpu[i*3+1]+1 
					 << " " << faces_ind_cpu[i*3+2]+1 << "\n";
		});

		free((void*)collision_pairs_cpu);
		free((void*)faces_ind_cpu);
		free((void*)verts_x_cpu);
	}
#endif
}

void GaussianSplatting::CollisionDetection(const float minimal_dist) {
	ConstructBVH();
	BoardPhaseCulling(minimal_dist);
	NarrowPhaseDetection(minimal_dist);
}

void GaussianSplatting::Update() {
	if (quasi_static) {
		if (global_step >= max_quasi_sequence) {
			return;
		}
		std::vector<Pos> seq_pos;
		std::vector<int> seq_faces;
		std::ostringstream fname;
		fname << "sequence/frame_" << std::setw(4) << std::setfill('0') << global_step + 1 << ".obj";
		loadObj(global_path + fname.str(), seq_pos, seq_faces);
		FLOG(sim_log<< global_path + fname.str() << " " << total_faces << " " << seq_faces.size() / 3 << std::endl; );

		PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemset(verts_selected_d, 0, sizeof(int) * total_verts)));
		PROFILE(xpbd_time, sim::enforce_quasi_boundary_condition(total_faces * 3, verts_x_d, faces_ind_d, quasi_verts_d, quasi_faces_ind_d, verts_selected_d));
		PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemcpy(quasi_verts_d, seq_pos.data(), sizeof(float) * 3 * seq_pos.size(), cudaMemcpyHostToDevice)));
		PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemcpy(quasi_faces_ind_d, seq_faces.data(), sizeof(int) * seq_faces.size(), cudaMemcpyHostToDevice)));
		PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemset(cells_multiplier_d, 0, sizeof(float)*total_cells)));
		for (int i = 0; i < rest_iter; i++) {
			PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemset(verts_dp_d, 0, sizeof(float)*3*total_verts)));
			PROFILE(fem_solve_time, sim::solve_FEM_constraints(total_cells, 
									frame_dt, // `frame_dt` instead of `dt0` 
									cells_mu_d, 
									cells_lambda_d, 
									cells_ind_d, 
									verts_x_d, // `verts_x_d` instead of `verts_new_x_d`
									verts_inv_m_d, 
									cells_DS_inv_d, 
									cells_vol0_d, 
									verts_dp_d, 
									cells_multiplier_d,
									rigid_verts_group_d,
									/*block_size*/ 128));
			PROFILE(xpbd_time, sim::pbd_post_solve(total_verts, 
												   verts_dp_d, 
												   verts_x_d, // `verts_x_d` instead of `verts_new_x_d`
												   verts_selected_d, 
												   1024));
		}
	} else {
		// FEM-PBD Simulation
		GetControllerVel(
			left.controller_pos_d, left.controller_vel_d, left.controller_ang_vel_d, 
			right.controller_pos_d, right.controller_vel_d, right.controller_ang_vel_d, 
			frame_dt);
		constexpr int LEFT_BIT = 1;
		constexpr int RIGHT_BIT = 2;
		if (left.triggered && left.last_triggered && !left.last_last_triggered) {
			sim::select_vertices(total_verts, controller_radius, 
							left.controller_pos_d, LEFT_BIT,
							verts_x_d, verts_selected_d);
		} else if (!left.triggered) {
			clean_bit << < (total_verts + 255) / 256, 256  >> > (total_verts, verts_selected_d, LEFT_BIT);
		}
		if (right.triggered && right.last_triggered && !right.last_last_triggered) {
			sim::select_vertices(total_verts, controller_radius, 
							right.controller_pos_d, RIGHT_BIT,
							verts_x_d, verts_selected_d);
		} else if (!right.triggered) {
			clean_bit << < (total_verts + 255) / 256, 256  >> > (total_verts, verts_selected_d, RIGHT_BIT);
		}

		int collision_cnt = 0;
		float dt_left = frame_dt;
		while (dt_left > 0.f) {
			if (collision_cnt % collision_dection_iter_interval == 0) {
				PROFILE(collision_detection_time, CollisionDetection(minimal_dist));
			}
			++collision_cnt;

			float dt0 = min(dt, dt_left);
			dt_left -= dt0;
			PROFILE(xpbd_time, sim::apply_external_force(total_verts, dt0, gravity, damping_coeffient, verts_inv_m_d, verts_x_d, verts_v_d, verts_new_x_d,
			left.controller_pos_d, right.controller_pos_d,
			left.controller_vel_d, right.controller_vel_d,
			left.controller_ang_vel_d, right.controller_ang_vel_d,
			verts_selected_d, zup));
			PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemset(cells_multiplier_d, 0, sizeof(float)*total_cells)));
			for (int i = 0; i < rest_iter; i++) {
				PROFILE(xpbd_time, CUDA_SAFE_CALL(cudaMemset(verts_dp_d, 0, sizeof(float)*3*total_verts)));
				PROFILE(fem_solve_time, sim::solve_FEM_constraints(total_cells, 
									dt0, 
									cells_mu_d, 
									cells_lambda_d, 
									cells_ind_d, 
									verts_new_x_d, 
									verts_inv_m_d, 
									cells_DS_inv_d, 
									cells_vol0_d, 
									verts_dp_d, 
									cells_multiplier_d,
									rigid_verts_group_d,
									/*block_size*/ 128));
				PROFILE(collision_solve_time, sim::solve_triangle_point_distance_constraint(
									total_exact_pairs_h,
									minimal_dist,
									collision_stiffness,
									exact_collision_pairs,
									verts_new_x_d,
									verts_inv_m_d,
									verts_dp_d,
									/*block_size*/ 128));
				
				PROFILE(xpbd_time, sim::pbd_post_solve(total_verts, verts_dp_d, verts_new_x_d, verts_selected_d, /*block_size*/ 1024));
			}
			PROFILE(xpbd_time, sim::pbd_advance(total_verts, dt0, verts_inv_m_d, verts_v_d, verts_x_d, verts_new_x_d, zup, ground_height));
		}
	}

	PROFILE(xpbd_time, sim::solve_rigid(
			total_verts, 
			gs_objects.size(),
			rigid_verts_group_d,
			rigid_m_d,
			rigid_cm0_d,
			rigid_cm_d,
			rigid_A_d,
			rigid_R_d,
			verts_X_d,
			verts_x_d,
			verts_m_d,
			/*block_size*/ 256));
// #define DEBUG_RIGID
#ifdef DEBUG_RIGID
	double *cm0_cpu = (double*)malloc(sizeof(double) * 3 * gs_objects.size());
	double *cm_cpu = (double*)malloc(sizeof(double) * 3 * gs_objects.size());
	double *R_cpu = (double*)malloc(sizeof(double) * 9 * gs_objects.size());
	CUDA_SAFE_CALL(cudaMemcpy(cm0_cpu, rigid_cm0_d, sizeof(double) * 3 * gs_objects.size(), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(cm_cpu, rigid_cm_d, sizeof(double) * 3 * gs_objects.size(), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(R_cpu, rigid_R_d, sizeof(double) * 9 * gs_objects.size(), cudaMemcpyDeviceToHost));

	FLOG(for (int i = 0; i < gs_objects.size(); i++) {
			sim_log << "cm0 " << cm0_cpu[i*3+0] << " " 
							<< cm0_cpu[i*3+1] << " " 
							<< cm0_cpu[i*3+2] << "\n";
		});
	FLOG(for (int i = 0; i < gs_objects.size(); i++) {
			sim_log << "cm " << cm_cpu[i*3+0] << " " 
							<< cm_cpu[i*3+1] << " " 
							<< cm_cpu[i*3+2] << "\n";
		});
	FLOG(for (int i = 0; i < gs_objects.size(); i++) {
			sim_log << "R " << R_cpu[i*9+0] << " " 
							<< R_cpu[i*9+1] << " " 
							<< R_cpu[i*9+2] << "\n";
			sim_log << "  " << R_cpu[i*9+3] << " " 
							<< R_cpu[i*9+4] << " " 
							<< R_cpu[i*9+5] << "\n";
			sim_log << "  " << R_cpu[i*9+6] << " " 
							<< R_cpu[i*9+7] << " " 
							<< R_cpu[i*9+8] << "\n";
		});
#endif
	PROFILE(embedding_time, sim::apply_interpolation(total_gs,
                        pos_d, scale_d, rot_d, 
                        cov_d, cells_ind_d, verts_X_d, verts_x_d,
                        global_tet_idx, global_tet_w, local_tet_w,
						disable_LG_interp, local_rot_d));

	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	global_step += 1;
#ifdef PROFILING
	long long test_num = 1000LL;
	if (global_step == test_num) {
		FDUMP(
			"sim_time.txt",
			sim_log << "Collision Detection " << collision_detection_time / test_num << " ms\n";
			sim_log << "FEM Solve " << fem_solve_time / test_num << " ms\n";
			sim_log << "Collision Solve " << collision_solve_time / test_num << " ms\n";
			sim_log << "XPBD " << xpbd_time / test_num << " ms\n";
			sim_log << "Embedding " << embedding_time / test_num << " ms\n";

			sim_log << "Shadow " << shadow_time / test_num << " ms\n";
			sim_log << "Left Eye " << left_time / test_num << " ms\n";
			sim_log << "Right Eye " << right_time / test_num << " ms\n";
		);
	}
#endif
}

void GaussianSplatting::GetControllerVel(
	float *lpos, float *lvel, float *lang_vel_d, 
	float *rpos, float *rvel, float *rang_vel_d, 
	float dt) {
	CUDA_SAFE_CALL(cudaMemcpy(lpos, left.last_controller_pos.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
	if (left.triggered && left.last_triggered) {
		float lvel_h[3];
		for (int i = 0; i < 3; i++) {
			lvel_h[i] = (left.controller_pos[i] - left.last_controller_pos[i]) / dt;
		}
		Matrix3f lang_vel = left.controller_rot * left.last_controller_rot.inverse();
		CUDA_SAFE_CALL(cudaMemcpy(lvel, lvel_h, sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(lang_vel_d, lang_vel.data(), sizeof(float) * 9, cudaMemcpyHostToDevice));
	} else {
		CUDA_SAFE_CALL(cudaMemset(lvel, 0, sizeof(float) * 3));
		CUDA_SAFE_CALL(cudaMemset(lang_vel_d, 0, sizeof(float) * 9));
	}

	CUDA_SAFE_CALL(cudaMemcpy(rpos, right.last_controller_pos.data(), sizeof(float) * 3, cudaMemcpyHostToDevice));
	if (right.triggered && right.last_triggered) {
		float rvel_h[3];
		for (int i = 0; i < 3; i++) {
			rvel_h[i] = (right.controller_pos[i] - right.last_controller_pos[i]) / dt;
		}
		Matrix3f rang_vel = right.controller_rot * right.last_controller_rot.inverse();
		CUDA_SAFE_CALL(cudaMemcpy(rvel, rvel_h, sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(rang_vel_d, rang_vel.data(), sizeof(float) * 9, cudaMemcpyHostToDevice));
	} else {
		CUDA_SAFE_CALL(cudaMemset(rvel, 0, sizeof(float) * 3));
		CUDA_SAFE_CALL(cudaMemset(rang_vel_d, 0, sizeof(float) * 9));
	}
}

void GaussianSplatting::SetSimParams(
	float _frame_dt, 
	float _dt, 
	float _gravity,
	float _damping_coeffient,
	int _rest_iter, 
	float _collision_stiffness, 
	float _minimal_dist, 
	float _collision_detection_iter_interval,
	float _shadow_eps,
	float _shadow_factor,
	int _zup, 
	float _ground_height,
	float _global_scale, 
	float *_global_offset, 
	float *_global_rotation,
	int _num_objects, 
	float *_object_offsets, 
	float *_object_materials,
	int _disable_LG_interp,
	int _quasi_static,
	int _max_quasi_sequence,
	int _enable_export_frames,
    int _max_export_sequence,
	int _boundary) {
	frame_dt = _frame_dt;
	dt = _dt;
	gravity = _gravity;
	damping_coeffient = _damping_coeffient;
	rest_iter = _rest_iter;
	collision_stiffness = _collision_stiffness;
	minimal_dist = _minimal_dist;
	collision_dection_iter_interval = _collision_detection_iter_interval;
	shadow_eps = _shadow_eps;
	shadow_factor = _shadow_factor;
	zup = _zup;
	ground_height = _ground_height;

	global_scale = _global_scale;
	Eigen::Quaternion<float> q(_global_rotation[3], _global_rotation[0], _global_rotation[1], _global_rotation[2]);
	global_rotation = q.toRotationMatrix();
	global_rotation_q = q;
	global_offset = std::vector<float>(_global_offset, _global_offset + 3);
	object_offsets = std::vector<float>(_object_offsets, _object_offsets + 3 * _num_objects);
	object_materials = std::vector<float>(_object_materials, _object_materials + material_property_size * _num_objects);

	// Experimental Options
	disable_LG_interp = _disable_LG_interp;
	quasi_static = _quasi_static;
	max_quasi_sequence = _max_quasi_sequence;
	enable_export_frames = _enable_export_frames;
    max_export_sequence = _max_export_sequence;
	boundary = _boundary;
}

void GaussianSplatting::SetController(
		float radius,
		float *lpos, float *lrot, int ltriggered,
		float *rpos, float *rrot, int rtriggered) {
	controller_radius = radius;
	left.last_last_triggered = left.last_triggered;
	left.last_triggered = left.triggered;
	left.last_controller_pos = left.controller_pos;
	left.last_controller_rot = left.controller_rot;
	left.triggered = ltriggered;
	left.controller_pos = Vector3f(lpos);
	left.controller_rot = Eigen::Quaternion<float>(lrot[3], lrot[0], lrot[1], lrot[2]).toRotationMatrix();

	right.last_last_triggered = right.last_triggered;
	right.last_triggered = right.triggered;
	right.last_controller_pos = right.controller_pos;
	right.last_controller_rot = right.controller_rot;
	right.triggered = rtriggered;
	right.controller_pos = Vector3f(rpos);
	right.controller_rot = Eigen::Quaternion<float>(rrot[3], rrot[0], rrot[1], rrot[2]).toRotationMatrix();
}

void GaussianSplatting::GetSceneSize(float* scene_min, float* scene_max) {
	scene_min[0] = _scenemin.x();
	scene_min[1] = _scenemin.y();
	scene_min[2] = _scenemin.z();
	scene_max[0] = _scenemax.x();
	scene_max[1] = _scenemax.y();
	scene_max[2] = _scenemax.z();
}

void GaussianSplatting::Load(const char* filepath) {
	use_shadow_map = false;
	global_step = 0;
	collision_detection_time = 0;
	fem_solve_time = 0;
	collision_solve_time = 0;
	xpbd_time = 0;
	embedding_time = 0;
	left_time = 0;
	right_time = 0;
	shadow_time = 0;
	rendering_right = false;

	global_path = filepath;
	FLOG_FLUSH();

	gs_objects.clear();
	total_gs = 0;
	total_verts = 0;
	total_edges = 0;
	total_faces = 0;
	total_cells = 0;
	Vector3f _origin_scenemin = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	Vector3f _origin_scenemax = -_origin_scenemin;
	_scenemin = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
	_scenemax = -_scenemin;
	int i = 0;
	while (true) {
		string ply_path = (std::string(filepath) + std::to_string(i) + std::string("_point_cloud.ply")).c_str();
		string tet_path = (std::string(filepath) + std::to_string(i) + std::string("_tetgen.txt")).c_str();
		if (std::filesystem::exists(ply_path.c_str())) {
			gs_objects.push_back(GSObject());

			// Load the PLY data (AoS) to the GPU (SoA)
			if (_sh_degree == 1) {
				gs_objects[i].num_gs = loadPly<1>(ply_path.c_str(), gs_objects[i].pos, gs_objects[i].shs, gs_objects[i].opacity, gs_objects[i].scale, gs_objects[i].rot, _origin_scenemin, _origin_scenemax);
			}
			else if (_sh_degree == 2) {
				gs_objects[i].num_gs = loadPly<2>(ply_path.c_str(), gs_objects[i].pos, gs_objects[i].shs, gs_objects[i].opacity, gs_objects[i].scale, gs_objects[i].rot, _origin_scenemin, _origin_scenemax);
			}
			else if (_sh_degree == 3) {
				gs_objects[i].num_gs = loadPly<3>(ply_path.c_str(), gs_objects[i].pos, gs_objects[i].shs, gs_objects[i].opacity, gs_objects[i].scale, gs_objects[i].rot, _origin_scenemin, _origin_scenemax);
			}

			gs_objects[i].is_rigid = (object_materials[i * material_property_size + 4] > 0);
			const float local_scale = object_materials[i * material_property_size + 3];
			Eigen::Quaternion<float> lq(object_materials[i * material_property_size + 8], 
										object_materials[i * material_property_size + 5], 
										object_materials[i * material_property_size + 6], 
										object_materials[i * material_property_size + 7]);
			Matrix3f local_rotation = lq.toRotationMatrix();
			gs_objects[i].init_rot.resize(gs_objects[i].num_gs, local_rotation * global_rotation);
			FLOG(
				sim_log << lq.x() << " " << lq.y() << " " << lq.z() << " " << lq.w() << "\n";
			);
			for (auto &p : gs_objects[i].pos) {
				Vector3f roted_p = local_rotation * global_rotation * p;
				for (int dim = 0; dim < 3; ++dim)
					p[dim] = (roted_p[dim] * global_scale * local_scale + global_offset[dim]) + object_offsets[i * 3 + dim];
				_scenemin = _scenemin.cwiseMin(p);
				_scenemax = _scenemax.cwiseMax(p);
			}
			for (auto &s : gs_objects[i].scale) {
				for (int dim = 0; dim < 3; ++dim)
					s.scale[dim] *= global_scale * local_scale;
			}
			for (auto &r : gs_objects[i].rot) {
				Eigen::Quaternionf r_q(r.rot[0], r.rot[1], r.rot[2], r.rot[3]); 
				Eigen::Quaternionf roted_r_q = (lq * (global_rotation_q * r_q)).normalized();
				r.rot[0] = roted_r_q.w();
				r.rot[1] = roted_r_q.x();
				r.rot[2] = roted_r_q.y();
				r.rot[3] = roted_r_q.z();
			}

			if (std::filesystem::exists(tet_path.c_str())) {
				gs_objects[i].is_background = false;
				// Local Tetrahedron Mesh Data
				loadTet(tet_path.c_str(), gs_objects[i].mesh_verts, gs_objects[i].mesh_edges, gs_objects[i].mesh_faces, gs_objects[i].mesh_cells);
				gs_objects[i].num_verts = gs_objects[i].mesh_verts.size();
				gs_objects[i].num_edges = gs_objects[i].mesh_edges.size() / 2;
				gs_objects[i].num_faces = gs_objects[i].mesh_faces.size() / 3;
				gs_objects[i].num_cells = gs_objects[i].mesh_cells.size() / 4;

				// Add initial offset for non-background objects
				for (auto &p : gs_objects[i].mesh_verts) {
					Vector3f roted_p = local_rotation * global_rotation * p;
					for (int dim = 0; dim < 3; ++dim)
						p[dim] = (roted_p[dim] * global_scale * local_scale + global_offset[dim]) + object_offsets[i * 3 + dim];
				}
				// Assign verts group (for object-object collision)
				gs_objects[i].mesh_verts_group.resize(gs_objects[i].num_verts, i);
				// rigid_group for rigid body (not same)
				if (gs_objects[i].is_rigid) {
					gs_objects[i].rigid_group = std::vector<int>(gs_objects[i].num_verts, i);
				} else {
					gs_objects[i].rigid_group = std::vector<int>(gs_objects[i].num_verts, -1);
				}

				// Initialize per-object simulation parameters
				float density = object_materials[i * material_property_size + 0];
				if (density <= 1e-5f) {
					// Fixed object will not calculate elasticity, which could optimize performance.
					gs_objects[i].num_cells = 0;
				} else {
					float E = object_materials[i * material_property_size + 1];	// Young's modulus
					float nu = object_materials[i * material_property_size + 2];	// Poisson's ratio: nu \in [0, 0.5)
					float mu = E / (2 * (1 + nu));
					float lambda = E * nu / ((1 + nu) * (1 -2 * nu));
					gs_objects[i].density.resize(gs_objects[i].num_cells, density);
					gs_objects[i].mu.resize(gs_objects[i].num_cells, mu);
					gs_objects[i].lambda.resize(gs_objects[i].num_cells, lambda);
				}
			} else {
				gs_objects[i].is_background = true;
				gs_objects[i].num_verts = 0;
				gs_objects[i].num_edges = 0;
				gs_objects[i].num_faces = 0;
				gs_objects[i].num_cells = 0;
			}
			assert(!(gs_objects[i].is_rigid && gs_objects[i].is_background));

			gs_objects[i].gs_offset = total_gs;
			gs_objects[i].verts_offset = total_verts;
			gs_objects[i].edges_offset = total_edges;
			gs_objects[i].faces_offset = total_faces;
			gs_objects[i].cells_offset = total_cells;
			

			// Increment local EV/FV/CV indices into global indices
			for (auto &d : gs_objects[i].mesh_edges) {
				d += gs_objects[i].verts_offset;
			}

			for (auto &d : gs_objects[i].mesh_faces) {
				d += gs_objects[i].verts_offset;
			}

			for (auto &d : gs_objects[i].mesh_cells) {
				d += gs_objects[i].verts_offset;
			}

			total_gs += gs_objects[i].num_gs;
			total_verts += gs_objects[i].num_verts;
			total_edges += gs_objects[i].num_edges;
			total_faces += gs_objects[i].num_faces;
			total_cells += gs_objects[i].num_cells;

			++i;
		} else {
			break;
		}
	}

	FDUMP("log.obj",
	for (int i = 0; i < gs_objects.size(); i++) {
		for (int k = 0; k < gs_objects[i].num_verts; k++) {
			sim_log << "v " << gs_objects[i].mesh_verts[k][0] << " " << gs_objects[i].mesh_verts[k][1] << " " << gs_objects[i].mesh_verts[k][2] << "\n";
		}
	}

	for (int i = 0; i < gs_objects.size(); i++) {
		for (int k = 0; k < gs_objects[i].num_faces; k++) {
			sim_log << "f " << gs_objects[i].mesh_faces[k*3+0]+1 << " " << gs_objects[i].mesh_faces[k*3+1]+1 << " " << gs_objects[i].mesh_faces[k*3+2]+1 << "\n";
		}
	});
	
	// Gaussian Splatting Data
	reset_and_zero(pos_d, sizeof(float) * 3, total_gs);
	reset_and_zero(rot_d, sizeof(float) * 4, total_gs);
	reset_and_zero(init_rot_d, sizeof(float) * 9, total_gs);
	reset_and_zero(local_rot_d, sizeof(float) * 9, total_gs);
	reset_and_zero(shs_d, sizeof(SHs<3>), total_gs);
	reset_and_zero(pre_color_d, sizeof(float) * 3, total_gs);
	reset_and_zero(opacity_d, sizeof(float), total_gs);
	reset_and_zero(scale_d, sizeof(float) * 3, total_gs);

	reset_and_zero(view_d, sizeof(Matrix4f), 1);
	reset_and_zero(proj_d, sizeof(Matrix4f), 1);
	reset_and_zero(proj_inv_d, sizeof(Matrix4f), 1);
	reset_and_zero(cam_pos_d, 3 * sizeof(float), 1);
	reset_and_zero(lighting_view_d, sizeof(Matrix4f), 1);
	reset_and_zero(lighting_proj_d, sizeof(Matrix4f), 1);
	reset_and_zero(lighting_cam_pos_d, 3 * sizeof(float), 1);
	reset_and_zero(background_d, 3 * sizeof(float), 1);
	reset_and_zero(rect_d, 2 * sizeof(int), total_gs);

	geomBufferFunc = resizeFunctional(&geomPtr, allocdGeom);
	binningBufferFunc = resizeFunctional(&binningPtr, allocdBinning);
	imgBufferFunc = resizeFunctional(&imgPtr, allocdImg);

	// Load Mesh Data
	reset_and_zero(verts_X_d, sizeof(float) * 3, total_verts);
	reset_and_zero(verts_x_d, sizeof(float) * 3, total_verts);
	reset_and_zero(verts_group_d, sizeof(int),   total_verts);
	// Mesh Indices (CUDA)
	reset_and_zero(edges_ind_d, sizeof(int) * 2, total_edges);
	reset_and_zero(faces_ind_d, sizeof(int) * 3, total_faces);
	reset_and_zero(cells_ind_d, sizeof(int) * 4, total_cells);

	// FEM-PBD Simulation Data
	reset_and_zero(verts_v_d, sizeof(float) * 3, total_verts);
	reset_and_zero(verts_f_d, sizeof(float) * 3, total_verts);
	reset_and_zero(verts_m_d, sizeof(float), total_verts);
	reset_and_zero(verts_inv_m_d, sizeof(float), total_verts);
	reset_and_zero(verts_new_x_d, sizeof(float) * 3, total_verts);
	reset_and_zero(verts_dp_d, sizeof(float) * 3, total_verts);
	reset_and_zero(verts_selected_d, sizeof(int), total_verts);
	reset_and_zero(rigid_verts_group_d, sizeof(int), total_verts);
	reset_and_zero(cells_multiplier_d, sizeof(float), total_cells);
	reset_and_zero(cells_DS_inv_d, sizeof(float) * 9, total_cells);
	reset_and_zero(cells_vol0_d, sizeof(float), total_cells);
	reset_and_zero(cells_density_d, sizeof(float), total_cells);
	reset_and_zero(cells_mu_d, sizeof(float), total_cells);
	reset_and_zero(cells_lambda_d, sizeof(float), total_cells);

	// Rigid Simulation Data
	reset_and_zero(rigid_m_d, sizeof(double), gs_objects.size());
	reset_and_zero(rigid_cm0_d, sizeof(double) * 3, gs_objects.size());
	reset_and_zero(rigid_cm_d, sizeof(double) * 3, gs_objects.size());
	reset_and_zero(rigid_A_d, sizeof(double) * 9, gs_objects.size());
	reset_and_zero(rigid_R_d, sizeof(double) * 9, gs_objects.size());

	// Collision Data
	// LBVH Data
	reset_and_zero(tri_aabbs, sizeof(lbvh::aabb<float>), total_faces);
	reset_and_zero(partial_aabb_d, sizeof(lbvh::aabb<float>), 64);
	partial_aabb_h = (lbvh::aabb<float>*)malloc(sizeof(lbvh::aabb<float>) * 64);
	assert(partial_aabb_h != nullptr);
	reset_and_zero(morton_code, sizeof(uint64_t), total_faces);
	reset_and_zero(sorted_morton_code, sizeof(uint64_t), total_faces);
	reset_and_zero(indices, sizeof(int), total_faces);
	reset_and_zero(sorted_indices, sizeof(int), total_faces);
	reset_and_zero(sorted_tri_aabbs, sizeof(lbvh::aabb<float>), total_faces);
	reset_and_zero(flags, sizeof(int), total_faces);

	size_t bvh_size = total_faces * 2 - 1;
	reset_and_zero(aabbs, sizeof(lbvh::aabb<float>), bvh_size);
	reset_and_zero(nodes, sizeof(lbvh::Node), bvh_size);

	sort_buffer_size = 0;
	radix_sort(sorted_morton_code, morton_code, sorted_indices, indices, nullptr, sort_buffer_size, total_faces);
	reset_and_zero(sort_buffer, sizeof(unsigned int), sort_buffer_size);

	// Board Phase Culling Data
	reset_and_zero(collision_pairs, sizeof(int2), max_collision_pairs);
	reset_and_zero(total_pairs_d, sizeof(int), 1);

	// Narrow Phase Detection Data
	reset_and_zero(exact_collision_pairs, sizeof(int4), max_collision_pairs);
	reset_and_zero(total_exact_pairs_d, sizeof(int), 1);

	// Interpolation Data
	reset_and_zero(cov_d, sizeof(float) * 9, total_gs);
	reset_and_zero(local_tet_x, sizeof(float) * 12, total_gs);
	reset_and_zero(local_tet_w, sizeof(float) * 3, total_gs);
	reset_and_zero(global_tet_idx, sizeof(int) * 4, total_gs);
	reset_and_zero(global_tet_w, sizeof(float) * 12, total_gs);

	// Controller Data
	reset_and_zero(left.controller_pos_d, sizeof(float) * 3, 1);
	reset_and_zero(left.controller_vel_d, sizeof(float) * 3, 1);
	reset_and_zero(left.controller_ang_vel_d, sizeof(float) * 9, 1);
	reset_and_zero(right.controller_pos_d, sizeof(float) * 3, 1);
	reset_and_zero(right.controller_vel_d, sizeof(float) * 3, 1);
	reset_and_zero(right.controller_ang_vel_d, sizeof(float) * 9, 1);

	// Quasi Simulation Data
	reset_and_zero(quasi_verts_d, sizeof(float) * 3, total_verts);
	reset_and_zero(quasi_faces_ind_d, sizeof(int) * 3, total_faces);
	

	// Copying data from host to device
#if (BENCHMARK<=10)
	bool white_bg = true;
#else
	bool white_bg = false;
#endif
	float bg[3] = { white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f, white_bg ? 1.f : 0.f };
	CUDA_SAFE_CALL_ALWAYS(cudaMemcpy(background_d, bg, 3 * sizeof(float), cudaMemcpyHostToDevice));

	for (int i = 0; i < gs_objects.size(); i++) {
		CUDA_SAFE_CALL(cudaMemcpy(pos_d + gs_objects[i].gs_offset * 3, gs_objects[i].pos.data(), gs_objects[i].num_gs * sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(rot_d + gs_objects[i].gs_offset * 4, gs_objects[i].rot.data(), gs_objects[i].num_gs * sizeof(float) * 4, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(shs_d + gs_objects[i].gs_offset * sizeof(SHs<3>) / sizeof(float), gs_objects[i].shs.data(), gs_objects[i].num_gs * sizeof(SHs<3>), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(opacity_d + gs_objects[i].gs_offset, gs_objects[i].opacity.data(), gs_objects[i].num_gs * sizeof(float), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(scale_d + gs_objects[i].gs_offset * 3, gs_objects[i].scale.data(), gs_objects[i].num_gs * sizeof(float) * 3, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(init_rot_d + gs_objects[i].gs_offset * 9, gs_objects[i].init_rot.data(), gs_objects[i].num_gs * sizeof(float) * 9, cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < gs_objects.size(); i++) {
		if (!gs_objects[i].is_background) {
			CUDA_SAFE_CALL(cudaMemcpy(verts_X_d + gs_objects[i].verts_offset * 3, gs_objects[i].mesh_verts.data(), gs_objects[i].num_verts * sizeof(float) * 3, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(verts_x_d + gs_objects[i].verts_offset * 3, gs_objects[i].mesh_verts.data(), gs_objects[i].num_verts * sizeof(float) * 3, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(verts_group_d + gs_objects[i].verts_offset, gs_objects[i].mesh_verts_group.data(), gs_objects[i].num_verts * sizeof(int), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(rigid_verts_group_d + gs_objects[i].verts_offset, gs_objects[i].rigid_group.data(), gs_objects[i].num_verts * sizeof(int), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(edges_ind_d + gs_objects[i].edges_offset * 2, gs_objects[i].mesh_edges.data(), gs_objects[i].num_edges * sizeof(int) * 2, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(faces_ind_d + gs_objects[i].faces_offset * 3, gs_objects[i].mesh_faces.data(), gs_objects[i].num_faces * sizeof(int) * 3, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cells_ind_d + gs_objects[i].cells_offset * 4, gs_objects[i].mesh_cells.data(), gs_objects[i].num_cells * sizeof(int) * 4, cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cells_density_d + gs_objects[i].cells_offset, gs_objects[i].density.data(), gs_objects[i].num_cells * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cells_mu_d + gs_objects[i].cells_offset, gs_objects[i].mu.data(), gs_objects[i].num_cells * sizeof(float), cudaMemcpyHostToDevice));
			CUDA_SAFE_CALL(cudaMemcpy(cells_lambda_d + gs_objects[i].cells_offset, gs_objects[i].lambda.data(), gs_objects[i].num_cells * sizeof(float), cudaMemcpyHostToDevice));
		}
	}

	// Interpolation Initialization
	sim::initialize_covariance(total_gs, scale_d, rot_d, cov_d);
	sim::get_local_embeded_tets(total_gs, pos_d, cov_d, local_tet_x, local_tet_w, disable_LG_interp);
	CUDA_SAFE_CALL(cudaMemset(global_tet_idx, 0xFF, sizeof(int)*4*total_gs));

	for (int i = 0; i < gs_objects.size(); i++) {
		if (gs_objects[i].is_background) {
			continue;
		}

		for (int t_id = gs_objects[i].cells_offset; 
				 t_id < gs_objects[i].cells_offset + gs_objects[i].num_cells; 
				 ++t_id) {
			sim::get_global_embeded_tet(gs_objects[i].num_gs, 
										gs_objects[i].gs_offset, 
										t_id, 
										verts_X_d, 
										cells_ind_d, 
										local_tet_x, 
										global_tet_idx, 
										global_tet_w);
		}

		float density = object_materials[i * material_property_size + 0];
		// If is not a fixed object, deactivate gs that not embeded in cage mesh
		if (density > 1e-5f) {
			sim::deactivate_opacity(gs_objects[i].num_gs, gs_objects[i].gs_offset, opacity_d, global_tet_idx);
		}
	}

	// PBD-FEM Initialization
	sim::init_FEM_bases(total_cells, cells_density_d, cells_ind_d, verts_X_d, verts_m_d, cells_DS_inv_d, cells_vol0_d);
	sim::init_inv_mass(total_verts, verts_X_d, verts_m_d, verts_inv_m_d, boundary);

	// Rigid Initialization
	sim::init_rigid(total_verts, rigid_m_d, rigid_cm0_d, rigid_verts_group_d, verts_X_d, verts_m_d);
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	FLOG(
		sim_log << "N(GS) = " << total_gs << "\n";
		sim_log << "N(Vert) = " << total_verts << "\n";
		sim_log << "N(Face) = " << total_faces << "\n";
		sim_log << "N(Elem) = " << total_cells << "\n";
	);
}

void GaussianSplatting::RenderShadowMap() {
	if (!sm_buffer_initialized || lighting_H != last_lighting_H || lighting_W != last_lighting_W) {
		reset_and_zero(sm_image_d, sizeof(float) * 3, lighting_H * lighting_W);
		reset_and_zero(sm_depth_d, sizeof(float), lighting_H * lighting_W);
		reset_and_zero(sm_alpha_d, sizeof(float), lighting_H * lighting_W);

		last_lighting_H = lighting_H;
		last_lighting_W = lighting_W;
		sm_buffer_initialized = true;
	}

	float aspect_ratio = (float)lighting_W / (float)lighting_H;
	float tan_fovy = tan(lighting_fovy * 0.5f);
	float tan_fovx = tan_fovy * aspect_ratio;

	CUDA_SAFE_CALL(cudaMemcpy(lighting_view_d, lighting_view_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(lighting_proj_d, lighting_proj_view_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(lighting_cam_pos_d, &lighting_pos, sizeof(float) * 3, cudaMemcpyHostToDevice));

	PROFILE(shadow_time, CudaRasterizer::Rasterizer::forward(
		geomBufferFunc,
		binningBufferFunc,
		imgBufferFunc,
		total_gs, _sh_degree, 16,
		background_d,
		lighting_W, lighting_H,
		pos_d,
		shs_d,
		nullptr,
		opacity_d,
		scale_d,
		_scalingModifier,
		rot_d,
		nullptr,
		lighting_view_d,
		lighting_proj_d,
		lighting_cam_pos_d,
		tan_fovx,
		tan_fovy,
		false,
		sm_image_d,
		sm_depth_d,
		sm_alpha_d,
		nullptr,
		false
	));

	CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

void GaussianSplatting::RenderImage(float* image_d, float* depth_d, float* alpha_d, Matrix4f view_mat, Matrix4f proj_mat, Vector3f position, float fovy, int width, int height) {
	float aspect_ratio = (float)width / (float)height;
	float tan_fovy = tan(fovy * 0.5f);
	float tan_fovx = tan_fovy * aspect_ratio;

	CUDA_SAFE_CALL(cudaMemcpy(view_d, view_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(proj_d, proj_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cam_pos_d, &position, sizeof(float) * 3, cudaMemcpyHostToDevice));

#if BENCHMARK == 0 // Stool
	tan_fovx = 0.36000002589322094;
	tan_fovy = 0.36000002589322094;
	float view_mat_data[16] = {
		-6.9772e-01, -3.6951e-02, -7.1541e-01,  0.0000e+00,
		7.1637e-01, -3.5989e-02, -6.9679e-01,  0.0000e+00,
		-3.2221e-09, -9.9867e-01,  5.1581e-02,  0.0000e+00,
		0.0000e+00, -0.0000e+00,  2.0000e+00,  1.0000e+00
	};
	float proj_mat_data[16] = {
		-1.9381e+00, -1.0264e-01, -7.1549e-01, -7.1541e-01,
		1.9899e+00, -9.9970e-02, -6.9686e-01, -6.9679e-01,
		-8.9502e-09, -2.7741e+00,  5.1586e-02,  5.1581e-02,
		0.0000e+00,  0.0000e+00,  1.9902e+00,  2.0000e+00
	};
	float campos[3] = {1.4308,  1.3936, -0.1032};
	CUDA_SAFE_CALL(cudaMemcpy(view_d, view_mat_data, sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(proj_d, proj_mat_data, sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cam_pos_d, campos, sizeof(float) * 3, cudaMemcpyHostToDevice));
#elif BENCHMARK == 1 // Chair
	tan_fovx = 0.36000002589322094;
	tan_fovy = 0.36000002589322094;
	float view_mat_data[16] = {
		-9.2501e-01, -2.7489e-01,  2.6227e-01,  0.0000e+00,
		-3.7993e-01,  6.6927e-01, -6.3854e-01,  0.0000e+00,
		-1.1369e-09, -6.9030e-01, -7.2352e-01,  0.0000e+00,
		-5.3721e-08,  2.0115e-08,  4.0311e+00,  1.0000e+00
	};
	float proj_mat_data[16] = {
		-2.5695e+00, -7.6358e-01,  2.6229e-01,  2.6227e-01,
		-1.0554e+00,  1.8591e+00, -6.3860e-01, -6.3854e-01,
		-3.1582e-09, -1.9175e+00, -7.2359e-01, -7.2352e-01,
		-1.4922e-07,  5.5874e-08,  4.0215e+00,  4.0311e+00
	};
	float campos[3] = {-1.0572, 2.5740, 2.9166};
	CUDA_SAFE_CALL(cudaMemcpy(view_d, view_mat_data, sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(proj_d, proj_mat_data, sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cam_pos_d, campos, sizeof(float) * 3, cudaMemcpyHostToDevice));
#elif BENCHMARK == 2 // Materials
	tan_fovx = 0.31999998709328725;
	tan_fovy = 0.31999998709328725;
	float view_mat_data[16] = {
		9.9898e-01,  2.9578e-02, -3.4070e-02,  0.0000e+00,
		4.5118e-02, -6.5491e-01,  7.5436e-01,  0.0000e+00,
		2.2382e-10, -7.5513e-01, -6.5557e-01,  0.0000e+00,
		-4.4743e-09,  5.9848e-08,  4.0311e+00, 1.0000e+00
	};
	float proj_mat_data[16] = {
		3.1218e+00,  9.2431e-02, -3.4073e-02, -3.4070e-02,
		1.4099e-01, -2.0466e+00,  7.5444e-01,  7.5436e-01,
		6.9942e-10, -2.3598e+00, -6.5564e-01, -6.5557e-01,
		-1.3982e-08,  1.8703e-07,  4.0215e+00,  4.0311e+00
	};
	float campos[3] = {0.1373, -3.0409, 2.6427};
	CUDA_SAFE_CALL(cudaMemcpy(view_d, view_mat_data, sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(proj_d, proj_mat_data, sizeof(Matrix4f), cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(cam_pos_d, campos, sizeof(float) * 3, cudaMemcpyHostToDevice));
#else
	// Do Nothing
#endif

	sim::convert_SH(total_gs, cam_pos_d, shs_d, pos_d, local_rot_d, init_rot_d, pre_color_d);

	long long *tm = rendering_right ? &right_time : &left_time;
	rendering_right = !rendering_right;
	if (use_shadow_map) {
		Matrix4f lighting_view_inv_mat = lighting_view_mat * proj_mat.inverse();
		Matrix4f lighting_proj_inv_mat = lighting_proj_view_mat * proj_mat.inverse();
		CUDA_SAFE_CALL(cudaMemcpy(lighting_view_d, lighting_view_inv_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(lighting_proj_d, lighting_proj_inv_mat.data(), sizeof(Matrix4f), cudaMemcpyHostToDevice));

		PROFILE(*tm, CudaRasterizer::Rasterizer::forward(
		geomBufferFunc,
		binningBufferFunc,
		imgBufferFunc,
		total_gs, _sh_degree, 16,
		background_d,
		width, height,
		pos_d,
		shs_d,
		pre_color_d,
		opacity_d,
		scale_d,
		_scalingModifier,
		rot_d,
		nullptr,
		view_d,
		proj_d,
		cam_pos_d,
		tan_fovx,
		tan_fovy,
		false,
		image_d,
		depth_d,
		alpha_d,
		nullptr,
		false,

		lighting_W, lighting_H,
		sm_depth_d,
		lighting_view_d,
		lighting_proj_d,
		shadow_eps,
		shadow_factor
		));
	} else {
		PROFILE(*tm, CudaRasterizer::Rasterizer::forward(
		geomBufferFunc,
		binningBufferFunc,
		imgBufferFunc,
		total_gs, _sh_degree, 16,
		background_d,
		width, height,
		pos_d,
		shs_d,
		pre_color_d,
		opacity_d,
		scale_d,
		_scalingModifier,
		rot_d,
		nullptr,
		view_d,
		proj_d,
		cam_pos_d,
		tan_fovx,
		tan_fovy,
		false,
		image_d,
		depth_d,
		alpha_d,
		nullptr,
		false
		));
	}

	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	int frame = global_step - 1;
	if (enable_export_frames && frame < max_export_sequence) {
		std::ostringstream fname;
		fname << "images/frame_" << std::setw(4) << std::setfill('0') << frame << ".png";
		save_image(image_d, width, height, (global_path + fname.str()).c_str());
		// NOTE(changyu): adhoc for gaolin data
		Save();
	}
}

void GaussianSplatting::Save() {
	int *faces_ind_cpu = (int*)malloc(sizeof(int) * 3 * total_faces);
	CUDA_SAFE_CALL(cudaMemcpy(faces_ind_cpu, faces_ind_d, sizeof(int) * 3 * total_faces, cudaMemcpyDeviceToHost));
	float *verts_x_cpu = (float*)malloc(sizeof(float) * 3 * total_verts);
	CUDA_SAFE_CALL(cudaMemcpy(verts_x_cpu, verts_x_d, sizeof(float) * 3 * total_verts, cudaMemcpyDeviceToHost));

	int frame = global_step - 1;
	std::ostringstream fname;
	fname << "sequences/frame_" << std::setw(4) << std::setfill('0') << frame;
	FDUMP((fname.str() + ".obj").c_str(),
		for (int i = 0; i < total_verts; i++) {
			sim_log << "v " << verts_x_cpu[i*3+0] << " " 
							<< verts_x_cpu[i*3+1] << " " 
							<< verts_x_cpu[i*3+2] << "\n";
		}

		for (int i = 0; i < total_faces; i++) {
			sim_log << "f " << faces_ind_cpu[i*3+0]+1 
					 << " " << faces_ind_cpu[i*3+1]+1 
					 << " " << faces_ind_cpu[i*3+2]+1 << "\n";
		});
	int g_begin = 0;
	int g_end = total_gs;
	int count = g_end - g_begin;
	std::vector<Pos> pos(count);
	std::vector<Rot> rot(count);
	std::vector<float> opacity(count);
	std::vector<SHs<3>> shs(count);
	std::vector<Scale> scale(count);
	CUDA_SAFE_CALL(cudaMemcpy(pos.data(), pos_d + g_begin, sizeof(Pos) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(rot.data(), rot_d + g_begin, sizeof(Rot) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(opacity.data(), opacity_d + g_begin, sizeof(float) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(shs.data(), shs_d + g_begin, sizeof(SHs<3>) * count, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(scale.data(), scale_d + g_begin, sizeof(Scale) * count, cudaMemcpyDeviceToHost));
	savePly((global_path + fname.str() + ".ply").c_str(), pos, shs, opacity, scale, rot);
}