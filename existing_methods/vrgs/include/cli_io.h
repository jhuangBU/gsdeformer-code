#ifndef CLI_IO_H
#define CLI_IO_H

#include <set>
#include <utility>
#include <algorithm>
#include <vector>
#include <sstream>  
#include <stdexcept>
#include <random>
#include <cfloat>

#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include "SimKernels.h"

using namespace std;

// region misc math

inline float sigmoid(const float m1) { return 1.0f / (1.0f + exp(-m1)); }
inline float inverse_sigmoid(const float m1) { return log(m1 / (1.0f - m1)); }

// endregion

// region file IO

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

// endregion

// region CUDA IO

struct TetCUDA {
    float* d_verts;      // vertex positions (x,y,z interleaved)
    int* d_edges;        // edge vertex indices
    int* d_faces;        // face vertex indices
    int* d_cells;        // cell vertex indices
    int num_verts;
    int num_edges;
    int num_faces;
    int num_cells;

    TetCUDA() : d_verts(nullptr), d_edges(nullptr), d_faces(nullptr), d_cells(nullptr),
                num_verts(0), num_edges(0), num_faces(0), num_cells(0) {}

    ~TetCUDA() {
        if (d_verts) CUDA_SAFE_CALL(cudaFree(d_verts));
        if (d_edges) CUDA_SAFE_CALL(cudaFree(d_edges));
        if (d_faces) CUDA_SAFE_CALL(cudaFree(d_faces));
        if (d_cells) CUDA_SAFE_CALL(cudaFree(d_cells));
    }

    // Prevent copying
    TetCUDA(const TetCUDA&) = delete;
    TetCUDA& operator=(const TetCUDA&) = delete;

    // Allow moving
    TetCUDA(TetCUDA&& other) noexcept {
        *this = std::move(other);
    }

    TetCUDA& operator=(TetCUDA&& other) noexcept {
        if (this != &other) {
            d_verts = other.d_verts;
            d_edges = other.d_edges;
            d_faces = other.d_faces;
            d_cells = other.d_cells;
            num_verts = other.num_verts;
            num_edges = other.num_edges;
            num_faces = other.num_faces;
            num_cells = other.num_cells;

            other.d_verts = nullptr;
            other.d_edges = nullptr;
            other.d_faces = nullptr;
            other.d_cells = nullptr;
        }
        return *this;
    }
};

TetCUDA loadTetCUDA(const std::string& filename) {
    // First load into CPU vectors
    std::vector<Pos> verts;
    std::vector<int> edges;
    std::vector<int> faces;
    std::vector<int> cells;
    loadTet(filename, verts, edges, faces, cells);

    // Create and populate CUDA struct
    TetCUDA tet;
    tet.num_verts = verts.size();
    tet.num_edges = edges.size() / 2;  // edges stored as pairs
    tet.num_faces = faces.size() / 3;  // faces stored as triplets
    tet.num_cells = cells.size() / 4;  // cells stored as quads

    // Allocate and copy vertex data
    CUDA_SAFE_CALL(cudaMalloc(&tet.d_verts, tet.num_verts * 3 * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(tet.d_verts, verts.data(), tet.num_verts * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate and copy edge data
    CUDA_SAFE_CALL(cudaMalloc(&tet.d_edges, edges.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(tet.d_edges, edges.data(), edges.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and copy face data
    CUDA_SAFE_CALL(cudaMalloc(&tet.d_faces, faces.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(tet.d_faces, faces.data(), faces.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate and copy cell data
    CUDA_SAFE_CALL(cudaMalloc(&tet.d_cells, cells.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(tet.d_cells, cells.data(), cells.size() * sizeof(int), cudaMemcpyHostToDevice));

    return tet;
}

struct GSCUDA {
    float3* d_pos;
    float* d_shs;     // Flattened SH coefficients
    float* d_opacities;
    float4* d_scales; // xyz scales + padding
    float4* d_rot;    // quaternion rotation
    size_t count;
    Vector3f scene_min;
    Vector3f scene_max;

    // Constructor
    GSCUDA() : d_pos(nullptr), d_shs(nullptr), d_opacities(nullptr), 
               d_scales(nullptr), d_rot(nullptr), count(0) {}

    // Destructor
    ~GSCUDA() {
        if (d_pos) CUDA_SAFE_CALL(cudaFree(d_pos));
        if (d_shs) CUDA_SAFE_CALL(cudaFree(d_shs));
        if (d_opacities) CUDA_SAFE_CALL(cudaFree(d_opacities));
        if (d_scales) CUDA_SAFE_CALL(cudaFree(d_scales));
        if (d_rot) CUDA_SAFE_CALL(cudaFree(d_rot));
    }

    // Move constructor
    GSCUDA(GSCUDA&& other) noexcept {
        d_pos = other.d_pos;
        d_shs = other.d_shs;
        d_opacities = other.d_opacities;
        d_scales = other.d_scales;
        d_rot = other.d_rot;
        count = other.count;
        scene_min = other.scene_min;
        scene_max = other.scene_max;

        other.d_pos = nullptr;
        other.d_shs = nullptr;
        other.d_opacities = nullptr;
        other.d_scales = nullptr;
        other.d_rot = nullptr;
    }

    // Move assignment
    GSCUDA& operator=(GSCUDA&& other) noexcept {
        if (this != &other) {
            // Free existing resources
            if (d_pos) CUDA_SAFE_CALL(cudaFree(d_pos));
            if (d_shs) CUDA_SAFE_CALL(cudaFree(d_shs));
            if (d_opacities) CUDA_SAFE_CALL(cudaFree(d_opacities));
            if (d_scales) CUDA_SAFE_CALL(cudaFree(d_scales));
            if (d_rot) CUDA_SAFE_CALL(cudaFree(d_rot));

            // Move resources
            d_pos = other.d_pos;
            d_shs = other.d_shs;
            d_opacities = other.d_opacities;
            d_scales = other.d_scales;
            d_rot = other.d_rot;
            count = other.count;
            scene_min = other.scene_min;
            scene_max = other.scene_max;

            other.d_pos = nullptr;
            other.d_shs = nullptr;
            other.d_opacities = nullptr;
            other.d_scales = nullptr;
            other.d_rot = nullptr;
        }
        return *this;
    }

    // Delete copy constructor and assignment
    GSCUDA(const GSCUDA&) = delete;
    GSCUDA& operator=(const GSCUDA&) = delete;
};

GSCUDA loadPlyCUDA(const char* filename) {
    Vector3f origin_scenemin(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3f origin_scenemax = -origin_scenemin;

    // Temporary host vectors
    std::vector<Pos> h_pos;
    std::vector<SHs<3>> h_shs;
    std::vector<float> h_opacities;
    std::vector<Scale> h_scales;
    std::vector<Rot> h_rot;

    // Load data to host vectors
    int count = loadPly<3>(filename, h_pos, h_shs, h_opacities, h_scales, h_rot, 
                          origin_scenemin, origin_scenemax);

    GSCUDA result;
    result.count = count;
    result.scene_min = origin_scenemin;
    result.scene_max = origin_scenemax;

    // Allocate and copy position data
    CUDA_SAFE_CALL(cudaMalloc(&result.d_pos, count * sizeof(Pos)));
    CUDA_SAFE_CALL(cudaMemcpy(result.d_pos, h_pos.data(), count * sizeof(float3), cudaMemcpyHostToDevice));

    // Allocate and copy SH coefficients (48 floats per SH for degree 3)
    CUDA_SAFE_CALL(cudaMalloc(&result.d_shs, count * sizeof(SHs<3>)));
    CUDA_SAFE_CALL(cudaMemcpy(result.d_shs, h_shs.data(), count * 48 * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate and copy opacity data
    CUDA_SAFE_CALL(cudaMalloc(&result.d_opacities, count * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(result.d_opacities, h_opacities.data(), count * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate and copy scale data
    CUDA_SAFE_CALL(cudaMalloc(&result.d_scales, count * sizeof(float3)));
    CUDA_SAFE_CALL(cudaMemcpy(result.d_scales, h_scales.data(), count * sizeof(float3), cudaMemcpyHostToDevice));

    // Allocate and copy rotation data
    CUDA_SAFE_CALL(cudaMalloc(&result.d_rot, count * sizeof(Rot)));
    CUDA_SAFE_CALL(cudaMemcpy(result.d_rot, h_rot.data(), count * sizeof(float4), cudaMemcpyHostToDevice));

    return result;
}

void savePlyCUDA(const char* filename, const GSCUDA& gs) {
    std::vector<Pos> h_pos(gs.count);
    std::vector<SHs<3>> h_shs(gs.count);
    std::vector<float> h_opacities(gs.count);
    std::vector<Scale> h_scales(gs.count);
    std::vector<Rot> h_rot(gs.count);

    CUDA_SAFE_CALL(cudaMemcpy(h_pos.data(), gs.d_pos, gs.count * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_shs.data(), gs.d_shs, gs.count * 48 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_opacities.data(), gs.d_opacities, gs.count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_scales.data(), gs.d_scales, gs.count * sizeof(float3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(h_rot.data(), gs.d_rot, gs.count * sizeof(float4), cudaMemcpyDeviceToHost));

    savePly(filename, h_pos, h_shs, h_opacities, h_scales, h_rot);
}

#endif
