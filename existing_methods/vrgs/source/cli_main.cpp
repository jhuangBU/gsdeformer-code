#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <set>
#include <utility>
#include <algorithm>
#include <vector>
#include <sstream>  
#include <stdexcept>
#include <random> 

// Project includes after CUDA headers
#include "GaussianSplatting.h"
#include "CudaKernels.h"
#include "SimKernels.h"

#include "cli_io.h"

// region arguments

struct Arguments {
    std::string source_ply;
    std::string source_tet_mesh;
    std::string dest_tet_mesh;
    std::string output_ply;
};

Arguments parseArguments(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <source_ply> <source_tet_mesh> <dest_tet_mesh> <output_ply>" << std::endl;
        std::cout << "Arguments:" << std::endl;
        std::cout << "  source_ply      - Input PLY file with Gaussian splats" << std::endl;
        std::cout << "  source_tet_mesh - Source tetrahedral mesh" << std::endl;
        std::cout << "  dest_tet_mesh   - Destination tetrahedral mesh" << std::endl;
        std::cout << "  output_ply      - Output PLY file for transformed splats" << std::endl;
        exit(1);
    }
    
    Arguments args;
    args.source_ply = argv[1];
    args.source_tet_mesh = argv[2];
    args.dest_tet_mesh = argv[3];
    args.output_ply = argv[4];
    
    return args;
}

// endregion

// region main

int main(int argc, char* argv[]) {
    Arguments args = parseArguments(argc, argv);

    // Load source PLY file
    GSCUDA gs = loadPlyCUDA(args.source_ply.c_str());

    // Load source tet mesh
    TetCUDA src_tet = loadTetCUDA(args.source_tet_mesh.c_str());

    // Load destination tet mesh
    TetCUDA dst_tet = loadTetCUDA(args.dest_tet_mesh.c_str());  

    // 1. Initialize covariance matrices for each Gaussian
    float* cov_d;
    CUDA_SAFE_CALL(cudaMalloc(&cov_d, sizeof(float) * 9 * gs.count));
    sim::initialize_covariance(
        gs.count, 
        reinterpret_cast<float*>(gs.d_scales), 
        reinterpret_cast<float*>(gs.d_rot), 
        cov_d
    );

    // 2. Get local tetrahedral embeddings
    float* local_tet_x;
    float* local_tet_w;
    CUDA_SAFE_CALL(cudaMalloc(&local_tet_x, sizeof(float) * 12 * gs.count));
    CUDA_SAFE_CALL(cudaMalloc(&local_tet_w, sizeof(float) * 3 * gs.count));
    sim::get_local_embeded_tets(gs.count, reinterpret_cast<float*>(gs.d_pos), 
                            cov_d, local_tet_x, local_tet_w, 0);

    // 3. Get global tetrahedral embeddings
    int* global_tet_idx;
    float* global_tet_w;
    CUDA_SAFE_CALL(cudaMalloc(&global_tet_idx, sizeof(int) * 4 * gs.count));
    CUDA_SAFE_CALL(cudaMalloc(&global_tet_w, sizeof(float) * 12 * gs.count));
    CUDA_SAFE_CALL(cudaMemset(global_tet_idx, 0xFF, sizeof(int) * 4 * gs.count));

    for (int t_id = 0; t_id < src_tet.num_cells; ++t_id) {
        sim::get_global_embeded_tet(
            gs.count, 0, t_id,
            src_tet.d_verts, src_tet.d_cells,
            local_tet_x, global_tet_idx, global_tet_w
        );
    }

    // 4. Apply interpolation using destination tet mesh
    auto disable_LG_interp = 0;
    float* local_rot_d = nullptr;
    reset_and_zero(local_rot_d, sizeof(float) * 9, gs.count);
    sim::apply_interpolation(
        gs.count,
        reinterpret_cast<float*>(gs.d_pos), reinterpret_cast<float*>(gs.d_scales), reinterpret_cast<float*>(gs.d_rot), cov_d, 
        src_tet.d_cells, src_tet.d_verts, dst_tet.d_verts,
        global_tet_idx, global_tet_w, local_tet_w,
        disable_LG_interp, local_rot_d
    );

    // Free temporary CUDA memory
    CUDA_SAFE_CALL(cudaFree(cov_d));
    CUDA_SAFE_CALL(cudaFree(local_tet_x));
    CUDA_SAFE_CALL(cudaFree(local_tet_w)); 
    CUDA_SAFE_CALL(cudaFree(global_tet_idx));
    CUDA_SAFE_CALL(cudaFree(global_tet_w));

    // Save transformed PLY file
    savePlyCUDA(args.output_ply.c_str(), gs);

    return 0;

}

// endregion