#include "forward_shadow.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderWithShadowMapCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ depths,
	const float* __restrict__ depths_proj,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ out_alpha,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,

	int SM_W, int SM_H,
	const float* __restrict__ shadow_map,
	const float* __restrict__ proj_inv_light_viewmatrix,
	const float* __restrict__ proj_inv_light_projmatrix,
	const float shadow_eps,
	const float shadow_factor)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float weight = 0;
	float D = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0008f)
			{
				done = true;
				continue;
			}

			// NOTE(changyu): Shadow map part
			float3 eye_sreen_coord = { pix2Ndc(pixf.x, W), pix2Ndc(pixf.y, H), depths_proj[collected_id[j]] };
			float4 p_hom = transformPoint4x4(eye_sreen_coord, proj_inv_light_projmatrix);
			float p_w = 1.0f / (p_hom.w + 0.0000001f);
			float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
			float2 light_screen_coord = { ndc2Pix(p_proj.x, SM_W), ndc2Pix(p_proj.y, SM_H) };
			int2 sm_coord = { int(light_screen_coord.x), int(light_screen_coord.y) };

			float final_factor = 1.0f;
			if (sm_coord.x >= 0 && sm_coord.x < SM_W && sm_coord.y >= 0 && sm_coord.y < SM_H) {
				float view_z = 
				(proj_inv_light_viewmatrix[2] * eye_sreen_coord.x + 
				 proj_inv_light_viewmatrix[6] * eye_sreen_coord.y + 
				 proj_inv_light_viewmatrix[10] * eye_sreen_coord.z + 
				 proj_inv_light_viewmatrix[14]) / 
				(proj_inv_light_viewmatrix[3] * eye_sreen_coord.x + 
				 proj_inv_light_viewmatrix[7] * eye_sreen_coord.y + 
				 proj_inv_light_viewmatrix[11] * eye_sreen_coord.z + 
				 proj_inv_light_viewmatrix[15] + 0.0000001f);
				float shadow_z = shadow_map[SM_W * sm_coord.y + sm_coord.x];
				final_factor = (view_z - shadow_z) > shadow_eps ? shadow_factor : 1.0f;
			}

			weight += alpha * T;

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T * final_factor;
			D += depths[collected_id[j]] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_alpha[pix_id] = weight; //1 - T;
		out_depth[pix_id] = D / weight;
	}
}

void FORWARD::renderWithShadowMap(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float* depths,
	const float* depths_proj,
	const float4* conic_opacity,
	float* out_alpha,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
    
	int SM_W, int SM_H,
	const float* shadow_map,
	const float* proj_inv_light_viewmatrix,
	const float* proj_inv_light_projmatrix,
	const float shadow_eps,
	const float shadow_factor)
{
	renderWithShadowMapCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		depths,
		depths_proj,
		conic_opacity,
		out_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		
		SM_W, SM_H,
		shadow_map,
		proj_inv_light_viewmatrix,
		proj_inv_light_projmatrix,
		shadow_eps,
		shadow_factor
		);
}