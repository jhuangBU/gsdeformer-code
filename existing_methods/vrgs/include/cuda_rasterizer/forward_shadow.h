/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_SHADOW_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_SHADOW_H_INCLUDED

#include "forward.h"

namespace FORWARD
{
	// Main rasterization method.
	void renderWithShadowMap(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
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
		const float shadow_factor);
}


#endif