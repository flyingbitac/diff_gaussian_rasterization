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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		int probe_debug,
		bool prefiltered,
		bool antialiasing);

	// Batch version: each Gaussian projects to N cameras
	void preprocessBatch(int P, int D, int M, int N,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,   // (N, 4, 4)
		const float* projmatrix,   // (N, 4, 4)
		const glm::vec3* cam_pos,  // (N, 3)
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,                // (N, P)
		float2* points_xy_image,   // (N, P)
		float* depths,             // (N, P)
		float* cov3Ds,
		float* colors,             // (P, 3)
		float4* conic_opacity,     // (N, P)
		const dim3 grid,
		uint32_t* tiles_touched,   // (N, P)
		int probe_debug,
		bool prefiltered,
		bool antialiasing,
		bool render_color);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* depths,
		float* depth);

	// Batch render: N cameras with 2D ranges (camera_id * tile_count + tile_id)
	void renderBatch(
		const dim3 grid, dim3 block,
		const uint2* ranges,       // (N * tile_count)
		const uint32_t* point_list,
		int P, int N, int W, int H,
		const float2* points_xy_image,  // (N, P)
		const float* features,          // (stride/3, 3) - shared or per-camera
		const float4* conic_opacity,    // (N, P)
		float* final_T,                 // (N, H*W)
		uint32_t* n_contrib,            // (N, H*W)
		const float* bg_color,
		float* out_color,               // (N, 3, H, W)
		float* depths,                  // (N, P)
		float* depth,                   // (N, 1, H, W)
		int feature_stride);            // 0 for shared colors_precomp, P*3 for per-camera SH, -1 for no color
}


#endif
