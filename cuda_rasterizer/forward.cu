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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// __device__ means this function runs on the GPU and can only be called
	// from other GPU code. It is a helper, not a kernel entry point.
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];
	// SH coefficients are stored per Gaussian. This function evaluates the basis
	// in the view direction and accumulates the resulting RGB color.

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
	// The original implementation adds a bias before clamping so colors remain
	// in a convenient range for later rendering.

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	// The backward pass needs to know which channels hit the clamp, because the
	// derivative through max(x, 0) is zero on the clamped side.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// This helper projects a 3D covariance into screen space. The return value is
	// the symmetric 2x2 covariance matrix stored as (xx, xy, yy).
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);
	// t is the Gaussian center in camera/view space.

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);
	// J is the local Jacobian of the projection from 3D to screen space.

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	// This is the usual covariance transform: A^T * Sigma * A.

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Convert the learned scale + rotation parameters into a 3D covariance matrix.
	// Only the upper-triangular part is written because the matrix is symmetric.
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	// The commented normalization hints that the quaternion is expected to be
	// normalized already by upstream code or training dynamics.
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__device__ __forceinline__ bool isProbeGaussian(int id)
{
	// Tiny debug helper: only a small set of hard-coded ids emit printf logs.
	return id == 0 || id == 1 || id == 2 || id == 1024 || id == 65536 || id == 262144 || id == 524288 || id == 1048576 || id == 1500000 || id == 1999999;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	int probe_debug,
	bool prefiltered,
	bool antialiasing)
{
	// One GPU thread preprocesses one Gaussian. The host launches enough threads
	// to cover P elements, then each thread exits if its index is out of range.
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	// This makes "culled" or invalid Gaussians cheap to detect later.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	// If the Gaussian center is outside the frustum, none of the later screen-space
	// work is needed for this Gaussian.
	float3 p_view;
	bool in_view = in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view);
	if (probe_debug && isProbeGaussian(idx))
		printf("[PPROBE][SER] gid=%d in=%d pview=(%f,%f,%f)\n", (int)idx, (int)in_view, (double)p_view.x, (double)p_view.y, (double)p_view.z);
	if (!in_view)
		return;

	// Transform point by projecting
	// p_proj is in normalized device coordinates (roughly [-1, 1] before clipping).
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	// cov3Ds is an output scratch buffer used only when covariances are not supplied.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	// This describes the ellipse footprint of the Gaussian after projection.
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
	// Anti-aliasing slightly broadens the covariance and compensates opacity so
	// total energy stays stable.

	// Invert covariance (EWA algorithm)
	// The renderer uses the inverse covariance ("conic" form) because evaluating
	// the Gaussian in image space is then just a quadratic form.
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// The radius is chosen as roughly 3 standard deviations, which captures most
	// of the Gaussian's visible influence.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	// Convert NDC coordinates into pixel coordinates so later kernels can work in
	// screen pixels and tiles directly.
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	// This keeps the rasterizer flexible: either consume RGB directly or derive it
	// on the fly from SH lighting coefficients.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	// These arrays are the bridge from preprocess to binning and rendering.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	// Packing related values together improves locality when the render kernel
	// fetches them many times.
	float opacity = opacities[idx];


	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };


	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	// tiles_touched is later prefix-scanned so each Gaussian can reserve exactly
	// enough slots in the duplicated point list.
	if (probe_debug && isProbeGaussian(idx))
		printf("[PPROBE][SER] gid=%d radius=%d tiles=%u det=%f cov=(%f,%f,%f)\n", (int)idx, (int)my_radius, (unsigned)tiles_touched[idx], (double)det, (double)cov.x, (double)cov.y, (double)cov.z);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth)
{
	// Identify current tile and associated min/max pixel range.
	// __launch_bounds__(BLOCK_X * BLOCK_Y) tells the compiler the intended block
	// size. Here one block corresponds to one image tile, and one thread usually
	// corresponds to one pixel in that tile.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	// pix_id is the flattened linear index into image-sized buffers.

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	// Even if a thread maps to an out-of-bounds pixel, it still participates in
	// synchronization so the block can cooperate safely.
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	// ranges contains a contiguous slice in point_list for this tile.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	// __shared__ memory lives on-chip and is shared by all threads in one block.
	// It is much faster than global memory, so the block stages Gaussian data here
	// before all pixels consume it.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	// T is transmittance: how much background light is still visible after the
	// Gaussians processed so far.
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	// The tile may overlap many Gaussians. Rather than loading all of them at once,
	// the block processes them in chunks of BLOCK_SIZE.
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		// __syncthreads_count(done) is both a barrier and a reduction: it counts how
		// many threads in the block currently have done == true.
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		// Each thread loads at most one Gaussian record for the current batch.
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();
		// All threads must wait until shared memory is fully populated.

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// The quadratic form below evaluates the exponent of the 2D Gaussian.
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
			// Very small alpha values are ignored because they contribute negligibly
			// but still cost arithmetic and can hurt numerical stability.
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				// Once transmittance is tiny, this pixel is effectively opaque and can
				// stop early even if more Gaussians remain.
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			// Accumulate premultiplied color contribution for this pixel.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	// The output layout is channel-major: all pixels of channel 0 first, then
	// all pixels of channel 1, etc.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth)
{
	// CPU wrapper around the actual GPU render kernel.
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	int probe_debug,
	bool prefiltered,
	bool antialiasing)
{
	// CPU wrapper that launches one preprocess thread per Gaussian.
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		probe_debug,
		prefiltered,
		antialiasing
		);
}

template<int C>
__global__ void preprocessCUDABatch(int P, int D, int M, int N,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrixs,
	const float* projmatrixs,
	const glm::vec3* cam_poss,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	int probe_debug,
	bool prefiltered,
	bool antialiasing)
{
	// Batched preprocess flattens the logical 2D problem:
	// row = camera, column = Gaussian, linear index = camera * P + Gaussian.
	auto idx = cg::this_grid().thread_rank();
	int total = P * N;
	if (idx >= total) return;

	int cam_id = idx / P;
	int gaussian_id = idx % P;
	// The same Gaussian is visited once per camera because projection, visibility,
	// and screen-space footprint all depend on the camera pose.

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	const float* viewmatrix = viewmatrixs + cam_id * 16;
	const float* projmatrix = projmatrixs + cam_id * 16;
	const glm::vec3 cam_pos = cam_poss[cam_id];
	// viewmatrixs / projmatrixs are packed as one 4x4 matrix (16 floats) per camera.

	float3 p_view;
	bool in_view = in_frustum(gaussian_id, orig_points, viewmatrix, projmatrix, prefiltered, p_view);
	if (probe_debug && cam_id == 0 && isProbeGaussian(gaussian_id))
		printf("[PPROBE][BAT] gid=%d in=%d pview=(%f,%f,%f)\n", (int)gaussian_id, (int)in_view, (double)p_view.x, (double)p_view.y, (double)p_view.z);
	if (!in_view)
		return;
	// From here down, the math mirrors the single-camera preprocess kernel; the
	// main difference is that outputs are stored in per-camera slices.
	float3 p_orig = { orig_points[3 * gaussian_id], orig_points[3 * gaussian_id + 1], orig_points[3 * gaussian_id + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + gaussian_id * 6;
	}
	else
	{
		float* cov3D_slot = cov3Ds + (cam_id * P + gaussian_id) * 6;
		// Even though the 3D covariance is camera-independent, the batched code stores
		// it per (camera, Gaussian) so all later accesses stay in a simple flattened layout.
		computeCov3D(scales[gaussian_id], scale_modifier, rotations[gaussian_id], cov3D_slot);
		cov3D = cov3D_slot;
	}

	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;
	if (antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov));

	const float det = det_cov_plus_h_cov;
	if (det == 0.0f) return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) return;

	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(gaussian_id, D, M, (glm::vec3*)orig_points, cam_pos, shs, clamped + cam_id * P * 3);
		float* rgb_slot = rgb + (cam_id * P + gaussian_id) * C;
		// Unlike the single-camera path, the clamped flags and RGB buffers are stored
		// separately for each camera because SH color depends on view direction.
		rgb_slot[0] = result.x;
		rgb_slot[1] = result.y;
		rgb_slot[2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	float opacity = opacities[gaussian_id];
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacity * h_convolution_scaling };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	if (probe_debug && cam_id == 0 && isProbeGaussian(gaussian_id))
		printf("[PPROBE][PRO] gid=%d radius=%d tiles=%u det=%f cov=(%f,%f,%f)\n", (int)gaussian_id, (int)my_radius, (unsigned)tiles_touched[idx], (double)det, (double)cov.x, (double)cov.y, (double)cov.z);
}

void FORWARD::preprocessBatch(int P, int D, int M, int N,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrixs,
	const float* projmatrixs,
	const glm::vec3* cam_poss,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	int probe_debug,
	bool prefiltered,
	bool antialiasing)
{
	int total = P * N;
	// Launch one GPU thread for each (camera, Gaussian) pair.
	preprocessCUDABatch<NUM_CHANNELS> << <(total + 255) / 256, 256 >> > (
		P, D, M, N,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrixs,
		projmatrixs,
		cam_poss,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		probe_debug,
		prefiltered,
		antialiasing);
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDABatch(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int P, int N, int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth,
	int feature_stride)
{
	// The batched render kernel reuses the same "one block = one tile" idea as the
	// single-camera version, but the grid's z dimension selects the camera.
	auto block = cg::this_thread_block();
	int cam_id = block.group_index().z;
	if (cam_id >= N) return;

	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint32_t tile_count_per_cam = horizontal_blocks * ((H + BLOCK_Y - 1) / BLOCK_Y);
	// Each camera owns its own contiguous slice in the ranges table.
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;

	uint2 range = ranges[cam_id * tile_count_per_cam + block.group_index().y * horizontal_blocks + block.group_index().x];
	// point_list still stores only Gaussian ids; camera identity comes from blockIdx.z.
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	const float2* cam_points = points_xy_image + cam_id * P;
	const float4* cam_conic = conic_opacity + cam_id * P;
	const float* cam_depths = depths + cam_id * P;
	// Slice the flattened (N, P, ...) arrays down to the current camera.

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float expected_invdepth = 0.0f;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE) break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = cam_points[coll_id];
			collected_conic_opacity[block.thread_rank()] = cam_conic[coll_id];
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;

			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f) { done = true; continue; }

			int feat_base = (feature_stride == 0 ? collected_id[j] * CHANNELS : cam_id * feature_stride + collected_id[j] * CHANNELS);
			// feature_stride distinguishes two memory layouts:
			// 1) precomputed colors shared by all cameras (stride 0)
			// 2) per-camera colors generated during preprocessBatch (non-zero stride)
			for (int ch = 0; ch < CHANNELS; ch++) C[ch] += features[feat_base + ch] * alpha * T;
			if (invdepth) expected_invdepth += (1.0f / cam_depths[collected_id[j]]) * alpha * T;
			T = test_T;
			last_contributor = contributor;
		}
	}

	if (inside)
	{
		// Batched outputs are camera-major: all data for camera 0, then camera 1, etc.
		final_T[cam_id * H * W + pix_id] = T;
		n_contrib[cam_id * H * W + pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[(cam_id * CHANNELS + ch) * H * W + pix_id] = C[ch] + T * bg_color[ch];
		if (invdepth) invdepth[cam_id * H * W + pix_id] = expected_invdepth;
	}
}

void FORWARD::renderBatch(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int P, int N, int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth,
	int feature_stride)
{
	// CPU wrapper around the batched GPU render kernel.
	renderCUDABatch<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		P, N, W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths,
		depth,
		feature_stride);
}
