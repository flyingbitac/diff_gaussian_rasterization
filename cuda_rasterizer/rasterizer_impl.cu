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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

static bool gs_preprocess_probe_enabled() {
	const char* env = std::getenv("GS_PREPROCESS_PROBE");
	return env && env[0] == '1';
}

static bool gs_batch_debug_enabled() {
    const char* env = std::getenv("GS_BATCH_DEBUG");
    return env && atoi(env) == 1;
}

__global__ void computeRadiiPosCountKernel(int* radii, int P, int N, int* out_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * N) return;
    if (radii[idx] > 0) {
        atomicAdd(&out_counts[idx / P], 1);
    }
}

__global__ void computeTilesTouchedSumKernel(uint32_t* tiles_touched, int P, int N, uint32_t* out_sums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P * N) return;
    atomicAdd(&out_sums[idx / P], tiles_touched[idx]);
}

__global__ void computeRangesValidCountKernel(uint2* ranges, int N, int tile_count, int* out_valid, int* out_oob) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * tile_count) return;
    uint2 range = ranges[idx];
    if (range.x < range.y) {
        atomicAdd(out_valid, 1);
    } else if (range.x > range.y || (range.x == 0 && range.y == 0)) {
    } else {
        atomicAdd(out_oob, 1);
    }
}

__global__ void computeOutColorNonzeroCountKernel(float* out_color, int N, int C, int H, int W, int* out_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int cam_id = idx / (C * H * W);
    if (out_color[idx] != 0.0f) {
        atomicAdd(&out_counts[cam_id], 1);
    }
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Batch version: each Gaussian projects to each camera, generating P*N tile entries.
// Key format: | camera_id (16 bits) | tile_id (16 bits) | depth (32 bits) |
// This ensures all entries for camera 0 come first (sorted by tile, then depth),
// then all entries for camera 1, etc.
__global__ void duplicateWithKeysBatch(
	int P,
	int N,
	const float2* points_xy,    // Per-camera 2D points: (N, P) or interleaved
	const float* depths,        // Per-camera depths: (N, P)
	const uint32_t* offsets,    // Prefix sum of tiles_touched per (camera, gaussian)
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,           // Per-camera radii: (N, P)
	dim3 grid)
{
	// Each thread handles one (camera, gaussian) pair
	// Global index = cam_id * P + gaussian_id
	auto idx = cg::this_grid().thread_rank();
	int cam_id = idx / P;
	int gaussian_id = idx % P;
	
	if (cam_id >= N)
		return;

	// Get per-camera data (assuming contiguous memory: points_xy[cam_id*P + gaussian_id])
	const float2* cam_points_xy = points_xy + cam_id * P;
	const float* cam_depths = depths + cam_id * P;
	const int* cam_radii = radii + cam_id * P;
	
	// Generate no key/value pair for invisible Gaussians
	if (cam_radii[gaussian_id] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		// offsets is computed per (camera, gaussian) in preprocess
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(cam_points_xy[gaussian_id], cam_radii[gaussian_id], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is:
		// | camera_id (16 bits) | tile_id (16 bits) | depth (32 bits) |
		// Sorting with this key yields Gaussian IDs sorted first by camera,
		// then by tile, then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = 0;
				
				// Pack camera_id in bits 48-63 (upper 16 bits)
				key |= ((uint64_t)cam_id & 0xFFFF) << 48;
				
				// Pack tile_id in bits 32-47 (middle 16 bits)
				uint32_t tile_id = y * grid.x + x;
				key |= ((uint64_t)tile_id & 0xFFFF) << 32;
				
				// Pack depth in lower 32 bits
				key |= *((uint32_t*)&cam_depths[gaussian_id]);
				
				gaussian_keys_unsorted[off] = key;
				// Value stores only gaussian_id; camera_id is extracted from key via blockIdx.z in render
				gaussian_values_unsorted[off] = (uint32_t)gaussian_id;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Batch version: handles keys with format | camera_id (16b) | tile_id (16b) | depth (32b) |
// ranges is 2D: ranges[camera_id * tile_count + tile_id]
__global__ void identifyTileRangesBatch(int L, int num_cameras, int tile_count, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Extract camera_id and tile_id from key
	// key format: | camera_id(16) | tile_id(16) | depth(32) |
	uint64_t key = point_list_keys[idx];
	uint32_t curr_camera = (key >> 48) & 0xFFFF;  // upper 16 bits
	uint32_t curr_tile = (key >> 32) & 0xFFFF;    // middle 16 bits
	uint32_t curr_idx = curr_camera * tile_count + curr_tile;
	
	if (idx == 0)
		ranges[curr_idx].x = 0;
	else
	{
		uint64_t prev_key = point_list_keys[idx - 1];
		uint32_t prev_camera = (prev_key >> 48) & 0xFFFF;
		uint32_t prev_tile = (prev_key >> 32) & 0xFFFF;
		uint32_t prev_idx = prev_camera * tile_count + prev_tile;
		
		if (curr_idx != prev_idx)
		{
			ranges[prev_idx].y = idx;
			ranges[curr_idx].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[curr_idx].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::GeometryStateBatch CudaRasterizer::GeometryStateBatch::fromChunk(char*& chunk, size_t P, size_t N)
{
	GeometryStateBatch geom;
	obtain(chunk, geom.depths, P * N, 128);
	obtain(chunk, geom.clamped, P * N * 3, 128);
	obtain(chunk, geom.internal_radii, P * N, 128);
	obtain(chunk, geom.means2D, P * N, 128);
	obtain(chunk, geom.cov3D, P * N * 6, 128);
	obtain(chunk, geom.conic_opacity, P * N, 128);
	obtain(chunk, geom.rgb, P * N * 3, 128);
	obtain(chunk, geom.tiles_touched, P * N, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P * N);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P * N, 128);
	return geom;
}

CudaRasterizer::ImageStateBatch CudaRasterizer::ImageStateBatch::fromChunk(char*& chunk, size_t N, size_t tile_count, size_t num_pixels)
{
	ImageStateBatch img;
	obtain(chunk, img.accum_alpha, N * num_pixels, 128);
	obtain(chunk, img.n_contrib, N * num_pixels, 128);
	obtain(chunk, img.ranges, N * tile_count, 128);
	return img;
}

CudaRasterizer::BinningStateBatch CudaRasterizer::BinningStateBatch::fromChunk(char*& chunk, size_t max_rendered)
{
	BinningStateBatch binning;
	obtain(chunk, binning.point_list, max_rendered, 128);
	obtain(chunk, binning.point_list_unsorted, max_rendered, 128);
	obtain(chunk, binning.point_list_keys, max_rendered, 128);
	obtain(chunk, binning.point_list_keys_unsorted, max_rendered, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, max_rendered);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	const int probe_debug = gs_preprocess_probe_enabled() ? 1 : 0;
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		probe_debug,
		prefiltered,
		antialiasing
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	if (debug)
		std::cout << "[Rasterizer::forward] P=" << P
				<< ", image=" << width << "x" << height
				<< ", num_rendered=" << num_rendered << std::endl;


	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid);
	CHECK_CUDA(cudaGetLastError(), debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(cudaGetLastError(), debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_invdepths,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dinvdepth,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		geomState.depths,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_invdepths,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dinvdepth), debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
		const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dinvdepth,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		antialiasing), debug);
}

// Batch forward: render N cameras in a single kernel launch
int CudaRasterizer::Rasterizer::forwardBatch(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const int N,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* depth,
	bool antialiasing,
	int* radii,
	bool debug)
{
	const bool gs_batch_debug = gs_batch_debug_enabled();
	const int probe_debug = gs_preprocess_probe_enabled() ? 1 : 0;
	
	if (gs_batch_debug) {
		std::cout << "[GS_BATCH_DEBUG] forwardBatch START: P=" << P << ", N=" << N << ", " << width << "x" << height << std::endl;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, N);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t geom_chunk_size = required<GeometryStateBatch>(P, N);
	char* geom_chunkptr = geometryBuffer(geom_chunk_size);
	GeometryStateBatch geomState = GeometryStateBatch::fromChunk(geom_chunkptr, P, N);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	size_t img_chunk_size = required<ImageStateBatch>(N, tile_grid.x * tile_grid.y, width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageStateBatch imgState = ImageStateBatch::fromChunk(img_chunkptr, N, tile_grid.x * tile_grid.y, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// STAGE 1: preprocessBatch
	CHECK_CUDA(FORWARD::preprocessBatch(
		P, D, M, N,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		probe_debug,
		prefiltered,
		antialiasing
	), debug)

	if (gs_batch_debug) {
		auto ret = cudaDeviceSynchronize();
		if (ret != cudaSuccess) {
			std::cerr << "\n[GS_BATCH_DEBUG] STAGE 1 preprocessBatch CUDA ERROR: " << cudaGetErrorString(ret) << std::endl;
			throw std::runtime_error(cudaGetErrorString(ret));
		}
		std::cout << "[GS_BATCH_DEBUG] STAGE 1 preprocessBatch PASSED" << std::endl;
		
		// Compute counters after preprocessBatch
		int* radii_pos_counts_host = new int[N]();
		uint32_t* tiles_touched_sums_host = new uint32_t[N]();
		int *radii_pos_counts;
		uint32_t *tiles_touched_sums;
		cudaMalloc(&radii_pos_counts, N * sizeof(int));
		cudaMalloc(&tiles_touched_sums, N * sizeof(uint32_t));
		cudaMemset(radii_pos_counts, 0, N * sizeof(int));
		cudaMemset(tiles_touched_sums, 0, N * sizeof(uint32_t));
		computeRadiiPosCountKernel<<<(P * N + 255) / 256, 256>>>(radii, P, N, radii_pos_counts);
		computeTilesTouchedSumKernel<<<(P * N + 255) / 256, 256>>>(geomState.tiles_touched, P, N, tiles_touched_sums);
		cudaDeviceSynchronize();
		cudaMemcpy(radii_pos_counts_host, radii_pos_counts, N * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(tiles_touched_sums_host, tiles_touched_sums, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);
		std::cout << "[GS_BATCH_DEBUG] preprocessBatch counts: ";
		for (int i = 0; i < N && i < 2; i++) {
			std::cout << "cam" << i << "_radii_pos=" << radii_pos_counts_host[i] << ", cam" << i << "_tiles_touched=" << tiles_touched_sums_host[i];
			if (i < N - 1 && i < 1) std::cout << "; ";
		}
		std::cout << std::endl;
		cudaFree(radii_pos_counts);
		cudaFree(tiles_touched_sums);
		delete[] radii_pos_counts_host;
		delete[] tiles_touched_sums_host;
	}

	// STAGE 2: scan
	// Compute prefix sum over full list of touched tile counts per (camera, gaussian)
	// This must be done BEFORE duplicateWithKeysBatch to initialize point_offsets
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P * N), debug)

	// Retrieve actual number of rendered entries
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P * N - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	if (debug)
		std::cout << "[Rasterizer::forwardBatch] P=" << P << ", N=" << N
			<< ", image=" << width << "x" << height
			<< ", num_rendered=" << num_rendered << std::endl;

	if (gs_batch_debug) {
		auto ret = cudaDeviceSynchronize();
		if (ret != cudaSuccess) {
			std::cerr << "\n[GS_BATCH_DEBUG] STAGE 2 scan CUDA ERROR: " << cudaGetErrorString(ret) << std::endl;
			throw std::runtime_error(cudaGetErrorString(ret));
		}
		std::cout << "[GS_BATCH_DEBUG] STAGE 2 scan PASSED, num_rendered=" << num_rendered << std::endl;
	}

	size_t num_tiles_per_cam = tile_grid.x * tile_grid.y;

	// Allocate binning state with actual num_rendered
	size_t binning_chunk_size = required<BinningStateBatch>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningStateBatch binningState = BinningStateBatch::fromChunk(binning_chunkptr, num_rendered);

	// STAGE 3: duplicateWithKeysBatch
	// Duplicate with keys - each Gaussian per camera generates tile entries
	CHECK_CUDA((duplicateWithKeysBatch << <(P * N + 255) / 256, 256 >> > (
		P, N,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)), debug)

	if (gs_batch_debug) {
		auto ret = cudaDeviceSynchronize();
		if (ret != cudaSuccess) {
			std::cerr << "\n[GS_BATCH_DEBUG] STAGE 3 duplicateWithKeysBatch CUDA ERROR: " << cudaGetErrorString(ret) << std::endl;
			throw std::runtime_error(cudaGetErrorString(ret));
		}
		std::cout << "[GS_BATCH_DEBUG] STAGE 3 duplicateWithKeysBatch PASSED" << std::endl;
	}

	// STAGE 4: sort
	// For batch, we use 48 bits for key (16 bit camera + 16 bit tile + 16 bit depth)
	int bit = getHigherMsb(tile_grid.x * tile_grid.y) + 16;  // +16 for camera_id bits

	// Sort by keys using actual num_rendered
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 64), debug)

	if (gs_batch_debug) {
		auto ret = cudaDeviceSynchronize();
		if (ret != cudaSuccess) {
			std::cerr << "\n[GS_BATCH_DEBUG] STAGE 4 sort CUDA ERROR: " << cudaGetErrorString(ret) << std::endl;
			throw std::runtime_error(cudaGetErrorString(ret));
		}
		std::cout << "[GS_BATCH_DEBUG] STAGE 4 sort PASSED" << std::endl;
		
		// Check for OOB in point_list
		int point_list_oob = 0;
		if (num_rendered > 0) {
			uint32_t* point_list_host = new uint32_t[num_rendered];
			cudaMemcpy(point_list_host, binningState.point_list, num_rendered * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			for (int i = 0; i < num_rendered; i++) {
				if (point_list_host[i] >= (uint32_t)P) {
					point_list_oob++;
				}
			}
			delete[] point_list_host;
		}
		std::cout << "[GS_BATCH_DEBUG] point_list_oob_count=" << point_list_oob << std::endl;
	}

	// STAGE 5: identifyTileRangesBatch
	// Initialize ranges
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, N * num_tiles_per_cam * sizeof(uint2)), debug);

	// Identify start and end of per-(camera, tile) workloads
	if (num_rendered > 0)
		identifyTileRangesBatch << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			N,
			num_tiles_per_cam,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(cudaGetLastError(), debug)

	if (gs_batch_debug) {
		auto ret = cudaDeviceSynchronize();
		if (ret != cudaSuccess) {
			std::cerr << "\n[GS_BATCH_DEBUG] STAGE 5 identifyTileRangesBatch CUDA ERROR: " << cudaGetErrorString(ret) << std::endl;
			throw std::runtime_error(cudaGetErrorString(ret));
		}
		std::cout << "[GS_BATCH_DEBUG] STAGE 5 identifyTileRangesBatch PASSED" << std::endl;
		
		// Compute ranges valid/oob counts
		int ranges_valid = 0, ranges_oob = 0;
		uint2* ranges_host = new uint2[N * num_tiles_per_cam];
		cudaMemcpy(ranges_host, imgState.ranges, N * num_tiles_per_cam * sizeof(uint2), cudaMemcpyDeviceToHost);
		for (int i = 0; i < N * num_tiles_per_cam; i++) {
			if (ranges_host[i].x < ranges_host[i].y) {
				ranges_valid++;
			} else if (ranges_host[i].x > ranges_host[i].y) {
				ranges_oob++;
			}
		}
		delete[] ranges_host;
		std::cout << "[GS_BATCH_DEBUG] ranges_valid_count=" << ranges_valid << ", ranges_oob_count=" << ranges_oob << std::endl;
	}

	// STAGE 6: renderBatch
	// Render all cameras
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	int feature_stride = colors_precomp != nullptr ? 0 : P * 3;
	CHECK_CUDA(FORWARD::renderBatch(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		P, N, width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		depth,
		feature_stride), debug)

	if (gs_batch_debug) {
		auto ret = cudaDeviceSynchronize();
		if (ret != cudaSuccess) {
			std::cerr << "\n[GS_BATCH_DEBUG] STAGE 6 renderBatch CUDA ERROR: " << cudaGetErrorString(ret) << std::endl;
			throw std::runtime_error(cudaGetErrorString(ret));
		}
		std::cout << "[GS_BATCH_DEBUG] STAGE 6 renderBatch PASSED" << std::endl;
		
		// Compute out_color nonzero counts per camera
		int* out_color_nonzero_counts_host = new int[N]();
		int *out_color_nonzero_counts;
		cudaMalloc(&out_color_nonzero_counts, N * sizeof(int));
		cudaMemset(out_color_nonzero_counts, 0, N * sizeof(int));
		computeOutColorNonzeroCountKernel<<<(N * 3 * height * width + 255) / 256, 256>>>(
			out_color, N, 3, height, width, out_color_nonzero_counts);
		cudaDeviceSynchronize();
		cudaMemcpy(out_color_nonzero_counts_host, out_color_nonzero_counts, N * sizeof(int), cudaMemcpyDeviceToHost);
		cudaFree(out_color_nonzero_counts);
		
		// Print JSON for cam0/cam1
		std::cout << "[GS_BATCH_DEBUG] JSON: {";
		for (int cam_id = 0; cam_id < N && cam_id < 2; cam_id++) {
			int radii_pos = 0;
			uint32_t tiles_sum = 0;
			int ranges_valid = 0, ranges_oob = 0;
			
			int* radii_pos_buf; uint32_t* tiles_sum_buf;
			cudaMalloc(&radii_pos_buf, sizeof(int));
			cudaMalloc(&tiles_sum_buf, sizeof(uint32_t));
			cudaMemset(radii_pos_buf, 0, sizeof(int));
			cudaMemset(tiles_sum_buf, 0, sizeof(uint32_t));
			computeRadiiPosCountKernel<<<1, 256>>>(radii + cam_id * P, P, 1, radii_pos_buf);
			computeTilesTouchedSumKernel<<<1, 256>>>(geomState.tiles_touched + cam_id * P, P, 1, tiles_sum_buf);
			cudaDeviceSynchronize();
			cudaMemcpy(&radii_pos, radii_pos_buf, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&tiles_sum, tiles_sum_buf, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaFree(radii_pos_buf);
			cudaFree(tiles_sum_buf);
			
			uint2* ranges_host = new uint2[num_tiles_per_cam];
			cudaMemcpy(ranges_host, imgState.ranges + cam_id * num_tiles_per_cam, num_tiles_per_cam * sizeof(uint2), cudaMemcpyDeviceToHost);
			for (int t = 0; t < num_tiles_per_cam; t++) {
				if (ranges_host[t].x < ranges_host[t].y) ranges_valid++;
				else if (ranges_host[t].x > ranges_host[t].y) ranges_oob++;
			}
			delete[] ranges_host;
			
			std::cout << "\"cam" << cam_id << "_radii_pos_count\":" << radii_pos
				<< ",\"cam" << cam_id << "_tiles_touched_sum\":" << tiles_sum
				<< ",\"cam" << cam_id << "_ranges_valid_count\":" << ranges_valid
				<< ",\"cam" << cam_id << "_ranges_oob_count\":" << ranges_oob
				<< ",\"cam" << cam_id << "_out_color_nonzero_count\":" << out_color_nonzero_counts_host[cam_id];
			if (cam_id < N - 1 && cam_id < 1) std::cout << ",";
		}
		std::cout << ",\"num_rendered_total\":" << num_rendered << "}" << std::endl;
		
		delete[] out_color_nonzero_counts_host;
		std::cout << "[GS_BATCH_DEBUG] forwardBatch END" << std::endl;
	}

	return num_rendered;
}
