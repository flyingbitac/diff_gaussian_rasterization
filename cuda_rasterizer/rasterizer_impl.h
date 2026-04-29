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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	// Batch versions for multi-camera rendering
	struct GeometryStateBatch
	{
		size_t scan_size;
		float* depths;           // (N, P)
		char* scanning_space;
		bool* clamped;           // (N, P, 3), only when SH RGB is rendered
		int* internal_radii;     // (N, P)
		float2* means2D;         // (N, P)
		float* cov3D;            // (P, 6), shared across cameras
		float4* conic_opacity;   // (N, P)
		float* rgb;              // (N, P, 3), only when SH RGB is rendered
		uint32_t* point_offsets; // (N, P)
		uint32_t* tiles_touched; // (N, P)

		static GeometryStateBatch fromChunk(char*& chunk, size_t P, size_t N, bool needs_sh_color);
	};

	struct ImageStateBatch
	{
		uint2* ranges;           // (N, tile_count)
		uint32_t* n_contrib;     // (N, H*W)
		float* accum_alpha;      // (N, H*W)

		static ImageStateBatch fromChunk(char*& chunk, size_t N, size_t tile_count, size_t num_pixels);
	};

	struct BinningStateBatch
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted; // (max_rendered)
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningStateBatch fromChunk(char*& chunk, size_t max_rendered);
	};

	template<typename T, typename... Args>
	size_t required(Args... args)
	{
		char* size = nullptr;
		T::fromChunk(size, args...);
		return ((size_t)size) + 128;
	}
};
