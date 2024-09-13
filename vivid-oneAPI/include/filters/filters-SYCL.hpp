#pragma once
#ifndef FILTERS_GPU_H
#define FILTERS_GPU_H

#include "SYCLUtils.hpp"
#include "pipeline_template.hpp"
#include <cmath>
#include <vector>

using namespace Pipeline_template;

// *********************************************************************************************************************
// FILTER 1:
// *********************************************************************************************************************
sycl::event cosine_filter_transpose_sycl(float *frame, float *ind, float *val, float *fb_array_main, const int height, const int width, const int filter_size, const int n_filters, const int f_pitch_f, sycl::queue &Q, const std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
sycl::event block_histogram_sycl(float *ptr_his, float *ptr_ind, float *ptr_val, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind, float pitch_val, sycl::queue &Q, const std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
// Basic implementation of pairwise distance (unoptimized)
sycl::event pwdist_sycl_basic(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *depends_on = nullptr);

// Tiled implementation of pairwise distance (optimized) with float
template <size_t tile_size>
sycl::event pwdist_sycl_tiled(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *depends_on = nullptr);

// Tiled implementation of pairwise distance (optimized) with float4
template <size_t tile_size>
sycl::event pwdist_sycl_tiled_float4(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *depends_on = nullptr);

#endif