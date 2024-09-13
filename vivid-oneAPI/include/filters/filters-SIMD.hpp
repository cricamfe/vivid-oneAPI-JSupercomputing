#pragma once
#ifndef FILTERS_SIMD_H
#define FILTERS_SIMD_H

#include <algorithm>
#include <cstddef>
#include <experimental/simd>
#include <iostream>
#include <vector>

namespace stdx = std::experimental;

template <typename T>
using simd = stdx::native_simd<T>;
using data_t = float;
using simd_t = stdx::native_simd<data_t>;
using simd_f = stdx::native_simd<float>;
using simd_i = stdx::native_simd<int>;
using mask_t = stdx::native_simd_mask<float>;

// *********************************************************************************************************************
// FILTER 1:
// *********************************************************************************************************************
void cosine_filter_SIMD(float *fr_data, float *ind, float *val, float *fb_array_main, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, int pitch);
// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
void block_histogram_SIMD(float *ptr_his, float *id_data, float *wt_data, int max_bin, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind);
// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
void pwdist_SIMD(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth);

#endif