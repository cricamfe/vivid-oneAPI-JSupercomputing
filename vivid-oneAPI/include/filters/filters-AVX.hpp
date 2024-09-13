#pragma once
#ifndef FILTERS_AVX_H
#define FILTERS_AVX_H

#include <immintrin.h>
#include <cmath>

// *********************************************************************************************************************
// FILTER 1:
// *********************************************************************************************************************
void cosine_filter_AVX(float* fr_data, float* ind, float *val, float* fb_array, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, int pitch);

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
void block_histogram_AVX(float *ptr_his, float *id_data, float *wt_data, int max_bin, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind);

// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
void pwdist_AVX_cache_locality(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth);
void pwdist_AVX(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth);


#endif