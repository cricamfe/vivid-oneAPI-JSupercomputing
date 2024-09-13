#pragma once
#ifndef FILTERS_H
#define FILTERS_H
/**********************************************************************************
* Filters
**********************************************************************************/
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include "pipeline_template.hpp"

using namespace std;
using namespace Pipeline_template;

// FIRST FILTER:
// Transposes the bank of filters
float * transposeBank(float* filter_bank);

// Optimized Filter 1 that works with a transposed bank of filters
void cosine_filter_transpose(float* fr_data, float* ind, float *val, float* fb_array_main, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, int pitch);

// SECOND FILTER:
void block_histogram(float *ptr_his, float *ptr_ind, float *ptr_val, int max_bin, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind);

// THIRD FILTER:
void pwdist_c(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth);

#endif