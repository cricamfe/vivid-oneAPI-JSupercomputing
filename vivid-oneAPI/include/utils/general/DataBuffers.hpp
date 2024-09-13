#ifndef DATA_BUFFERS_HPP
#define DATA_BUFFERS_HPP

#include "ApplicationData.hpp"
#include "circular-buffer.hpp"
#include "pipeline_template.hpp"
#include <memory>
#include <random>
#include <string>
#include <sycl/sycl.hpp>
#include <vector>

namespace DataBuffers {
FloatBuffer *createGlobalFrame(const std::unique_ptr<float[]> &f_imData, int height, int width, sycl::queue &Q);
float *createFilterBank(const int numFilters, const int filterDim, std::mt19937 &mte, sycl::queue &Q);
FloatBuffer *createGlobalCla(const int window_height, const int window_width, const int cell_size, const int block_size, const int dict_size, std::mt19937 &mte, sycl::queue &Q);
void createAllBuffers(ApplicationData &appData, const std::unique_ptr<float[]> &f_imData);
} // namespace DataBuffers

#endif // DATA_BUFFERS_HPP