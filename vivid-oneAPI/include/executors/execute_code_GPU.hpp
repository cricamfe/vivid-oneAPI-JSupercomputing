/**
 * @file execute_code_GPU.hpp
 * @brief This file provides definitions for executing the pipeline of image processing filters on the GPU using SYCL.
 */

#pragma once
#ifndef EXECUTECODE_GPU_TEMPLATE_H
#define EXECUTECODE_GPU_TEMPLATE_H

#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include "SYCLUtils.hpp"
#include "Tracer.hpp"
#include "filters-SYCL.hpp"
#include "pipeline_template.hpp"
#include <sycl/sycl.hpp>
#include <vector>

using namespace Pipeline_template;
struct basic; // forward declaration

// *********************************************************************************************************************
// FILTER 1:
// *********************************************************************************************************************
/**
 * @brief Computes the cosine filter on the given frame data using the GPU.
 *
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging the GPU execution.
 * @param[in] appData The ApplicationData object containing filter bank and other important parameters.
 * @param[in] inputArgs The InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to enqueue the filter.
 * @param[in,out] depends_on Optional vector of SYCL events to synchronize with before executing this operation.
 * @return SyclEventInfo containing the SYCL event and execution time.
 */
SyclEventInfo cosinefilter_GPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
/**
 * @brief Computes the block histogram of the given frame data using the GPU.
 *
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging the GPU execution.
 * @param[in] appData The ApplicationData object containing filter bank and other important parameters.
 * @param[in] inputArgs The InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to enqueue the filter.
 * @param[in,out] depends_on Optional vector of SYCL events to synchronize with before executing this operation.
 * @return SyclEventInfo containing the SYCL event and execution time.
 */
SyclEventInfo blockhistogram_GPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
/**
 * @brief Executes the pairwise distance between input data items on the GPU
 *
 * @tparam tile_size The tile size used for the optimized SYCL kernel (default: 16).
 * @tparam T The data type of the input data, allows us to select the kernel optimization we want to use (float, sycl::float4 or none).
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging the GPU execution.
 * @param[in] appData The ApplicationData object containing filter bank and other important parameters.
 * @param[in] inputArgs The InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to enqueue the filter.
 * @param[in,out] depends_on Optional vector of SYCL events to synchronize with before executing this operation.
 * @return SyclEventInfo containing the SYCL event and execution time.
 */
template <size_t tile_size, typename T>
SyclEventInfo pwdist_GPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

#endif