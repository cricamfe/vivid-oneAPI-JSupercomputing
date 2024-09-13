/**
 * @file execute_code_CPU.hpp
 * @brief This file provides definitions for executing the pipeline of image processing filters on the CPU using C++, SYCL or AVX.
 */

#pragma once
#ifndef EXECUTECODE_CPU_TEMPLATE_H
#define EXECUTECODE_CPU_TEMPLATE_H
#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include "SYCLUtils.hpp"
#include "Tracer.hpp"
#include "filters-CPP.hpp"
#include "pipeline_template.hpp"
#include <sycl/sycl.hpp>
#include <vector>

#ifdef SYCL_ENABLED
#include "filters-SYCL.hpp"
#endif

#ifdef AVX_ENABLED
#include "filters-AVX.hpp"
#endif

#ifdef SIMD_ENABLED
#include "filters-SIMD.hpp"
#endif

using namespace Pipeline_template;
struct basic; // forward declaration
// *********************************************************************************************************************
// FILTER 1:
// *********************************************************************************************************************
/**
 * @brief Computes the cosine filter on the given frame data using the CPU.
 *
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging the CPU execution.
 * @param[in] appData The ApplicationData object containing filter bank and other important parameters.
 * @param[in] inputArgs The InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to execute the filter.
 * @param[in,out] depends_on Optional pointer vector of SYCL events to synchronize with before executing this operation (default: nullptr).
 * @return SyclEventInfo containing the SYCL event and execution time.
 */
SyclEventInfo cosinefilter_CPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
/**
 * @brief Computes the block histogram of the given frame data using the CPU.
 *
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging the CPU execution.
 * @param[in] appData The ApplicationData object containing cell size and other parameters.
 * @param[in] inputArgs The InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to execute the filter.
 * @param[in,out] depends_on Optional pointer vector of SYCL events to synchronize with before executing this operation (default: nullptr).
 * @return SyclEventInfo containing the SYCL event and execution time.
 */
SyclEventInfo blockhistogram_CPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
/**
 * @brief Computes the pairwise distance between input matrices using the CPU.
 *
 * @tparam tile_size The tile size used for the optimized SYCL kernel (default: 64).
 * @tparam T The data type of the input data, allows us to select the kernel optimization we want to use (float, sycl::float4 or none)
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging the CPU execution.
 * @param[in] appData The ApplicationData object containing cell size and other parameters.
 * @param[in] inputArgs The InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to execute the filter.
 * @param[in,out] depends_on Optional pointer vector of SYCL events to synchronize with before executing this operation (default: nullptr).
 * @return SyclEventInfo containing the SYCL event and execution time.
 */
template <size_t tile_size, typename T>
SyclEventInfo pwdist_CPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

#endif