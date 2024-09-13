/**
 * @file execute_code.hpp
 * @brief This file provides act as a template for executing the pipeline of image processing filters on the CPU and GPU.
 */

#pragma once
#ifndef EXECUTECODE_TEMPLATE_H
#define EXECUTECODE_TEMPLATE_H
#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include "SYCLUtils.hpp"
#include "Tracer.hpp"
#include "filters-CPP.hpp"
#include "filters-SYCL.hpp"
#include <sycl/sycl.hpp>
#include <vector>
#if AVX_ENABLED
#include "filters-AVX.hpp"
#endif
#if SIMD_ENABLED
#include "filters-SIMD.hpp"
#endif
#include "common_macros.hpp"
#include "execute_code_CPU.hpp"
#include "execute_code_GPU.hpp"

using namespace Pipeline_template;

struct basic {}; // Define a new type tag for the basic version of kernel SYCL

namespace Details {
/**
 * @brief Executes the cosine filter on the given frame data with dependencies.
 *
 * @tparam D Accelerator type (Acc::CPU or Acc::GPU)
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging.
 * @param[in] appData ApplicationData object containing application-specific data
 * @param[in] inputArgs InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to enqueue the filter.
 * @param[in,out] depends_on Optional vector of SYCL events to synchronize with before executing this operation.
 * @return SyclEventInfo object containing the SYCL event associated with this operation and its execution time
 */
template <Acc D>
SyclEventInfo cosinefilter(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
/**
 * @brief Executes the block histogram on the given frame data with dependencies.
 *
 * @tparam D Accelerator type (Acc::CPU or Acc::GPU)
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging.
 * @param[in] appData ApplicationData object containing application-specific data.
 * @param[in] inputArgs InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to enqueue the filter.
 * @param[in,out] depends_on Optional vector of SYCL events to synchronize with before executing this operation.
 * @return SyclEventInfo object containing the SYCL event associated with this operation and its execution time
 */
template <Acc D>
SyclEventInfo blockhistogram(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);
// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
/**
 * @brief Executes the pair-wise distance (pwdist) calculation on the given frame data with dependencies.
 *
 * @tparam D Accelerator type (Acc::CPU or Acc::GPU)
 * @param[in,out] item ViVidItem pointer representing the item to be processed and which contains the input data and output buffers.
 * @param[in,out] my_tracer Tracer object for performance analysis and debugging.
 * @param[in] appData ApplicationData object containing application-specific data.
 * @param[in] inputArgs InputArgs object containing the input arguments for the application.
 * @param[in] Q The SYCL queue to enqueue the filter.
 * @param[in,out] depends_on Optional vector of SYCL events to synchronize with before executing this operation.
 * @return SyclEventInfo object containing the SYCL event associated with this operation and its execution time.
 */
template <Acc D>
SyclEventInfo pwdist(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);
} // namespace Details

SyclEventInfo cosinefilter(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);
SyclEventInfo blockhistogram(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);
SyclEventInfo pwdist(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on = nullptr);
SyclEventInfo workloadsimulator(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, int stage);

#endif