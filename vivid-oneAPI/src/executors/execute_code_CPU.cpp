/**
 * @file execute_code_CPU.cpp
 * @brief This file provides implementations for executing the pipeline of image processing filters on the CPU using C++, SYCL or AVX.
 * @author Cristian Campos
 * @date 2023-04-28
 */

#include "execute_code_CPU.hpp"
#include "common_macros.hpp"

using namespace std;

SyclEventInfo cosinefilter_CPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    // Start tracing and timing
    trace_start(item, my_tracer, "CPU");
    start_timer(item);
    // Get the pointers to the data
    float *ptr_frame, *ptr_ind, *ptr_val;
    int f_pitch_f;
    get_ptrs_cosine(item, ptr_frame, ptr_ind, ptr_val, f_pitch_f);

    sycl::event m_event;
    if constexpr (SYCL_ENABLED) {
        if (depends_on != nullptr) {
            m_event = cosine_filter_transpose_sycl(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterSize, appData.numFilters, f_pitch_f, Q, depends_on);
            wait_sycl_event(m_event);
        } else {
            m_event = cosine_filter_transpose_sycl(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterSize, appData.numFilters, f_pitch_f, Q);
        }
        // Save the execution time
        save_time_info_on_sycl(item, inputArgs, m_event, 0, "CPU_S");
    } else {
        if constexpr (AVX_ENABLED) {
            cosine_filter_AVX(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterDim, appData.filterDim, appData.numFilters, item->val->pitch);
        } else if constexpr (SIMD_ENABLED) {
            cosine_filter_SIMD(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterDim, appData.filterDim, appData.numFilters, item->val->pitch);
        } else {
            cosine_filter_transpose(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterDim, appData.filterDim, appData.numFilters, item->val->pitch);
        }
        // Save the trace information and execution time
        save_trace_info(item);
        save_time_info_normal(item, 0, "CPU_S");
    }
    // End tracing
    trace_end(item, my_tracer, "CPU");

    return {m_event, item->execution_time, Acc::CPU};
}

SyclEventInfo blockhistogram_CPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    // Start tracing and timing
    trace_start(item, my_tracer, "CPU");
    start_timer(item);
    // Get the pointers to the data
    float *ptr_his, *ptr_val, *ptr_ind;
    int histogram_pitch_f, assignments_pitch_f, weights_pitch_f;
    get_ptrs_histogram(item, appData, ptr_his, ptr_val, ptr_ind, histogram_pitch_f, assignments_pitch_f, weights_pitch_f);

    sycl::event m_event;
    if constexpr (SYCL_ENABLED) {
        if (depends_on != nullptr) {
            m_event = block_histogram_sycl(ptr_his, ptr_ind, ptr_val, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f, weights_pitch_f, Q, depends_on);
            wait_sycl_event(m_event);
        } else {
            m_event = block_histogram_sycl(ptr_his, ptr_ind, ptr_val, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f, weights_pitch_f, Q);
        }
        // Save the execution time
        save_time_info_on_sycl(item, inputArgs, m_event, 1, "CPU_S");
    } else {
        if constexpr (AVX_ENABLED) {
            block_histogram_AVX(ptr_his, ptr_ind, ptr_val, appData.numFilters, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f);
        } else if constexpr (SIMD_ENABLED) {
            block_histogram_SIMD(ptr_his, ptr_ind, ptr_val, appData.numFilters, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f);
        } else {
            block_histogram(ptr_his, ptr_ind, ptr_val, appData.numFilters, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f);
        }
        // Save the trace information and execution time
        save_trace_info(item);
        save_time_info_normal(item, 1, "CPU_S");
    }
    // End tracing
    trace_end(item, my_tracer, "CPU");

    return {m_event, item->execution_time, Acc::CPU};
}

template <size_t tile_size, typename T>
SyclEventInfo pwdist_CPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    // Start tracing and timing
    trace_start(item, my_tracer, "CPU");
    start_timer(item);
    // Get the pointers to the data
    float *ptra, *ptrb, *out;
    int owidth, aheight, awidth, bheight, adatawidth;
    get_ptrs_pwdist(item, ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth);

    sycl::event m_event;
    if constexpr (SYCL_ENABLED) {
        if constexpr (std::is_same_v<T, float>) {
            if (depends_on != nullptr) {
                m_event = pwdist_sycl_tiled<tile_size>(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth, Q, depends_on);
                wait_sycl_event(m_event);
            } else {
                m_event = pwdist_sycl_tiled<tile_size>(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth, Q);
            }
        } else if constexpr (std::is_same_v<T, sycl::float4>) {
            if (depends_on != nullptr) {
                m_event = pwdist_sycl_tiled_float4<tile_size>(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth, Q, depends_on);
                wait_sycl_event(m_event);
            } else {
                m_event = pwdist_sycl_tiled_float4<tile_size>(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth, Q);
            }
        } else if constexpr (std::is_same_v<T, basic>) {
            if (depends_on != nullptr) {
                m_event = pwdist_sycl_basic(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth, Q, depends_on);
                wait_sycl_event(m_event);
            } else {
                m_event = pwdist_sycl_basic(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth, Q);
            }
        }
        // Save the execution time
        save_time_info_on_sycl(item, inputArgs, m_event, 2, "CPU_S");
    } else {
        if constexpr (AVX_ENABLED) {
            pwdist_AVX_cache_locality(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth);
        } else if constexpr (SIMD_ENABLED) {
            pwdist_SIMD(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth);
        } else {
            pwdist_c(ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth);
        }
        save_trace_info(item);
        save_time_info_normal(item, 2, "CPU_S");
    }
    // End tracing
    trace_end(item, my_tracer, "CPU");

    return {m_event, item->execution_time, Acc::CPU};
}

template SyclEventInfo pwdist_CPU<64, float>(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on);
template SyclEventInfo pwdist_CPU<64, sycl::float4>(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on);
template SyclEventInfo pwdist_CPU<0, basic>(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on);