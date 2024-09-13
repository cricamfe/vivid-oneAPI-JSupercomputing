/**
 * @file execute_code_GPU.cpp
 * @brief This file provides implementations for executing the pipeline of image processing filters on the GPU using SYCL.
 */

#include "execute_code_GPU.hpp"
#include "common_macros.hpp"

using namespace std;

// Implementation of cosinefilter_GPU function
SyclEventInfo cosinefilter_GPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    // Start tracing and timing
    trace_start(item, my_tracer, "GPU");
    start_timer(item);
    // Get the pointers to the data
    float *ptr_frame, *ptr_ind, *ptr_val;
    int f_pitch_f;
    get_ptrs_cosine(item, ptr_frame, ptr_ind, ptr_val, f_pitch_f);

    // Reset the execution time
    item->execution_time = 0.0;
    // Create the event info
    sycl::event m_event;

    if (depends_on != nullptr) {
        m_event = cosine_filter_transpose_sycl(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterSize, appData.numFilters, f_pitch_f, Q, depends_on);
        wait_sycl_event(m_event);
    } else {
        m_event = cosine_filter_transpose_sycl(ptr_frame, ptr_ind, ptr_val, appData.filterBank, appData.height, appData.width, appData.filterSize, appData.numFilters, f_pitch_f, Q);
    }

    // Save the execution time and end tracing
    save_time_info_on_sycl(item, inputArgs, m_event, 0, "GPU_S");
    trace_end(item, my_tracer, "GPU");

    return SyclEventInfo(m_event, item->execution_time, Acc::GPU);
}

// Implementation of blockhistogram_GPU function
SyclEventInfo blockhistogram_GPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    // Start tracing and timing
    trace_start(item, my_tracer, "GPU");
    start_timer(item);
    // Get the pointers to the data
    float *ptr_his, *ptr_val, *ptr_ind;
    int histogram_pitch_f, assignments_pitch_f, weights_pitch_f;
    get_ptrs_histogram(item, appData, ptr_his, ptr_val, ptr_ind, histogram_pitch_f, assignments_pitch_f, weights_pitch_f);

    // Reset the execution time
    item->execution_time = 0.0;
    // Create the event info
    sycl::event m_event;

    if (depends_on != nullptr) {
        m_event = block_histogram_sycl(ptr_his, ptr_ind, ptr_val, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f, weights_pitch_f, Q, depends_on);
        if constexpr (TRACE_ENABLED)
            m_event.wait();
    } else {
        m_event = block_histogram_sycl(ptr_his, ptr_ind, ptr_val, appData.cellSize, appData.height, appData.width, histogram_pitch_f, assignments_pitch_f, weights_pitch_f, Q);
    }

    // Save the execution time and end tracing
    save_time_info_on_sycl(item, inputArgs, m_event, 1, "GPU_S");
    trace_end(item, my_tracer, "GPU");

    return SyclEventInfo(m_event, item->execution_time, Acc::GPU);
}

// Implementation of pwdist_GPU function
template <size_t tile_size, typename T>
SyclEventInfo pwdist_GPU(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    // Start tracing and timing
    trace_start(item, my_tracer, "GPU");
    start_timer(item);
    // Get the pointers to the data
    float *ptra, *ptrb, *out;
    int owidth, aheight, awidth, bheight, adatawidth;
    get_ptrs_pwdist(item, ptra, ptrb, out, owidth, aheight, awidth, bheight, adatawidth);

    // Reset the execution time
    item->execution_time = 0.0;
    // Create the event info
    sycl::event m_event;

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
    } else {
        std::cout << "ERROR: pwdist_GPU: type not supported" << std::endl;
        exit(1);
    }

    // Save the execution time and end tracing
    save_time_info_on_sycl(item, inputArgs, m_event, 2, "GPU_S");
    trace_end(item, my_tracer, "GPU");

    return SyclEventInfo(m_event, item->execution_time, Acc::GPU);
}

// Explicit template instantiation
template SyclEventInfo pwdist_GPU<0, basic>(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on);
template SyclEventInfo pwdist_GPU<16, sycl::float4>(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on);
template SyclEventInfo pwdist_GPU<16, float>(ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on);