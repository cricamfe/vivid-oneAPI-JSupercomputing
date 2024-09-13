#pragma once

#include "GlobalParameters.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
#include <tbb/tick_count.h>

inline void save_time_info_on_sycl(ViVidItem *item, InputArgs &inputArgs, sycl::event &m_event, int stage, const std::string &accStr) {
    if constexpr (TRACE_ENABLED || TIMESTAGES_ENABLED || AUTOMODE_ENABLED || ADVANCEDMETRICS_ENABLED) {
        if (inputArgs.pipelineName != PipelineType::SYCLEvents) {
            item->execution_time = (m_event.get_profiling_info<sycl::info::event_profiling::command_end>() - m_event.get_profiling_info<sycl::info::event_profiling::command_start>());
            if (accStr == "CPU_S") {
                item->timeCPU_S[stage] = item->execution_time * 1e-6;
            } else if (accStr == "GPU_S") {
                item->timeGPU_S[stage] = item->execution_time * 1e-6;
            }
        }
    }
}

inline void start_timer(ViVidItem *item) {
    if constexpr (TIMESTAGES_ENABLED || TRACE_ENABLED || AUTOMODE_ENABLED || ADVANCEDMETRICS_ENABLED) {
        item->filter_start = tbb::tick_count::now();
    }
}

inline void trace_start(ViVidItem *item, Tracer &tracer, const std::string &accStr) {
    if constexpr (TRACE_ENABLED) {
        if (accStr == "CPU") {
            tracer.cpu_start(item);
        } else if (accStr == "GPU") {
            tracer.gpu_start(item);
        }
    }
}

inline void wait_sycl_event(sycl::event &m_event) {
    if constexpr (TRACE_ENABLED) {
        m_event.wait();
    }
}

inline void save_trace_info(ViVidItem *item) {
    if constexpr (TRACE_ENABLED) {
        item->execution_time = (tbb::tick_count::now() - item->filter_start).seconds();
    }
}

inline void trace_end(ViVidItem *item, Tracer &tracer, const std::string &accStr) {
    if constexpr (TRACE_ENABLED) {
        if (accStr == "CPU") {
            tracer.cpu_end(item, item->execution_time);
        } else if (accStr == "GPU") {
            tracer.gpu_end(item, item->execution_time);
        }
    }
}

inline void save_time_info_normal(ViVidItem *item, int stage, const std::string &accStr) {
    if constexpr (TIMESTAGES_ENABLED || AUTOMODE_ENABLED || ADVANCEDMETRICS_ENABLED) {
        if (accStr == "CPU_S") {
            item->timeCPU_S[stage] += (tbb::tick_count::now() - item->filter_start).seconds() * 1000;
        } else if (accStr == "GPU_S") {
            item->timeGPU_S[stage] += (tbb::tick_count::now() - item->filter_start).seconds() * 1000;
        }
    }
}

inline void get_ptrs_cosine(ViVidItem *item, float *&ptr_frame, float *&ptr_ind, float *&ptr_val, int &f_pitch_f) {
    ptr_frame = item->frame->get_HOST_PTR(BUF_READ);
    ptr_ind = item->ind->get_HOST_PTR(BUF_WRITE);
    ptr_val = item->val->get_HOST_PTR(BUF_WRITE);
    f_pitch_f = item->frame->pitch / sizeof(float);
}

inline void get_ptrs_histogram(ViVidItem *item, ApplicationData &appData, float *&ptr_his, float *&ptr_val, float *&ptr_ind, int &histogram_pitch_f, int &assignments_pitch_f, int &weights_pitch_f) {
    ptr_his = item->his->get_HOST_PTR(BUF_WRITE);
    ptr_val = item->val->get_HOST_PTR(BUF_READ);
    ptr_ind = item->ind->get_HOST_PTR(BUF_READ);
    histogram_pitch_f = item->his->pitch / sizeof(float);
    assignments_pitch_f = item->ind->pitch / sizeof(float);
    weights_pitch_f = item->val->pitch / sizeof(float);
}

inline void get_ptrs_pwdist(ViVidItem *item, float *&ptra, float *&ptrb, float *&out, int &owidth, int &aheight, int &awidth, int &bheight, int &adatawidth) {
    ptra = item->cla->get_HOST_PTR(BUF_READ);
    ptrb = item->his->get_HOST_PTR(BUF_READ);
    out = item->out->get_HOST_PTR(BUF_WRITE);
    owidth = item->out->pitch / sizeof(float);
    aheight = item->cla->height;
    awidth = item->cla->pitch / sizeof(float);
    bheight = item->his->height;
    adatawidth = item->cla->width;
}
