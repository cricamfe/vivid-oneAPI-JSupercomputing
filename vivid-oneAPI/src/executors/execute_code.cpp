// execute_code.cpp
#include "execute_code.hpp"
#include "GlobalParameters.hpp"
#include "WorkloadSimulator.hpp"
#include "common_macros.hpp"

namespace Details {
// *********************************************************************************************************************
// FILTER 1:
// *********************************************************************************************************************
template <>
SyclEventInfo cosinefilter<Acc::CPU>(Pipeline_template::ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    SyclEventInfo my_event = cosinefilter_CPU(item, my_tracer, appData, inputArgs, Q, depends_on);
    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        appData.numFiltersCPU[0]++;
    }
    return my_event;
}

template <>
SyclEventInfo cosinefilter<Acc::GPU>(Pipeline_template::ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    SyclEventInfo my_event = cosinefilter_GPU(item, my_tracer, appData, inputArgs, Q, depends_on);
    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        appData.numFiltersGPU[0]++;
    }
    return my_event;
}

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
template <>
SyclEventInfo blockhistogram<Acc::CPU>(Pipeline_template::ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    SyclEventInfo my_event = blockhistogram_CPU(item, my_tracer, appData, inputArgs, Q, depends_on);
    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        appData.numFiltersCPU[1]++;
    }
    return my_event;
}

template <>
SyclEventInfo blockhistogram<Acc::GPU>(Pipeline_template::ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    SyclEventInfo my_event = blockhistogram_GPU(item, my_tracer, appData, inputArgs, Q, depends_on);
    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        appData.numFiltersGPU[1]++;
    }
    return my_event;
}

// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
template <>
SyclEventInfo pwdist<Acc::CPU>(Pipeline_template::ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
#if __PWDIST__ == 1
    constexpr size_t tile_size = 64;
    using T = float;
#elif __PWDIST__ == 2
    constexpr size_t tile_size = 64;
    using T = sycl::float4;
#elif __PWDIST__ == 3
    constexpr size_t tile_size = 0;
    using T = basic;
#else
    constexpr size_t tile_size = 64;
    using T = float;
#endif
    SyclEventInfo my_event = pwdist_CPU<tile_size, T>(item, my_tracer, appData, inputArgs, Q, depends_on);
    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        appData.numFiltersCPU[2]++;
    }
    return my_event;
}

template <>
SyclEventInfo pwdist<Acc::GPU>(Pipeline_template::ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
#if __PWDIST__ == 1
    constexpr size_t tile_size = 16;
    using T = float;
#elif __PWDIST__ == 2
    constexpr size_t tile_size = 16;
    using T = sycl::float4;
#elif __PWDIST__ == 3
    constexpr size_t tile_size = 0;
    using T = basic;
#else
    constexpr size_t tile_size = 16;
    using T = sycl::float4;
#endif
    SyclEventInfo my_event = pwdist_GPU<tile_size, T>(item, my_tracer, appData, inputArgs, Q, depends_on);
    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        appData.numFiltersGPU[2]++;
    }
    return my_event;
}
} // namespace Details

// General dispatch function
SyclEventInfo cosinefilter(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    if (acc == Acc::GPU) {
        return Details::cosinefilter<Acc::GPU>(item, my_tracer, appData, inputArgs, Q, depends_on);
    } else {
        return Details::cosinefilter<Acc::CPU>(item, my_tracer, appData, inputArgs, Q, depends_on);
    }
}

// General dispatch function
SyclEventInfo blockhistogram(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    if (acc == Acc::GPU) {
        return Details::blockhistogram<Acc::GPU>(item, my_tracer, appData, inputArgs, Q, depends_on);
    } else {
        return Details::blockhistogram<Acc::CPU>(item, my_tracer, appData, inputArgs, Q, depends_on);
    }
}

// General dispatch function
SyclEventInfo pwdist(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
    if (acc == Acc::GPU) {
        return Details::pwdist<Acc::GPU>(item, my_tracer, appData, inputArgs, Q, depends_on);
    } else {
        return Details::pwdist<Acc::CPU>(item, my_tracer, appData, inputArgs, Q, depends_on);
    }
}

SyclEventInfo workloadsimulator(Acc acc, ViVidItem *item, Tracer &my_tracer, ApplicationData &appData, InputArgs &inputArgs, int stage) {
    trace_start(item, my_tracer, (acc == Acc::GPU) ? "GPU" : "CPU");
    start_timer(item);

    // Get double value for the throughput
    double throughput = (acc == Acc::GPU) ? inputArgs.throughput_GPU[stage] : inputArgs.throughput_CPU[stage];
    int num_threads = (acc == Acc::GPU) ? 1 : (SYCL_ENABLED ? 1 : inputArgs.nThreads);

    // Simulate the workload
    WorkloadSimulator workloadSimulator(throughput, num_threads);
    workloadSimulator.simulate();

    if constexpr (ENERGYPCM_ENABLED || AUTOMODE_ENABLED || TIMESTAGES_ENABLED) {
        if (acc == Acc::GPU) {
            appData.numFiltersGPU[stage]++;
        } else {
            appData.numFiltersCPU[stage]++;
        }
    }

    // Save the trace information and execution time
    save_trace_info(item);
    save_time_info_normal(item, stage, (acc == Acc::GPU) ? "GPU_S" : "CPU_S");

    // End tracing
    trace_end(item, my_tracer, (acc == Acc::GPU) ? "GPU" : "CPU");
    return SyclEventInfo();
}
