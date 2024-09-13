#ifndef TIME_MEASUREMENTS_HPP
#define TIME_MEASUREMENTS_HPP

#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "pipeline_template.hpp"
#include <sycl/sycl.hpp>

inline void timeMeasurements_advanced(ApplicationData &appData, ViVidItem *item) {
    for (int i = 0; i < NUM_STAGES; ++i) {
        if (item->timeGPU_S[i] > 0) {
            appData.time_GPU_S[i] += item->timeGPU_S[i];
            if constexpr (TIMESTAGES_ENABLED) {
                appData.numGPUframes++;
            } else if constexpr (ADVANCEDMETRICS_ENABLED) {
                appData.numFiltersGPU[i]++;
            }
        }
        if (item->timeCPU_S[i] > 0) {
            appData.time_CPU_S[i] += item->timeCPU_S[i];
            if constexpr (TIMESTAGES_ENABLED) {
                appData.numCPUframes++;
            } else if constexpr (ADVANCEDMETRICS_ENABLED) {
                appData.numFiltersCPU[i]++;
            }
        }
    }
}

inline void timeMeasurements_basic(ApplicationData &appData, ViVidItem *item) {
    auto &frameRef = item->GPU_item ? appData.numGPUframes : appData.numCPUframes;
    auto &timeRef = item->GPU_item ? appData.time_GPU_S : appData.time_CPU_S;
    auto &itemTimeRef = item->GPU_item ? item->timeGPU_S : item->timeCPU_S;
    frameRef++;
    for (int i = 0; i < NUM_STAGES; ++i) {
        timeRef[i] += itemTimeRef[i];
    }
}

inline void timeMeasurements_syclevents(ApplicationData &appData, ViVidItem *item, Acc accelerator) {
    int counter = 0;
    for (const auto &event : item->stage_events) {
        const Acc stage_acc = item->stage_acc[counter]; // Cachear el valor de stage_acc
        const bool isGPU = (stage_acc == Acc::GPU);
        const bool isCPU = (stage_acc == Acc::CPU);

        if constexpr (USE_VIVID_APP) {
            if (isGPU) {
                // Calcular tiempos para GPU en modo VIVID_APP
                auto command_end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                auto command_start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                appData.time_GPU_S[counter] += (command_end - command_start) * 1e-6;
            } else if (isCPU) {
                if constexpr (SYCL_ENABLED) {
                    // Calcular tiempos para CPU en modo SYCL
                    auto command_end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    auto command_start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    appData.time_CPU_S[counter] += (command_end - command_start) * 1e-6;
                } else {
                    // Usar tiempos predefinidos si no está SYCL habilitado
                    appData.time_CPU_S[counter] += item->timeCPU_S[counter];
                }
            }
        } else {
            // Si no estamos en VIVID_APP, usar tiempos predefinidos
            if (isGPU) {
                appData.time_GPU_S[counter] += item->timeGPU_S[counter];
            } else if (isCPU) {
                appData.time_CPU_S[counter] += item->timeCPU_S[counter];
            }
        }
        counter++;
    }
}

inline void timeMeasurements(ApplicationData &appData, ViVidItem *item, Acc accelerator = Acc::OTHER) {
    if (accelerator == Acc::OTHER) {
        timeMeasurements_advanced(appData, item);
    } else {
        timeMeasurements_syclevents(appData, item, accelerator);
    }
}

#endif // TIME_MEASUREMENTS_HPP
