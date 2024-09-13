#pragma once
#ifndef RESULTS_HPP
#define RESULTS_HPP

#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include <iomanip>
#include <iostream>

inline void calculateAndDisplayResults(ApplicationData &appData, const InputArgs &inputArgs) {
    // Compute the total time and throughput
    appData.totalTime = (appData.pipeline_end - appData.pipeline_start).seconds() * 1000;
    appData.throughput = (inputArgs.numFrames * 1000) / appData.totalTime;

    // Compute the system time and throughput if auto mode is enabled
    if constexpr (AUTOMODE_ENABLED) {
        appData.systemTime = (appData.pipeline_end - appData.system_start).seconds() * 1000;
        // int num_frames = inputArgs.numFrames - inputArgs.numFrames * ((AVX_ENABLED || SIMD_ENABLED) ? TEST_FRAMES_VECTORIZED : TEST_FRAMES_BASIC);
        int totalNumFramesSystem = inputArgs.numFrames - inputArgs.sampleFrames;
        std::cout << "System time: " << appData.systemTime << " ms" << std::endl;
        std::cout << "Num frames: " << inputArgs.numFrames << " - Sample frames: " << inputArgs.sampleFrames << " - After balance: " << totalNumFramesSystem << std::endl;
        appData.throughputSystem = (totalNumFramesSystem * 1000) / appData.systemTime;
    }

    auto safe_divide = [](double numerator, int denominator) -> double {
        return (denominator == 0) ? 0 : numerator / denominator;
    };

    if constexpr (ADVANCEDMETRICS_ENABLED) {
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << " TOTAL FILTERS BY DEVICE" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << " GPU: " << appData.numFiltersGPU[0].load() << " - " << appData.numFiltersGPU[1].load() << " - " << appData.numFiltersGPU[2].load() << std::endl;
        std::cout << " CPU: " << appData.numFiltersCPU[0].load() << " - " << appData.numFiltersCPU[1].load() << " - " << appData.numFiltersCPU[2].load() << std::endl;
    }

    if constexpr (ADVANCEDMETRICS_ENABLED || TIMESTAGES_ENABLED) {
        int totalNumFiltersGPU = appData.numFiltersGPU[0].load() + appData.numFiltersGPU[1].load() + appData.numFiltersGPU[2].load();
        int totalNumFiltersCPU = appData.numFiltersCPU[0].load() + appData.numFiltersCPU[1].load() + appData.numFiltersCPU[2].load();
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << " TIME PER STAGE ( GPU frames: " << totalNumFiltersGPU << "; CPU frames: " << totalNumFiltersCPU << " )" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        if (inputArgs.selectedPath == PathSelection::Decoupled && !TIMESTAGES_ENABLED) {
            std::string stagesStr;
            for (int i = 1; i <= NUM_STAGES; i++) {
                stagesStr += std::to_string(i);
                if (i < NUM_STAGES) {
                    stagesStr += "-";
                }
            }
            if (totalNumFiltersGPU > 0) {
                std::cout << " GPU Time: " << safe_divide(appData.time_GPU_S[0] + appData.time_GPU_S[1] + appData.time_GPU_S[2], totalNumFiltersGPU) << " ms" << std::endl;
                std::cout << " - Stage " << stagesStr << ":\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_GPU_S[0], appData.numFiltersGPU[0].load()) << " ms" << std::endl;
            }
            if (totalNumFiltersCPU > 0) {
                std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                std::cout << " CPU Time: " << safe_divide(appData.time_CPU_S[0] + appData.time_CPU_S[1] + appData.time_CPU_S[2], totalNumFiltersCPU) << " ms" << std::endl;
                std::cout << " - Stage " << stagesStr << ":\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_CPU_S[0], appData.numFiltersCPU[0].load()) << " ms" << std::endl;
            }
        } else {
            if (totalNumFiltersGPU > 0) {
                std::cout << " GPU Time: " << safe_divide(appData.time_GPU_S[0] + appData.time_GPU_S[1] + appData.time_GPU_S[2], totalNumFiltersGPU) << " ms" << std::endl;
                std::cout << " - Stage 1:\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_GPU_S[0], appData.numFiltersGPU[0].load()) << " ms" << std::endl;
                std::cout << " - Stage 2:\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_GPU_S[1], appData.numFiltersGPU[1].load()) << " ms" << std::endl;
                std::cout << " - Stage 3:\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_GPU_S[2], appData.numFiltersGPU[2].load()) << " ms" << std::endl;
            }
            if (totalNumFiltersCPU > 0) {
                std::cout << "---------------------------------------------------------------------------------------" << std::endl;
                std::cout << " CPU Time: " << safe_divide(appData.time_CPU_S[0] + appData.time_CPU_S[1] + appData.time_CPU_S[2], totalNumFiltersCPU) << " ms" << std::endl;
                std::cout << " - Stage 1:\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_CPU_S[0], appData.numFiltersCPU[0].load()) << " ms" << std::endl;
                std::cout << " - Stage 2:\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_CPU_S[1], appData.numFiltersCPU[1].load()) << " ms" << std::endl;
                std::cout << " - Stage 3:\t" << std::setprecision(2) << std::fixed << safe_divide(appData.time_CPU_S[2], appData.numFiltersCPU[2].load()) << " ms" << std::endl;
            }
        }
    }

    // Print the basic results (throughput and total time)
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << " RESULTS" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Throughput: \t" << std::setprecision(2) << std::fixed << appData.throughput << " FPS" << std::endl;
    std::cout << " Total time: \t" << std::setprecision(2) << std::fixed << appData.totalTime << " ms" << std::endl;

    if constexpr (AUTOMODE_ENABLED) {
        std::cout << "---------------------------------------------------------------------------------------" << std::endl;
        std::cout << " Throughput (balance): \t" << std::setprecision(2) << std::fixed << appData.throughputBalance << " FPS" << std::endl;
        std::cout << " Throughput (system): \t" << std::setprecision(2) << std::fixed << appData.throughputSystem << " FPS" << std::endl;
        std::cout << " Throughput (expect): \t" << std::setprecision(2) << std::fixed << appData.throughputSystemExpected << " FPS" << std::endl;
        std::cout << " Balance time: \t" << std::setprecision(2) << std::fixed << appData.sampleTime << " ms" << std::endl;
        std::cout << " System time: \t" << std::setprecision(2) << std::fixed << appData.systemTime << " ms" << std::endl;
    }
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
}

#endif // RESULTS_HPP
