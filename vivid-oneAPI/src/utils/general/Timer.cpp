#include "Timer.hpp"

void startTimerIfNeeded(ApplicationData &appData, InputArgs &inputArgs) {
    if (inputArgs.hasDuration()) {
        std::thread timer([&inputArgs, &appData]() {
            if constexpr (VERBOSE_ENABLED) {
                std::cout << " Set timer for " << inputArgs.duration.count() << " seconds" << std::endl;
            }
            std::this_thread::sleep_for(inputArgs.duration);
            inputArgs.numFrames = appData.id;
            inputArgs.duration = std::chrono::seconds(0);
            if constexpr (VERBOSE_ENABLED) {
                std::cout << " Timer finished" << std::endl;
            }
        });
        timer.detach(); // Detach the thread so it runs in the background
    }
}

void startTimeMeasurement(ApplicationData &appData, InputArgs &inputArgs) {
    if (inputArgs.hasTimeSampling()) {
        std::thread timer_sampling([&inputArgs, &appData]() {
            if constexpr (VERBOSE_ENABLED) {
                std::cout << " Set timer for " << inputArgs.timeSampling.count() << " seconds" << std::endl;
            }
            std::this_thread::sleep_for(inputArgs.timeSampling);
            inputArgs.timeSampling = std::chrono::seconds(0);
            if constexpr (VERBOSE_ENABLED) {
                std::cout << " Timer finished" << std::endl;
            }
        });
        timer_sampling.detach(); // Detach the thread so it runs in the background
    }
}