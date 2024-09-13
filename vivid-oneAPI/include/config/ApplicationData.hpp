#ifndef APPLICATION_DATA_HPP
#define APPLICATION_DATA_HPP

#include "pipeline_template.hpp"
#include <array>
#include <atomic>
#include <random>
#include <sycl/sycl.hpp>
#include <tbb/tick_count.h>
#include <vector>

using namespace Pipeline_template;

class ApplicationData {
  public:
    int height = 0;
    int width = 0;
    const int numFilters = 100;
    const int window_height = 128;
    const int windowWidth = 64;
    const int cellSize = 8;
    const int blockSize = 2;
    const int dictSize = 100;
    const int filterDim = 3;
    const int filterSize = 9;
    float *goldenFrame = nullptr;

    bool autoMode = true;

    sycl::queue USM_queue;

    int id = 0;

    // Buffers
    FloatBuffer *globalFrame = nullptr;
    FloatBuffer *globalCla = nullptr;
    float *filterBank = nullptr;

    // Number of frames to process in the optimization
    std::atomic<int> numGPUframes = 0;
    std::atomic<int> numCPUframes = 0;

    // Vectors of atomic integers for filters on GPU and CPU
    std::vector<std::atomic<int>> numFiltersGPU; // Atomic integers for GPU filters
    std::vector<std::atomic<int>> numFiltersCPU; // Atomic integers for CPU filters

    // Variables relacionadas con el tiempo
    std::vector<double> time_GPU_S{{0.0, 0.0, 0.0}};
    std::vector<double> time_CPU_S{{0.0, 0.0, 0.0}};
    float totalTime{0.0f};
    float sampleTime{0.0f};
    float systemTime{0.0f};
    float throughput{0.0f};
    float throughputBalance{0.0f};
    float throughputSystem{0.0f};
    float throughputSystemExpected{0.0f};

    // Variables para guardar el tiempo
    tbb::tick_count pipeline_start, pipeline_end;
    tbb::tick_count system_start, sample_end;

    // Random seed generator
    std::random_device seed;
    std::mt19937 mte{seed()};

    // ViVidItem for debugging
    ViVidItem *item_debug = nullptr;

    float energyCPU{0.0f};
    float energyGPU{0.0f};
    float energyUncore{0.0f};
    float energyTotal{0.0f};
    float actAvgFreq{0.0f};
    float avgTemperature{0.0f};
    float minTemperature{0.0f};
    float maxTemperature{0.0f};
    float avgPower_W{0.0f};
    float totalKilowattHours{0.0f};

    // Constructor to initialize the atomic vectors
    ApplicationData()
        : numFiltersGPU(NUM_STAGES), numFiltersCPU(NUM_STAGES) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            numFiltersGPU[i] = 0;
            numFiltersCPU[i] = 0;
        }
    }

    void selectUSMQueue(sycl::queue &Q);

    ~ApplicationData();
};

#endif // APPLICATION_DATA_HPP
