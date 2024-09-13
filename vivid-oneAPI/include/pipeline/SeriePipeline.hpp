#pragma once
#include "Comparer.hpp"
#include "PipelineInterface.hpp"
#include "execute_code.hpp"
#include <oneapi/tbb.h>

class SeriePipeline : public PipelineInterface {
  public:
    void executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM = nullptr) override;
};
