#pragma once
#include "Comparer.hpp"
#include "GlobalParameters.hpp"
#include "PipelineInterface.hpp"
#include "execute_code.hpp"
#include <functional>
#include <oneapi/tbb.h>
#include <unordered_map>

template <std::size_t N>
class ParallelPipeline : public PipelineInterface {
  public:
    void executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM = nullptr) override;

  private:
    template <typename FilterType>
    void addStage(FilterType &filter, ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU);

    SyclEventInfo processStage(std::size_t stage, Acc acc, ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, sycl::queue &Q_CPU);
    void logProcessing(const std::string &message);
    Acc selectPathWrapper(InputArgs &inputArgs, std::size_t stage, bool GPU_item, ViVidItem *item, Tracer *traceFile);
};