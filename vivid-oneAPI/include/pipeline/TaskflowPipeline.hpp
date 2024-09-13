#pragma once
#include "ApplicationData.hpp"
#include "InputArgs.hpp"
#include "PipelineInterface.hpp"
#include "Tracer.hpp"
#include <functional>
#include <sycl/sycl.hpp>
#include <taskflow/algorithm/pipeline.hpp>
#include <taskflow/taskflow.hpp>
#include <vector>

template <std::size_t N>
class TaskflowPipeline : public PipelineInterface {
  public:
    void executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) override;

  private:
    SyclEventInfo processStage(std::size_t stage, Acc acc, ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, sycl::queue &Q_CPU);
};