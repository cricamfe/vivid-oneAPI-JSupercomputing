// PipelineInterface.hpp
#pragma once
#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include "SYCLUtils.hpp"
#include "TimeMeasurements.hpp"
#include "Timer.hpp"
#include "Tracer.hpp"
#include "circular-buffer.hpp"

// Forward declaration EnergyPCM
class EnergyPCM;

class PipelineInterface {
  public:
    virtual ~PipelineInterface() = default;
    virtual void executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM = nullptr) = 0;

    // Public wrapper for reduceCountersAfterProcessing
    void publicReduceCountersAfterProcessing(const InputArgs &inputArgs, const ApplicationData &appData, Acc accelerator, int index, sycl::queue *Q = nullptr, sycl::event *event = nullptr, std::vector<sycl::event> *vectorEvents = nullptr);

  protected:
    Acc selectPath(InputArgs &inputArgs, int index, bool &isGPUFrame, ViVidItem *item = nullptr, Tracer *tracer = nullptr);
    Acc selectPathDecoupled(InputArgs &inputArgs, int index, bool &isGPUFrame);
    Acc selectPathCoupled(InputArgs &inputArgs, int index, bool &isGPUFrame);

    // Function to reduce counters after processing
    void reduceCountersAfterProcessing(const InputArgs &inputArgs, const ApplicationData &appData, Acc accelerator, int index, sycl::queue *Q = nullptr, sycl::event *event = nullptr, std::vector<sycl::event> *vectorEvents = nullptr);

    // Function to process input nodes
    ViVidItem *processInputNode(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile);

    // New helper functions for improved modularity
    void logProcessing(ViVidItem *item);
    void adjustCountersAfterProcessing(ViVidItem *item, InputArgs &inputArgs, ApplicationData &appData, Acc accelerator = Acc::OTHER, sycl::event *event = nullptr, sycl::queue *Q_CPU = nullptr, sycl::queue *Q_GPU = nullptr);
    void handleTimeMeasurements(ViVidItem *item, ApplicationData &appData, InputArgs &inputArgs, Acc accelerator = Acc::OTHER);
    bool isAutoModeEnabled(ApplicationData &appData, ViVidItem *item, InputArgs &inputArgs);
    void optimizePipeline(ApplicationData &appData, InputArgs &inputArgs);
    void debugAndTrace(ViVidItem *item, ApplicationData &appData, Tracer &traceFile);
    void recycleItem(circular_buffer &bufferItems, ViVidItem *item);
};