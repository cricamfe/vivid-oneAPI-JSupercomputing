#include "ParallelPipeline.hpp"
#include "GlobalParameters.hpp"
#include "Timer.hpp"
#include <oneapi/tbb.h>
#include <unordered_map>

template <std::size_t N>
template <typename FilterType>
void ParallelPipeline<N>::addStage(FilterType &filter, ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU) {
    for (std::size_t stage = 0; stage < N; ++stage) {
        filter = filter & oneapi::tbb::make_filter<ViVidItem *, ViVidItem *>(oneapi::tbb::filter_mode::parallel,
                                                                             [&, stage](ViVidItem *item) -> ViVidItem * {
                                                                                 tbb::tick_count filter_start = tbb::tick_count::now();

                                                                                 logProcessing("Start: Processing item " + std::to_string(item->item_id) + " in stage " + std::to_string(stage));

                                                                                 Acc acc = selectPathWrapper(inputArgs, stage, item->GPU_item, item, &traceFile);
                                                                                 SyclEventInfo eventInfo = processStage(stage, acc, item, traceFile, appData, inputArgs, Q_GPU, Q_CPU);

                                                                                 logProcessing("End: Processing item " + std::to_string(item->item_id) + " in stage " + std::to_string(stage) + " took " + std::to_string((tbb::tick_count::now() - filter_start).seconds()) + " seconds");

                                                                                 reduceCountersAfterProcessing(inputArgs, appData, acc, stage);

                                                                                 return item;
                                                                             });
    }
}

template <std::size_t N>
SyclEventInfo ParallelPipeline<N>::processStage(std::size_t stage, Acc acc, ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, sycl::queue &Q_CPU) {
    using StageFunction = std::function<SyclEventInfo(Acc, ViVidItem *, Tracer &, ApplicationData &, InputArgs &, sycl::queue &, std::vector<sycl::event> *)>;
    static const std::unordered_map<std::size_t, StageFunction> stageFunctions = {
        {0, cosinefilter},
        {1, blockhistogram},
        {2, pwdist}};

    if constexpr (USE_VIVID_APP) {
        if (stageFunctions.find(stage) != stageFunctions.end()) {
            return stageFunctions.at(stage)(acc, item, traceFile, appData, inputArgs, (acc == Acc::GPU) ? Q_GPU : Q_CPU, nullptr);
        }
    } else {
        return workloadsimulator(acc, item, traceFile, appData, inputArgs, stage);
    }
    return {};
}

template <std::size_t N>
void ParallelPipeline<N>::logProcessing(const std::string &message) {
    if constexpr (LOG_ENABLED) {
        std::clog << message << " with thread " << std::this_thread::get_id() << ".\n";
    }
}

template <std::size_t N>
Acc ParallelPipeline<N>::selectPathWrapper(InputArgs &inputArgs, std::size_t stage, bool GPU_item, ViVidItem *item, Tracer *traceFile) {
    Acc acc = selectPath(inputArgs, stage, GPU_item, item, traceFile);
    logProcessing("Selected path for item " + std::to_string(item->item_id) + " in stage " + std::to_string(stage) + ": " + (acc == Acc::GPU ? "GPU" : "CPU"));
    return acc;
}

template <std::size_t N>
void ParallelPipeline<N>::executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Running PARALLEL_PIPELINE version with " << N << " stages..." << std::endl;
    }

    if (inputArgs.pipelineName != PipelineType::SYCLEvents) {
        tbb::global_control global_limit{tbb::global_control::max_allowed_parallelism, static_cast<size_t>(inputArgs.nThreads + inputArgs.GPUactive)};
    }

    if constexpr (LOG_ENABLED) {
        inputArgs.resourcesManager->startMonitoring();
    }

    appData.pipeline_start = tbb::tick_count::now();

    // We ensure the executions always last the same, regardless of whether the automatic mode is enabled or not
    if constexpr (AUTOMODE_ENABLED) {
        startTimeMeasurement(appData, inputArgs);
    } else {
        startTimerIfNeeded(appData, inputArgs);
    }

    auto pipeline = oneapi::tbb::make_filter<void, ViVidItem *>(oneapi::tbb::filter_mode::serial_in_order,
                                                                [&](oneapi::tbb::flow_control &fc) -> ViVidItem * {
                                                                    if (appData.id < inputArgs.numFrames || inputArgs.hasDuration()) {
                                                                        ViVidItem *item = processInputNode(appData, inputArgs, bufferItems, traceFile);
                                                                        logProcessing("Processing item " + std::to_string(appData.id) + " in the input node");
                                                                        return item;
                                                                    } else {
                                                                        fc.stop();
                                                                        return nullptr;
                                                                    }
                                                                });

    addStage(pipeline, appData, inputArgs, traceFile, Q_GPU, Q_CPU);

    auto outputFilter = oneapi::tbb::make_filter<ViVidItem *, void>(oneapi::tbb::filter_mode::serial_out_of_order,
                                                                    [&](ViVidItem *item) {
                                                                        // processOutputNode(item, appData, inputArgs, bufferItems, traceFile);

                                                                        // Save the previous number of tokens
                                                                        int prev_tokens;
                                                                        if constexpr (AUTOMODE_ENABLED) {
                                                                            prev_tokens = inputArgs.inFlightFrames;
                                                                        }

                                                                        // logProcessing(item);
                                                                        adjustCountersAfterProcessing(item, inputArgs, appData);
                                                                        handleTimeMeasurements(item, appData, inputArgs);
                                                                        if (isAutoModeEnabled(appData, item, inputArgs)) {
                                                                            optimizePipeline(appData, inputArgs);
                                                                            inputArgs.inFlightFrames = prev_tokens;
                                                                        }

                                                                        // Check debug, trace and recycle the item
                                                                        debugAndTrace(item, appData, traceFile);
                                                                        recycleItem(bufferItems, item);
                                                                    });

    oneapi::tbb::parallel_pipeline(inputArgs.inFlightFrames, pipeline & outputFilter);

    if constexpr (LOG_ENABLED) {
        inputArgs.resourcesManager->stopMonitoring();
    }
    appData.pipeline_end = tbb::tick_count::now();
}

template class ParallelPipeline<NUM_STAGES>;