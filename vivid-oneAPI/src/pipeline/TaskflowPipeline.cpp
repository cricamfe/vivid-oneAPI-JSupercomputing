#include "TaskflowPipeline.hpp"
#include "Timer.hpp"
#include "execute_code.hpp"
#include "taskflow/algorithm/pipeline.hpp"
#include "taskflow/taskflow.hpp"

template <std::size_t N>
void TaskflowPipeline<N>::executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << " Running SCALABLE_PIPELINE version..." << std::endl;
    }

    tf::Executor executor(inputArgs.nThreads + inputArgs.GPUactive);
    tf::Taskflow taskflow;

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

    std::vector<ViVidItem *> buffer(inputArgs.inFlightFrames, nullptr);

    auto input_pipe = [&](tf::Pipeflow &pf) {
        if (appData.id < inputArgs.numFrames || inputArgs.hasDuration()) {
            ViVidItem *item = this->processInputNode(appData, inputArgs, bufferItems, traceFile);
            if constexpr (LOG_ENABLED) {
                std::clog << "Processing item " << appData.id << " in the input node with thread " << std::this_thread::get_id() << ".\n";
            }
            buffer[pf.line()] = item;
        } else {
            pf.stop();
        }
    };

    auto make_stage_pipe = [&](std::size_t i) {
        return [&, i](tf::Pipeflow &pf) {
            ViVidItem *item = buffer[pf.line()];
            if (item) {
                tick_count filter_start = tbb::tick_count::now();
                SyclEventInfo eventInfo;
                if constexpr (LOG_ENABLED) {
                    std::clog << "Processing item " << item->item_id << " in stage " << i << " with thread " << std::this_thread::get_id() << ".\n";
                }
                Acc acc = this->selectPath(inputArgs, i, item->GPU_item, item, &traceFile);
                eventInfo = processStage(i, acc, item, traceFile, appData, inputArgs, Q_GPU, Q_CPU);
                this->reduceCountersAfterProcessing(inputArgs, appData, acc, i);
                if constexpr (LOG_ENABLED) {
                    std::clog << "Processing item " << item->item_id << " in stage " << i << " finished with thread " << std::this_thread::get_id() << ".\n";
                }
            }
        };
    };

    // Crear la tubería de salida
    auto output_pipe = [&](tf::Pipeflow &pf) {
        ViVidItem *item = buffer[pf.line()];
        if (item) {
            // Save the previous number of tokens
            int prev_tokens;
            if constexpr (AUTOMODE_ENABLED) {
                prev_tokens = inputArgs.inFlightFrames;
            }

            // logProcessing(item);
            this->adjustCountersAfterProcessing(item, inputArgs, appData);
            this->handleTimeMeasurements(item, appData, inputArgs);
            if (this->isAutoModeEnabled(appData, item, inputArgs)) {
                this->optimizePipeline(appData, inputArgs);
                inputArgs.inFlightFrames = prev_tokens;
            }

            // Check debug, trace and recycle the item
            this->debugAndTrace(item, appData, traceFile);
            this->recycleItem(bufferItems, item);
        }
    };

    // Crear un vector de pipes
    std::vector<tf::Pipe<std::function<void(tf::Pipeflow &)>>> pipes;

    // Input pipe - SERIAL
    pipes.emplace_back(tf::PipeType::SERIAL, input_pipe);

    for (std::size_t i = 0; i < N; ++i) {
        pipes.emplace_back(tf::PipeType::PARALLEL, make_stage_pipe(i));
    }

    // Output pipe - SERIAL
    pipes.emplace_back(tf::PipeType::SERIAL, output_pipe);

    tf::ScalablePipeline<decltype(pipes)::iterator> pipeline(inputArgs.inFlightFrames, pipes.begin(), pipes.end());

    tf::Task init = taskflow.emplace([]() {}).name("Start taskflow");
    tf::Task pipeline_task = taskflow.composed_of(pipeline).name("Pipeline");
    tf::Task stop = taskflow.emplace([]() {}).name("Stop taskflow");

    init.precede(pipeline_task);
    pipeline_task.precede(stop);

    executor.run(taskflow).wait();

    if constexpr (LOG_ENABLED) {
        inputArgs.resourcesManager->stopMonitoring();
    }
    appData.pipeline_end = tbb::tick_count::now();
}

template <std::size_t N>
SyclEventInfo TaskflowPipeline<N>::processStage(std::size_t stage, Acc acc, ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, sycl::queue &Q_CPU) {
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

// Explicit instantiation
template class TaskflowPipeline<NUM_STAGES>;