// SYCLEventsPipeline.cpp
#include "SYCLEventsPipeline.hpp"
#include "InputArgs.hpp"
#include <future>

// Define mutexes for input and output synchronization
std::mutex input_mutex;  ///< Mutex for protecting the ID counter and other critical sections
std::mutex output_mutex; ///< Mutex for protecting the output node

/**
 * @brief Executes the SYCL pipeline with the given application data, input arguments, and SYCL queues.
 *
 * @tparam N Number of stages in the pipeline.
 * @param appData Application data.
 * @param inputArgs Input arguments.
 * @param bufferItems Circular buffer containing items to process.
 * @param traceFile Trace file for logging.
 * @param Q_GPU SYCL queue for GPU.
 * @param Q_CPU SYCL queue for CPU.
 * @param energyPCM Optional energy PCM pointer.
 */
template <std::size_t N>
void SYCLEventsPipeline<N>::executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Running SYCL Events Pipeline with " << N << " stages" << std::endl;
    }

    // // Set the global control for TBB to limit the maximum allowed parallelism
    // tbb::global_control global_limit{tbb::global_control::max_allowed_parallelism, static_cast<size_t>(inputArgs.nThreads + inputArgs.GPUactive)};

    // Initialize the in-flight frames device
    inFlightFrames = std::make_unique<Device>(Acc::OTHER, inputArgs.nThreads + inputArgs.GPUactive);
    inFlightFrames->addStage(0, inputArgs.nThreads + inputArgs.GPUactive, inputArgs.inFlightFrames);

    // Start the pipeline timer
    appData.pipeline_start = tbb::tick_count::now();
    // We ensure the executions always last the same, regardless of whether the automatic mode is enabled or not
    if constexpr (AUTOMODE_ENABLED) {
        startTimeMeasurement(appData, inputArgs);
    } else {
        startTimerIfNeeded(appData, inputArgs);
    }

    // Vector para almacenar los futuros de los frames en vuelo
    std::vector<std::future<void>> framesInFlight;
    framesInFlight.reserve(inputArgs.inFlightFrames);

    // Lanza tareas asíncronas para procesar imágenes
    for (int i = 0; i < inputArgs.inFlightFrames; ++i) {
        framesInFlight.push_back(std::async(std::launch::async, &SYCLEventsPipeline::processImage, this,
                                            std::ref(appData), std::ref(inputArgs), std::ref(traceFile), std::ref(bufferItems),
                                            std::ref(appData.filterBank), std::ref(Q_GPU), std::ref(Q_CPU), energyPCM));
    }

    // Espera a que todas las tareas finalicen
    for (auto &frame : framesInFlight) {
        frame.get(); // Bloquea hasta que la tarea esté completa y maneja cualquier excepción
    }

    // Stop the pipeline timer
    appData.pipeline_end = tbb::tick_count::now();
}

/**
 * @brief Runs a stage wrapper function for a given stage.
 *
 * @tparam StageFunc Type of the stage function.
 * @param acc Accelerator type.
 * @param item Item to process.
 * @param traceFile Trace file for logging.
 * @param inputArgs Input arguments.
 * @param appData Application data.
 * @param Q SYCL queue.
 * @param stage_func Stage function to execute.
 * @param eventInfo SYCL event info.
 * @param stage_ID Stage identifier.
 */
template <std::size_t N>
template <typename StageFunc>
void SYCLEventsPipeline<N>::runStageWrapper(Acc acc, ViVidItem *item, Tracer &traceFile, InputArgs &inputArgs, ApplicationData &appData, sycl::queue &Q, StageFunc stage_func, SyclEventInfo &eventInfo, int stage_ID) {
    try {
        if constexpr (USE_VIVID_APP) {
            // Run the stage function depending on the accelerator type
            if (acc == Acc::GPU || (acc == Acc::CPU && SYCL_ENABLED)) {
                eventInfo = stage_func(acc, item, traceFile, appData, inputArgs, Q, &(item->stage_events));
            } else if (acc == Acc::CPU) {
                eventInfo.event = Q.submit([&](sycl::handler &cgh) {
                    if (!item->stage_events.empty()) {
                        cgh.depends_on(item->stage_events);
                    }
                    stage_func(acc, item, traceFile, appData, inputArgs, Q, &(item->stage_events));
                });
            }
        } else {
            eventInfo.event = Q.submit([&](sycl::handler &cgh) {
                if (!item->stage_events.empty()) {
                    cgh.depends_on(item->stage_events);
                }
                workloadsimulator(acc, item, traceFile, appData, inputArgs, stage_ID);
            });
        }
        item->stage_acc.push_back(acc);
        item->stage_events.push_back(eventInfo.getEvent());
        if (inputArgs.selectedPath != PathSelection::Decoupled) {
            reduceCountersAfterProcessing(inputArgs, appData, acc, stage_ID, &Q, &(eventInfo.event), &(item->stage_events));
        }
    } catch (const sycl::exception &e) {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
    }
}

/**
 * @brief Processes an image in the pipeline.
 *
 * @param appData Application data.
 * @param inputArgs Input arguments.
 * @param traceFile Trace file for logging.
 * @param bufferItems Circular buffer containing items to process.
 * @param filter_bank Filter bank for processing.
 * @param Q_GPU SYCL queue for GPU.
 * @param Q_CPU SYCL queue for CPU.
 * @param energyPCM Optional energy PCM pointer.
 */
template <std::size_t N>
void SYCLEventsPipeline<N>::processImage(ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, circular_buffer &bufferItems, float *filter_bank, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    while (appData.id < inputArgs.numFrames || inputArgs.hasDuration()) {
        ViVidItem *item = nullptr;
        reserveFrameInFlight();
        {
            std::lock_guard<std::mutex> lock(input_mutex);
            item = processInputNode(appData, inputArgs, bufferItems, traceFile);
        }
        releaseFrameInFlight();

        addStages(item, appData, inputArgs, traceFile, Q_GPU, Q_CPU);

        // Wait for all stages to finish
        sycl::event::wait_and_throw(item->stage_events);

        reserveFrameInFlight();
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            // Save the previous number of tokens
            int prev_tokens;
            if constexpr (AUTOMODE_ENABLED) {
                prev_tokens = inputArgs.inFlightFrames;
            }

            // Get the accelerator used for the item
            Acc acc = item->GPU_item ? Acc::GPU : Acc::CPU;

            if (inputArgs.selectedPath == PathSelection::Decoupled) {
                // Get the last event in the stage events vector
                sycl::event *last_event = &item->stage_events.back();
                reduceCountersAfterProcessing(inputArgs, appData, acc, -1, &(acc == Acc::GPU ? Q_GPU : Q_CPU), last_event, &item->stage_events);
            }

            handleTimeMeasurements(item, appData, inputArgs, acc);
            if (isAutoModeEnabled(appData, item, inputArgs)) {
                std::cout << "Launching optimization" << std::endl;
                optimizePipeline(appData, inputArgs);
                std::cout << "Optimization finished" << std::endl;
            }

            // Check debug, trace and recycle the item
            debugAndTrace(item, appData, traceFile);
            recycleItem(bufferItems, item);
        }
        releaseFrameInFlight();
    }
}

/**
 * @brief Gets the stage function for the specified stage.
 *
 * @param stage Stage number.
 * @return StageFunction The stage function.
 */
template <std::size_t N>
typename SYCLEventsPipeline<N>::StageFunction SYCLEventsPipeline<N>::getStageFunction(size_t stage) {
    if constexpr (USE_VIVID_APP) {
        return stageFunctions.at(stage);
    } else {
        return [stage](Acc acc, ViVidItem *item, Tracer &trace_file, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q, std::vector<sycl::event> *depends_on) {
            return workloadsimulator(acc, item, trace_file, appData, inputArgs, stage);
        };
    }
}

/**
 * @brief Adds stages to the pipeline.
 *
 * @param item Item to process.
 * @param appData Application data.
 * @param inputArgs Input arguments.
 * @param traceFile Trace file for logging.
 * @param Q_GPU SYCL queue for GPU.
 * @param Q_CPU SYCL queue for CPU.
 */
template <std::size_t N>
void SYCLEventsPipeline<N>::addStages(ViVidItem *item, ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU) {
    // Initialize stage functions based on the application mode
    stageFunctions = USE_VIVID_APP ? std::unordered_map<std::size_t, StageFunction>{
                                         {0, cosinefilter},
                                         {1, blockhistogram},
                                         {2, pwdist}}
                                   : std::unordered_map<std::size_t, StageFunction>{};

    // Iterate through the stages and process each stage
    for (size_t i = 0; i < N; ++i) {
        reserveFrameInFlight();
        Acc acc = selectPath(inputArgs, i, item->GPU_item, item, &traceFile);
        StageFunction stage_func = getStageFunction(i);
        SyclEventInfo eventInfo;
        runStageWrapper(acc, item, traceFile, inputArgs, appData, (acc == Acc::GPU ? Q_GPU : Q_CPU), stage_func, eventInfo, i);
        releaseFrameInFlight();
    }
}

/**
 * @brief Reserves a frame in flight.
 */
template <std::size_t N>
void SYCLEventsPipeline<N>::reserveFrameInFlight() {
    inFlightFrames->acquireCore(0);
}

/**
 * @brief Releases a frame in flight.
 */
template <std::size_t N>
void SYCLEventsPipeline<N>::releaseFrameInFlight() {
    inFlightFrames->release(0);
}

// Explicit template instantiation
template class SYCLEventsPipeline<NUM_STAGES>;