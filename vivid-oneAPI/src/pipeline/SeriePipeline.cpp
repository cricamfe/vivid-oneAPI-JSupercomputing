#include "SeriePipeline.hpp"
#include "Timer.hpp"
#include "execute_code.hpp" // Asegúrate de incluir este archivo para acceder a las funciones dentro del espacio de nombres Details

template <typename Func>
SyclEventInfo executeStage(Func func, Acc acc, ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, sycl::queue &Q_CPU, std::vector<sycl::event> *depends_on = nullptr) {
    return func(acc, item, traceFile, appData, inputArgs, acc == Acc::GPU ? Q_GPU : Q_CPU, depends_on);
}

void SeriePipeline::executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    // Print implementation information
    if constexpr (VERBOSE_ENABLED) {
        std::cout << " Running SERIAL version..." << std::endl;
        std::cout << " Device: " << (inputArgs.configStagesStr == "GPU" ? "GPU" : "CPU") << (inputArgs.useDependsOnSerial ? " with depends_on" : "") << std::endl;
    }

    // Common variables for the pipeline
    ViVidItem *item;
    tick_count filter_start;
    SyclEventInfo eventInfo_S1, eventInfo_S2, eventInfo_S3;
    Acc acc = inputArgs.configStagesStr == "GPU" ? Acc::GPU : Acc::CPU;

    // Start the timer
    appData.pipeline_start = tbb::tick_count::now();
    startTimerIfNeeded(appData, inputArgs);

    while (appData.id < inputArgs.numFrames || inputArgs.hasDuration()) {
        item = bufferItems.get();
        appData.id++;
        item->item_id = appData.id;
        if constexpr (TRACE_ENABLED) {
            traceFile.frame_start(item);
        }

        // Stage 1
        eventInfo_S1 = executeStage(cosinefilter, acc, item, traceFile, appData, inputArgs, Q_GPU, Q_CPU, inputArgs.useDependsOnSerial ? &item->stage_events : nullptr);

        // Stage 2
        eventInfo_S2 = executeStage(blockhistogram, acc, item, traceFile, appData, inputArgs, Q_GPU, Q_CPU, inputArgs.useDependsOnSerial ? &item->stage_events : nullptr);

        // Stage 3
        eventInfo_S3 = executeStage(pwdist, acc, item, traceFile, appData, inputArgs, Q_GPU, Q_CPU, inputArgs.useDependsOnSerial ? &item->stage_events : nullptr);

        // Measure the time of the stages
        if constexpr (TIMESTAGES_ENABLED) {
            timeMeasurements(appData, item);
        }

        // Release the item to the buffer
        bufferItems.recycle(item);

        // End the frame trace
        if constexpr (TRACE_ENABLED) {
            traceFile.frame_end(item);
        }
    }

    appData.pipeline_end = tbb::tick_count::now();
}
