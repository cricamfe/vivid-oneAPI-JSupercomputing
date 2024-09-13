// SYCLEventsPipeline.hpp
#pragma once
#include "Comparer.hpp"
#include "Device.hpp"
#include "PipelineInterface.hpp"
#include "Timer.hpp"
#include "execute_code.hpp"
#include <functional>
#include <mutex>
#include <oneapi/tbb.h>
#include <sycl/sycl.hpp>
#include <thread>
#include <unordered_map>
#include <vector>

// Mutexes for protecting critical sections
extern std::mutex input_mutex;  ///< Mutex for protecting the ID counter and other critical sections
extern std::mutex output_mutex; ///< Mutex for protecting the output node

/**
 * @brief SYCLEventsPipeline class template for managing and executing the SYCL pipeline.
 *
 * @tparam N Number of stages in the pipeline.
 */
template <std::size_t N>
class SYCLEventsPipeline : public PipelineInterface {
  public:
    /**
     * @brief Execute the SYCL pipeline.
     *
     * @param appData Application data.
     * @param inputArgs Input arguments.
     * @param bufferItems Circular buffer containing items to process.
     * @param traceFile Trace file for logging.
     * @param Q_GPU SYCL queue for GPU.
     * @param Q_CPU SYCL queue for CPU.
     * @param energyPCM Optional energy PCM pointer.
     */
    void executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM = nullptr) override;

  private:
    /**
     * @brief Type definition for the stage function.
     */
    using StageFunction = std::function<SyclEventInfo(Acc, ViVidItem *, Tracer &, ApplicationData &, InputArgs &, sycl::queue &, std::vector<sycl::event> *)>;

    std::unique_ptr<Device> inFlightFrames;                        ///< Device for managing in-flight frames.
    std::unordered_map<std::size_t, StageFunction> stageFunctions; ///< Map of stage functions.

    /**
     * @brief Run a stage wrapper.
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
    template <typename StageFunc>
    void runStageWrapper(Acc acc, ViVidItem *item, Tracer &traceFile, InputArgs &inputArgs, ApplicationData &appData, sycl::queue &Q, StageFunc stage_func, SyclEventInfo &eventInfo, int stage_ID);

    /**
     * @brief Process an image.
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
    void processImage(ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, circular_buffer &bufferItems, float *filter_bank, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM = nullptr);

    /**
     * @brief Add stages to the pipeline.
     *
     * @param item Item to process.
     * @param appData Application data.
     * @param inputArgs Input arguments.
     * @param traceFile Trace file for logging.
     * @param Q_GPU SYCL queue for GPU.
     * @param Q_CPU SYCL queue for CPU.
     */
    void addStages(ViVidItem *item, ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU);

    /**
     * @brief Get the stage function for the specified stage.
     *
     * @param stage Stage number.
     * @return StageFunction The stage function.
     */
    StageFunction getStageFunction(size_t stage);

    /**
     * @brief Reserve a frame in flight.
     */
    void reserveFrameInFlight();

    /**
     * @brief Release a frame in flight.
     */
    void releaseFrameInFlight();
};
