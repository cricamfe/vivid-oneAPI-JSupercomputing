#pragma once
#ifndef INPUT_ARGS_HPP
#define INPUT_ARGS_HPP

#include "../CLI11.hpp"
#include "GlobalParameters.hpp"
#include "ResourcesManager.hpp"
#include "WorkloadSimulator.hpp"
#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

class InputArgs {
  public:
    // Basic arguments that must be entered (some of them have default values)
    PipelineType pipelineName{PipelineType::ParallelPipeline};                                         //< Pipeline type (SeriePipeline, ParallelPipeline, FlowGraphFunctionalNode, FlowGraphAsyncNode, SYCLEvents) (Default: 1)
    int numFrames{DEFAULT_NUM_FRAMES};                                                                 //< Number of frames to process (Default: 1000)
    int imageResolution{DEFAULT_IMAGE_RESOLUTION};                                                     //< Image resolution 0: 1080p, 2: 1440p, 3: 2160p, 4: 2880p, 5: 4320p (Default: 1)
    int nThreads{DEFAULT_NUM_THREADS};                                                                 //< Number of threads to use (Default: 8)
    std::chrono::duration<double> duration{0};                                                         //< Duration in seconds (if user wants to use it)
    std::chrono::duration<double> timeSampling{0};                                                     //< Time sampling for model
    std::vector<StageState> stageExecutionState{std::vector<StageState>(NUM_STAGES, StageState::CPU)}; //< 0: CPU, 1: GPU, 2: CPU and GPU
    std::string configStagesStr{DEFAULT_CONFIG_STAGES};                                                //< Configuration of the stages (Default: 000)

    // Optional arguments that can be entered (some of them have default values)
    int inFlightFrames{0};                                                   //< Number of frames in flight (Default: -1)
    size_t sizeCircularBuffer{0};                                            //< Size of the circular buffer
    bool useDependsOnSerial{false};                                          //< Use SYCL depends_on with SerialPipeline (Default: false)
    std::vector<double> throughput_CPU{std::vector<double>(NUM_STAGES, -1)}; //< Throughput of the CPU in each stage (workload simulation)
    std::vector<double> throughput_GPU{std::vector<double>(NUM_STAGES, -1)}; //< Throughput of the GPU in each stage (workload simulation)

    // Internal variables that are used to control the behavior of the pipelines
    bool GPUactive{false};                               //< GPU active (Default: false)
    PathSelection selectedPath = PathSelection::Coupled; //< Path selected (Coupled, Decoupled, CustomCoupled) (Default: Coupled)

    // Control and management of semaphores (cores and task queues)
    std::unique_ptr<ResourcesManager> resourcesManager;                               //< Resources manager
    std::vector<Acc> executionDevicePriority{std::vector<Acc>(NUM_STAGES, Acc::CPU)}; //< Prefer GPU in each stage (false: CPU, true: GPU) (Default: false)

    bool preferGpu = true;

    // Number of frames used on sampling
    int sampleFrames{0};                    //< Number of frames used on sampling
    bool timeConvergence{false};            //< Time convergence
    int iteration{0};                       //< Iteration
    const int minIteration{5};              //< Minimum iteration before convergence
    const double convergenceThreshold{0.1}; //< Convergence umbral

    InputArgs(int argc, char *argv[]);
    const std::string getImageTypeToString() const;
    std::string getPrefDevice() const {
        std::string result;
        for (const auto &dev : executionDevicePriority) {
            result += (dev == Acc::GPU ? "1" : "0");
        }
        return result;
    }
    bool hasDuration() const { return duration.count() > 0; }
    bool hasTimeSampling() const { return timeSampling.count() > 0; }
    void printArguments() const;

  private:
    void parseArguments(int argc, char *argv[]);
    void setResources(const std::vector<int> &size, const std::vector<int> &cores, Acc acc);
    void setExecutionDevicePriority(const std::vector<int> &exeDevPriority, std::vector<Acc> &executionDevicePriority);
    void setThroughput(const std::vector<double> &th, std::vector<double> &throughput);
    bool parseConfigStages();
};

#endif // INPUT_ARGS_HPP
