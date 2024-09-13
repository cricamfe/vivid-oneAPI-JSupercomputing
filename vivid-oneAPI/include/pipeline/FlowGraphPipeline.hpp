// FlowGraphPipeline.hpp
#pragma once
#include "Comparer.hpp"
#include "PipelineInterface.hpp"
#include "Timer.hpp"
#include "execute_code.hpp"
#include <functional>
#include <oneapi/tbb.h>
#include <unordered_map>

// Declare the node types
struct FunctionalNode {};
struct AsyncNode {};

// Declare the types used on the Flow Graph Pipeline
using token_t = int;
using join_t = tbb::flow::join_node<std::tuple<ViVidItem *, token_t>, tbb::flow::reserving>;
using indexer_t = tbb::flow::indexer_node<ViVidItem *, ViVidItem *>;
using mfn_two_inputs_t = tbb::flow::multifunction_node<std::tuple<ViVidItem *, token_t>, std::tuple<ViVidItem *, ViVidItem *>>;
using mfn_one_input_t = tbb::flow::multifunction_node<indexer_t::output_type, std::tuple<ViVidItem *, ViVidItem *>>;
using CPUNode_t = tbb::flow::function_node<ViVidItem *, ViVidItem *>;

// Define some elements used on the Async Node version
using FGPU_t = tbb::flow::async_node<ViVidItem *, ViVidItem *>;
using gateway_type = FGPU_t::gateway_type;
using StageFunction = std::function<SyclEventInfo(Acc, ViVidItem *, Tracer &, ApplicationData &, InputArgs &, sycl::queue &, std::vector<sycl::event> *)>;

// Async GPU Node Definitions
class FGPU {
    tbb::task_arena a;
    std::function<SyclEventInfo(ViVidItem *, Tracer &, ApplicationData &, InputArgs &, sycl::queue &)> kernel_func;
    int stage;

  public:
    FGPU(std::function<SyclEventInfo(ViVidItem *, Tracer &, ApplicationData &, InputArgs &, sycl::queue &)> kernel_func, int stage);
    void submit(gateway_type &gateway, ViVidItem *item, Tracer &trace_file, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline);
};

// Create the Flow Graph Pipeline
template <typename NodeType, std::size_t N>
class FlowGraphPipeline : public PipelineInterface {
  public:
    void executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM = nullptr) override;

  private:
    void setupPipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM);
    void initTokenBuffer(tbb::flow::buffer_node<int> &token_buffer, InputArgs &inputArgs);
    auto create_Input_Node(tbb::flow::graph &g, ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile);
    auto create_Output_Node(tbb::flow::graph &g, tbb::flow::buffer_node<int> &token_buffer, ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, EnergyPCM *energyPCM);
    auto create_Indexer_Node(tbb::flow::graph &g);

    template <std::size_t Stage>
    void addStage(tbb::flow::graph &g, InputArgs &inputArgs, Tracer &traceFile, ApplicationData &appData, sycl::queue &Q_GPU, sycl::queue &Q_CPU, std::vector<indexer_t *> &indexers, std::vector<std::shared_ptr<void>> &nodes_storage);

    template <int Stage>
    auto create_Splitter_Node(tbb::flow::graph &g, InputArgs &inputArgs, Tracer &traceFile);

    template <int Stage>
    auto create_CPU_Node(tbb::flow::graph &g, ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, sycl::queue &Q_CPU);

    template <typename GPU_NodeType, int Stage>
    auto create_GPU_Node(tbb::flow::graph &g, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline);

    template <int Stage>
    auto create_GPU_Node_with_AN(tbb::flow::graph &g, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline);

    template <int Stage>
    auto create_GPU_Node_with_FN(tbb::flow::graph &g, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline);

    std::vector<std::shared_ptr<void>> nodes_storage_;
};