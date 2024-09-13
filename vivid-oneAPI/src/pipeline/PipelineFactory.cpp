#include "PipelineFactory.hpp"
#include "GlobalParameters.hpp"

std::unique_ptr<PipelineInterface> PipelineFactory::createPipeline(PipelineType type) {
    switch (type) {
    case PipelineType::ParallelPipeline:
        return std::make_unique<ParallelPipeline<NUM_STAGES>>();
    case PipelineType::FlowGraphFunctionalNode:
        return std::make_unique<FlowGraphPipeline<FunctionalNode, NUM_STAGES>>();
    case PipelineType::FlowGraphAsyncNode:
        return std::make_unique<FlowGraphPipeline<AsyncNode, NUM_STAGES>>();
    case PipelineType::SYCLEvents:
        return std::make_unique<SYCLEventsPipeline<NUM_STAGES>>();
    case PipelineType::Taskflow:
        return std::make_unique<TaskflowPipeline<NUM_STAGES>>();
    case PipelineType::Serie:
        return std::make_unique<SeriePipeline>();
    default:
        throw std::invalid_argument("Invalid pipeline type");
    }
}

PipelineType PipelineFactory::getPipelineType(const std::string &pipelineSelected) {
    if (pipelineSelected == "pipeline") {
        return PipelineType::ParallelPipeline;
    } else if (pipelineSelected == "fgfn") {
        return PipelineType::FlowGraphFunctionalNode;
    } else if (pipelineSelected == "fgan") {
        return PipelineType::FlowGraphAsyncNode;
    } else if (pipelineSelected == "syclevents") {
        return PipelineType::SYCLEvents;
    } else if (pipelineSelected == "taskflow") {
        return PipelineType::Taskflow;
    } else if (pipelineSelected == "serie") {
        return PipelineType::Serie;
    } else {
        throw std::invalid_argument("Invalid pipeline type string");
    }
}

std::string PipelineFactory::getPipelineTypeString(PipelineType type) {
    switch (type) {
    case PipelineType::ParallelPipeline:
        return "Pipeline";
    case PipelineType::FlowGraphFunctionalNode:
        return "FlowGraph Functional Node";
    case PipelineType::FlowGraphAsyncNode:
        return "FlowGraph Async Node";
    case PipelineType::SYCLEvents:
        return "SYCL Events";
    case PipelineType::Taskflow:
        return "Taskflow";
    case PipelineType::Serie:
        return "Serie";
    default:
        throw std::invalid_argument("Invalid pipeline type");
    }
}

std::string PipelineFactory::getPipelineTypeAsShortString(PipelineType type) {
    switch (type) {
    case PipelineType::ParallelPipeline:
        return "pipeline";
    case PipelineType::FlowGraphFunctionalNode:
        return "fgfn";
    case PipelineType::FlowGraphAsyncNode:
        return "fgan";
    case PipelineType::SYCLEvents:
        return "syclevents";
    case PipelineType::Taskflow:
        return "taskflow";
    case PipelineType::Serie:
        return "serie";
    default:
        throw std::invalid_argument("Invalid pipeline type");
    }
}
