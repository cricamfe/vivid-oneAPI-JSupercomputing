#ifndef PIPELINE_FACTORY_HPP
#define PIPELINE_FACTORY_HPP

#include "FlowGraphPipeline.hpp"
#include "GlobalParameters.hpp"
#include "ParallelPipeline.hpp"
#include "PipelineInterface.hpp"
#include "SYCLEventsPipeline.hpp"
#include "SeriePipeline.hpp"
#include "TaskflowPipeline.hpp"
#include <memory>
#include <stdexcept>
#include <string>

class PipelineFactory {
  public:
    static std::unique_ptr<PipelineInterface> createPipeline(PipelineType type);
    static PipelineType getPipelineType(const std::string &pipelineSelected);
    static std::string getPipelineTypeString(PipelineType type);
    static std::string getPipelineTypeAsShortString(PipelineType type);
};

#endif // PIPELINE_FACTORY_HPP
