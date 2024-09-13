// FlowGraphPipeline.cpp
#include "FlowGraphPipeline.hpp"
#include "Queue.hpp"

// Async GPU Node Definitions
FGPU::FGPU(std::function<SyclEventInfo(ViVidItem *, Tracer &, ApplicationData &, InputArgs &, sycl::queue &)> kernel_func, int stage) : kernel_func(kernel_func), stage(stage) {
    a.initialize(1, 0);
}

void FGPU::submit(gateway_type &gateway, ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline) {
    gateway.reserve_wait();
    a.execute([this, &item, &gateway, &traceFile, &appData, &inputArgs, &Q_GPU, &pipeline]() {
        SyclEventInfo eventInfo = kernel_func(item, traceFile, appData, inputArgs, Q_GPU);
        pipeline.publicReduceCountersAfterProcessing(inputArgs, appData, Acc::GPU, stage);
        gateway.try_put(item);
        gateway.release_wait();
    });
}

template <typename NodeType, std::size_t N>
void FlowGraphPipeline<NodeType, N>::executePipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    setupPipeline(appData, inputArgs, bufferItems, traceFile, Q_GPU, Q_CPU, energyPCM);
}

template <typename NodeType, std::size_t N>
void FlowGraphPipeline<NodeType, N>::setupPipeline(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, sycl::queue &Q_GPU, sycl::queue &Q_CPU, EnergyPCM *energyPCM) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << " Running FLOW GRAPH " << (std::is_same<NodeType, FunctionalNode>::value ? "FUNCTIONAL NODE" : "ASYNC NODE") << " version..." << std::endl;
    }

    tbb::global_control global_limit{tbb::global_control::max_allowed_parallelism, static_cast<size_t>(inputArgs.nThreads + inputArgs.GPUactive)};
    tbb::flow::graph g;

    auto in_node = create_Input_Node(g, appData, inputArgs, bufferItems, traceFile);
    tbb::flow::buffer_node<token_t> token_buffer{g};
    join_t join{g};

    std::vector<indexer_t *> indexers;

    // Configurar la primera etapa manualmente
    using GPUNode_t = decltype(create_GPU_Node<NodeType, 0>(g, traceFile, appData, inputArgs, Q_GPU, *this));

    // Configure the first stage manually
    auto gpu_cpu_split = std::make_shared<mfn_two_inputs_t>(create_Splitter_Node<0>(g, inputArgs, traceFile));
    auto filter_cpu = std::make_shared<CPUNode_t>(create_CPU_Node<0>(g, appData, inputArgs, traceFile, Q_CPU));
    auto filter_gpu = std::make_shared<GPUNode_t>(create_GPU_Node<NodeType, 0>(g, traceFile, appData, inputArgs, Q_GPU, *this));
    auto async_join = std::make_shared<indexer_t>(create_Indexer_Node(g));

    tbb::flow::make_edge(in_node, tbb::flow::input_port<0>(join));
    tbb::flow::make_edge(token_buffer, tbb::flow::input_port<1>(join));
    tbb::flow::make_edge(join, *gpu_cpu_split);
    tbb::flow::make_edge(tbb::flow::output_port<0>(*gpu_cpu_split), *filter_gpu);
    tbb::flow::make_edge(tbb::flow::output_port<1>(*gpu_cpu_split), *filter_cpu);
    tbb::flow::make_edge(*filter_gpu, tbb::flow::input_port<0>(*async_join));
    tbb::flow::make_edge(*filter_cpu, tbb::flow::input_port<1>(*async_join));

    indexers.push_back(async_join.get());
    nodes_storage_.reserve(N * 4);

    // Configurar las siguientes etapas utilizando recursión de plantillas
    addStage<1>(g, inputArgs, traceFile, appData, Q_GPU, Q_CPU, indexers, nodes_storage_);

    auto out_node = create_Output_Node(g, token_buffer, appData, inputArgs, bufferItems, traceFile, energyPCM);
    tbb::flow::make_edge(*indexers.back(), out_node);
    tbb::flow::make_edge(out_node, token_buffer);

    initTokenBuffer(token_buffer, inputArgs);

    appData.pipeline_start = tbb::tick_count::now();
    // We ensure the executions always last the same, regardless of whether the automatic mode is enabled or not
    if constexpr (AUTOMODE_ENABLED) {
        startTimeMeasurement(appData, inputArgs);
    } else {
        startTimerIfNeeded(appData, inputArgs);
    }
    in_node.activate();
    g.wait_for_all();
    appData.pipeline_end = tbb::tick_count::now();
}

template <typename NodeType, std::size_t N>
template <std::size_t Stage>
void FlowGraphPipeline<NodeType, N>::addStage(tbb::flow::graph &g, InputArgs &inputArgs, Tracer &traceFile, ApplicationData &appData, sycl::queue &Q_GPU, sycl::queue &Q_CPU, std::vector<indexer_t *> &indexers, std::vector<std::shared_ptr<void>> &nodes_storage) {
    if constexpr (Stage < N) {
        // Get the types of the nodes to be created
        using GPUNode_t = decltype(create_GPU_Node<NodeType, Stage>(g, traceFile, appData, inputArgs, Q_GPU, *this));

        auto gpu_cpu_split = std::make_shared<mfn_one_input_t>(create_Splitter_Node<Stage>(g, inputArgs, traceFile));
        auto filter_cpu = std::make_shared<CPUNode_t>(create_CPU_Node<Stage>(g, appData, inputArgs, traceFile, Q_CPU));
        auto filter_gpu = std::make_shared<GPUNode_t>(create_GPU_Node<NodeType, Stage>(g, traceFile, appData, inputArgs, Q_GPU, *this));
        auto async_join = std::make_shared<indexer_t>(create_Indexer_Node(g));

        tbb::flow::make_edge(*indexers.back(), *gpu_cpu_split);
        tbb::flow::make_edge(tbb::flow::output_port<0>(*gpu_cpu_split), *filter_gpu);
        tbb::flow::make_edge(tbb::flow::output_port<1>(*gpu_cpu_split), *filter_cpu);
        tbb::flow::make_edge(*filter_gpu, tbb::flow::input_port<0>(*async_join));
        tbb::flow::make_edge(*filter_cpu, tbb::flow::input_port<1>(*async_join));

        indexers.push_back(async_join.get());

        nodes_storage.push_back(gpu_cpu_split);
        nodes_storage.push_back(filter_cpu);
        nodes_storage.push_back(filter_gpu);
        nodes_storage.push_back(async_join);

        addStage<Stage + 1>(g, inputArgs, traceFile, appData, Q_GPU, Q_CPU, indexers, nodes_storage);
    }
}

template <typename NodeType, std::size_t N>
auto FlowGraphPipeline<NodeType, N>::create_Input_Node(tbb::flow::graph &g, ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile) {
    return tbb::flow::input_node<ViVidItem *>{g, [&](tbb::flow_control &fc) -> ViVidItem * {
                                                  if (appData.id < inputArgs.numFrames || inputArgs.hasDuration()) {
                                                      ViVidItem *item = processInputNode(appData, inputArgs, bufferItems, traceFile);
                                                      return item;
                                                  } else {
                                                      fc.stop();
                                                      return nullptr;
                                                  }
                                              }};
}

template <typename NodeType, std::size_t N>
template <int Stage>
auto FlowGraphPipeline<NodeType, N>::create_Splitter_Node(tbb::flow::graph &g, InputArgs &inputArgs, Tracer &traceFile) {
    if constexpr (Stage == 0) {
        return mfn_two_inputs_t{
            g, tbb::flow::unlimited, [&](const std::tuple<ViVidItem *, token_t> &v, typename mfn_two_inputs_t::output_ports_type &ports) {
                ViVidItem *item = std::get<0>(v);
                Acc acc = selectPath(inputArgs, Stage, item->GPU_item, item, &traceFile);
                if (acc == Acc::GPU) {
                    std::get<0>(ports).try_put(item);
                } else {
                    std::get<1>(ports).try_put(item);
                }
            }};
    } else {
        return mfn_one_input_t{
            g, tbb::flow::unlimited, [&](const typename indexer_t::output_type &v, typename mfn_one_input_t::output_ports_type &ports) {
                ViVidItem *item = tbb::flow::cast_to<ViVidItem *>(v);
                Acc acc = selectPath(inputArgs, Stage, item->GPU_item, item, &traceFile);
                if (acc == Acc::GPU) {
                    std::get<0>(ports).try_put(item);
                } else {
                    std::get<1>(ports).try_put(item);
                }
            }};
    }
}

template <typename NodeType, std::size_t N>
template <int Stage>
auto FlowGraphPipeline<NodeType, N>::create_CPU_Node(tbb::flow::graph &g, ApplicationData &appData, InputArgs &inputArgs, Tracer &traceFile, sycl::queue &Q_CPU) {
    return CPUNode_t{g, tbb::flow::unlimited, [&](ViVidItem *item) -> ViVidItem * {
                         SyclEventInfo eventInfo;
                         if (USE_VIVID_APP) {
                             if constexpr (Stage == 0) {
                                 eventInfo = Details::cosinefilter<Acc::CPU>(item, traceFile, appData, inputArgs, Q_CPU);
                             } else if constexpr (Stage == 1) {
                                 eventInfo = Details::blockhistogram<Acc::CPU>(item, traceFile, appData, inputArgs, Q_CPU);
                             } else if constexpr (Stage == 2) {
                                 eventInfo = Details::pwdist<Acc::CPU>(item, traceFile, appData, inputArgs, Q_CPU);
                             }
                         } else {
                             eventInfo = workloadsimulator(Acc::CPU, item, traceFile, appData, inputArgs, Stage);
                         }
                         reduceCountersAfterProcessing(inputArgs, appData, Acc::CPU, Stage);
                         return item;
                     }};
}

template <typename NodeType, std::size_t N>
auto FlowGraphPipeline<NodeType, N>::create_Indexer_Node(tbb::flow::graph &g) {
    return indexer_t{g};
}

template <typename NodeType, std::size_t N>
auto FlowGraphPipeline<NodeType, N>::create_Output_Node(tbb::flow::graph &g, tbb::flow::buffer_node<int> &token_buffer, ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile, EnergyPCM *energyPCM) {
    return tbb::flow::function_node<indexer_t::output_type, token_t>{g, 1, [&](const auto &v) -> token_t {
                                                                         ViVidItem *item = tbb::flow::cast_to<ViVidItem *>(v);
                                                                         // Save the previous number of tokens
                                                                         int prev_tokens;
                                                                         if constexpr (AUTOMODE_ENABLED) {
                                                                             prev_tokens = inputArgs.inFlightFrames;
                                                                         }

                                                                         logProcessing(item);
                                                                         adjustCountersAfterProcessing(item, inputArgs, appData);
                                                                         handleTimeMeasurements(item, appData, inputArgs);
                                                                         if (isAutoModeEnabled(appData, item, inputArgs)) {
                                                                             optimizePipeline(appData, inputArgs);
                                                                             if constexpr (!SYCL_ENABLED) {
                                                                                 int newTokens = inputArgs.inFlightFrames - prev_tokens;
                                                                                 if (newTokens > 0) {
                                                                                     for (int i = 0; i < newTokens; ++i) {
                                                                                         token_buffer.try_put(0);
                                                                                     }
                                                                                 }
                                                                             }
                                                                         }

                                                                         // Check debug, trace and recycle the item
                                                                         debugAndTrace(item, appData, traceFile);
                                                                         recycleItem(bufferItems, item);

                                                                         return (item->GPU_item ? 0 : 1);
                                                                     }};
}

template <typename NodeType, std::size_t N>
template <typename GPU_NodeType, int Stage>
auto FlowGraphPipeline<NodeType, N>::create_GPU_Node(tbb::flow::graph &g, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline) {
    if constexpr (std::is_same_v<NodeType, AsyncNode>) {
        return create_GPU_Node_with_AN<Stage>(g, traceFile, appData, inputArgs, Q_GPU, pipeline);
    } else if constexpr (std::is_same_v<NodeType, FunctionalNode>) {
        return create_GPU_Node_with_FN<Stage>(g, traceFile, appData, inputArgs, Q_GPU, pipeline);
    }
}

template <typename NodeType, std::size_t N>
template <int Stage>
auto FlowGraphPipeline<NodeType, N>::create_GPU_Node_with_AN(tbb::flow::graph &g, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline) {
    if constexpr (USE_VIVID_APP) {
        if constexpr (Stage == 0) {
            return FGPU_t{g, tbb::flow::unlimited, [&](ViVidItem *item, gateway_type &gateway) {
                              FGPU([](ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU) {
                                  return Details::cosinefilter<Acc::GPU>(item, traceFile, appData, inputArgs, Q_GPU);
                              },
                                   0)
                                  .submit(gateway, item, traceFile, appData, inputArgs, Q_GPU, pipeline);
                          }};
        } else if constexpr (Stage == 1) {
            return FGPU_t{g, tbb::flow::unlimited, [&](ViVidItem *item, gateway_type &gateway) {
                              FGPU([](ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU) {
                                  return Details::blockhistogram<Acc::GPU>(item, traceFile, appData, inputArgs, Q_GPU);
                              },
                                   1)
                                  .submit(gateway, item, traceFile, appData, inputArgs, Q_GPU, pipeline);
                          }};
        } else if constexpr (Stage == 2) {
            return FGPU_t{g, tbb::flow::unlimited, [&](ViVidItem *item, gateway_type &gateway) {
                              FGPU([](ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU) {
                                  return Details::pwdist<Acc::GPU>(item, traceFile, appData, inputArgs, Q_GPU);
                              },
                                   2)
                                  .submit(gateway, item, traceFile, appData, inputArgs, Q_GPU, pipeline);
                          }};
        }
    } else {
        return FGPU_t{g, tbb::flow::unlimited, [&](ViVidItem *item, gateway_type &gateway) {
                          FGPU([=](ViVidItem *item, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU) {
                              return workloadsimulator(Acc::GPU, item, traceFile, appData, inputArgs, Stage);
                          },
                               Stage)
                              .submit(gateway, item, traceFile, appData, inputArgs, Q_GPU, pipeline);
                      }};
    }
}

template <typename NodeType, std::size_t N>
template <int Stage>
auto FlowGraphPipeline<NodeType, N>::create_GPU_Node_with_FN(tbb::flow::graph &g, Tracer &traceFile, ApplicationData &appData, InputArgs &inputArgs, sycl::queue &Q_GPU, PipelineInterface &pipeline) {
    return CPUNode_t{g, tbb::flow::unlimited, [&](ViVidItem *item) -> ViVidItem * {
                         SyclEventInfo eventInfo;
                         if constexpr (USE_VIVID_APP) {
                             if constexpr (Stage == 0) {
                                 eventInfo = Details::cosinefilter<Acc::GPU>(item, traceFile, appData, inputArgs, Q_GPU);
                             } else if constexpr (Stage == 1) {
                                 eventInfo = Details::blockhistogram<Acc::GPU>(item, traceFile, appData, inputArgs, Q_GPU);
                             } else if constexpr (Stage == 2) {
                                 eventInfo = Details::pwdist<Acc::GPU>(item, traceFile, appData, inputArgs, Q_GPU);
                             }
                         } else {
                             eventInfo = workloadsimulator(Acc::GPU, item, traceFile, appData, inputArgs, Stage);
                         }
                         reduceCountersAfterProcessing(inputArgs, appData, Acc::GPU, Stage);
                         return item;
                     }};
}

template <typename NodeType, std::size_t N>
void FlowGraphPipeline<NodeType, N>::initTokenBuffer(tbb::flow::buffer_node<int> &token_buffer, InputArgs &inputArgs) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Filling the token_buffer with " << inputArgs.inFlightFrames << " tokens" << std::endl;
    }
    for (int i = 0; i < inputArgs.inFlightFrames; ++i) {
        token_buffer.try_put(0);
    }
}

// Explicit template instantiation
template class FlowGraphPipeline<FunctionalNode, NUM_STAGES>;
template class FlowGraphPipeline<AsyncNode, NUM_STAGES>;