#include "PipelineInterface.hpp"
#include "Comparer.hpp"
#include "Queue.hpp"
#include "ResourcesManager.hpp"
#include <functional>
#include <iostream>
#include <memory>

Acc PipelineInterface::selectPath(InputArgs &inputArgs, int index, bool &isGPUFrame, ViVidItem *item, Tracer *tracer) {
    Acc acc;
    if constexpr (TRACE_ENABLED) {
        if (item != nullptr && tracer != nullptr)
            tracer->wait_start(item);
    }
    while (true) {
        if (inputArgs.selectedPath == PathSelection::Decoupled) {
            acc = selectPathDecoupled(inputArgs, index, isGPUFrame);
            if (acc != Acc::OTHER) {
                break;
            }
        } else {
            acc = selectPathCoupled(inputArgs, index, isGPUFrame);
            if (acc != Acc::OTHER) {
                break;
            }
        }
    }
    if constexpr (TRACE_ENABLED) {
        if (item != nullptr && tracer != nullptr)
            tracer->wait_end(item);
    }
    return acc;
}

Acc PipelineInterface::selectPathCoupled(InputArgs &inputArgs, int index, bool &isGPUFrame) {
    auto [status, acc] = inputArgs.resourcesManager->acquireForStage(index, inputArgs.stageExecutionState[index], inputArgs.executionDevicePriority[index]);
    return acc;
}

Acc PipelineInterface::selectPathDecoupled(InputArgs &inputArgs, int index, bool &isGPUFrame) {
    if (index == -1) {
        auto [status, acc] = inputArgs.resourcesManager->acquireForStage(0, inputArgs.stageExecutionState[0], inputArgs.executionDevicePriority[0]);
        if (acc == Acc::GPU) {
            isGPUFrame = true;
        }
        return acc;
    }
    return (isGPUFrame ? Acc::GPU : Acc::CPU);
}

void PipelineInterface::reduceCountersAfterProcessing(const InputArgs &inputArgs, const ApplicationData &appData, Acc accelerator, int index, sycl::queue *Q, sycl::event *event, std::vector<sycl::event> *vectorEvents) {
    auto executeNotifyCoreAvailable = [&]() {
        if (inputArgs.selectedPath == PathSelection::Decoupled) {
            if (index == -1) {
                inputArgs.resourcesManager->releaseForStage(0, accelerator);
            }
        } else {
            inputArgs.resourcesManager->releaseForStage(index, accelerator);
        }
    };

    // If SYCL queue and event are provided, submit a task.
    if (Q != nullptr && event != nullptr) {
        auto m_event = Q->submit([&](sycl::handler &cgh) {
            cgh.depends_on(*event);
            executeNotifyCoreAvailable();
        });
    } else {
        // Directly execute the logic if no SYCL queue and event are provided.
        executeNotifyCoreAvailable();
    }
}

ViVidItem *PipelineInterface::processInputNode(ApplicationData &appData, InputArgs &inputArgs, circular_buffer &bufferItems, Tracer &traceFile) {
    if constexpr (LOG_ENABLED) {
        std::clog << "Processing input node in stage with thread " << std::this_thread::get_id() << ".\n";
    }

    // Incrementar el ID de appData y obtener el siguiente item
    appData.id++;
    ViVidItem *item = bufferItems.get();
    item->item_id = appData.id;

    // Rastrear el inicio del frame si TRACE_ENABLED está habilitado
    if constexpr (TRACE_ENABLED) {
        traceFile.frame_start(item);
    }

    // Ejecutar selectPathDecoupled si es necesario
    if (inputArgs.selectedPath == PathSelection::Decoupled) {
        selectPathDecoupled(inputArgs, -1, item->GPU_item);
    }

    return item;
}

void PipelineInterface::logProcessing(ViVidItem *item) {
    if constexpr (LOG_ENABLED) {
        std::clog << "Processing item " << item->item_id << " in stage with " << (item->GPU_item ? "GPU" : "CPU") << " accelerator and thread " << std::this_thread::get_id() << ".\n";
    }
}

void PipelineInterface::adjustCountersAfterProcessing(ViVidItem *item, InputArgs &inputArgs, ApplicationData &appData, Acc accelerator, sycl::event *event, sycl::queue *Q_CPU, sycl::queue *Q_GPU) {
    // if (event == nullptr) {
    //     std::cerr << "Error: Null event pointer passed to adjustCountersAfterProcessing." << std::endl;
    //     return;
    // }
    // if (Q_CPU == nullptr || Q_GPU == nullptr) {
    //     std::cerr << "Error: Null queue pointer passed to adjustCountersAfterProcessing." << std::endl;
    //     return;
    // }

    if (inputArgs.selectedPath == PathSelection::Decoupled) {
        if (inputArgs.pipelineName == PipelineType::SYCLEvents) {
            reduceCountersAfterProcessing(inputArgs, appData, accelerator, -1, (accelerator == Acc::GPU ? Q_GPU : Q_CPU), event, &item->stage_events);
        } else {
            reduceCountersAfterProcessing(inputArgs, appData, (item->GPU_item ? Acc::GPU : Acc::CPU), -1);
        }
    }
}

void PipelineInterface::handleTimeMeasurements(ViVidItem *item, ApplicationData &appData, InputArgs &inputArgs, Acc accelerator) {
    bool shouldMeasureTime = false;

    if constexpr (TIMESTAGES_ENABLED || ADVANCEDMETRICS_ENABLED) {
        shouldMeasureTime = true;
    }

    if constexpr (AUTOMODE_ENABLED) {
        if (appData.autoMode) {
            shouldMeasureTime = true;
        }
    }

    if (shouldMeasureTime) {
        if (inputArgs.pipelineName == PipelineType::SYCLEvents) {
            timeMeasurements(appData, item, accelerator);
        } else {
            timeMeasurements(appData, item);
        }
    }
}

bool PipelineInterface::isAutoModeEnabled(ApplicationData &appData, ViVidItem *item, InputArgs &inputArgs) {
    if constexpr (AUTOMODE_ENABLED) {
        // Comprobamos si inputArgs.timeSampling es igual a 0
        if (appData.autoMode && inputArgs.timeSampling.count() == 0) {
            // Asegurarnos que al menos se hayan medido frames en ambos dispositivos para todas las etapas
            for (int i = 0; i < NUM_STAGES; i++) {
                if (appData.time_CPU_S[i] < 1 || appData.time_GPU_S[i] < 1) {
                    return false;
                }
            }
            appData.autoMode = false;
            appData.sample_end = tbb::tick_count::now();
            return true;
        }
    }
    return false;
}

void PipelineInterface::optimizePipeline(ApplicationData &appData, InputArgs &inputArgs) {
    // Lógica de optimización
    // Imprimir mensaje del tiempo total de la etapa de sampling
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Entered after " << inputArgs.sampleFrames << " frames" << std::endl;
    }

    // Fase 0: Calcular el tiempo total de la etapa de sampling
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Phase 0: Get the mean time per stage for each accelerator" << std::endl;
    }

    std::vector<double> meanTimePerStage_CPU(NUM_STAGES, 0.0);
    std::vector<double> meanTimePerStage_GPU(NUM_STAGES, 0.0);
    // Imprimir el número de filtros por etapa que hay en CPU y GPU
    std::cout << "Number of filters in CPU: [" << appData.numFiltersCPU[0] << " " << appData.numFiltersCPU[1] << " " << appData.numFiltersCPU[2] << "]" << std::endl;
    std::cout << "Number of filters in GPU: [" << appData.numFiltersGPU[0] << " " << appData.numFiltersGPU[1] << " " << appData.numFiltersGPU[2] << "]" << std::endl;
    for (int i = 0; i < NUM_STAGES; i++) {
        meanTimePerStage_CPU[i] = appData.time_CPU_S[i] / appData.numFiltersCPU[i];
        meanTimePerStage_GPU[i] = appData.time_GPU_S[i] / appData.numFiltersGPU[i];
    }

    double throughput_serie = 0.49;

    std::vector<double> thC(NUM_STAGES, 0.0);
    std::vector<double> thG(NUM_STAGES, 0.0);

    // Loop for thC and thG
    auto threadsCPU = SYCL_ENABLED ? 1 : inputArgs.nThreads;
    for (int i = 0; i < NUM_STAGES; i++) {
        thC[i] = (1E3 * threadsCPU) / meanTimePerStage_CPU[i];
        thG[i] = 1E3 / meanTimePerStage_GPU[i];
    }

    if constexpr (VERBOSE_ENABLED) {
        std::cout << " INFO ABOUT THE SYSTEM" << std::endl;
        std::cout << "\t· time_CPU: ['" << meanTimePerStage_CPU[0] << "' '" << meanTimePerStage_CPU[1] << "' '" << meanTimePerStage_CPU[2] << "']" << std::endl;
        std::cout << "\t· time_GPU: ['" << meanTimePerStage_GPU[0] << "' '" << meanTimePerStage_GPU[1] << "' '" << meanTimePerStage_GPU[2] << "']" << std::endl;
        std::cout << "\t· thC=[" << thC[0] << " " << thC[1] << " " << thC[2] << "]" << std::endl;
        std::cout << "\t· thG=[" << thG[0] << " " << thG[1] << " " << thG[2] << "]" << std::endl;
    }

    auto results = PipelineOptimizer::findOptimalConfiguration(NUM_STAGES, thC, thG, threadsCPU);

    for (size_t idx = 0; idx < results.size(); ++idx) {
        const auto &result = results[idx];
        auto confOptP = result.confOptP;
        auto lambdaE = result.lambdae;
        auto confOptS = result.confOptS;
        auto confOptS_letter = confOptS == 0 ? 'C' : (confOptS == 1 ? 'G' : 'E');
        auto coresP = result.cP;
        auto coresS = result.cS;
        auto nTokens = result.ntokens;
        auto speedup = lambdaE / throughput_serie;
        auto NGP = result.NGP;
        auto NCP = result.NCP;
        auto NGS = result.NGS;
        auto NCS = result.NCS;

        printf("Top %zu\n", idx + 1);
        printf("confOptP=%s with lambdaOpt=%.6e\n", confOptP.c_str(), lambdaE);
        printf("GPU bottleneck in confOptP=%s and CPU in secondary path\n", confOptP.c_str());
        printf("Primary GPU path: M/M/1/NGP/NGP with NGP=%d and lambdaeGP=%.6e\n", NGP, result.lambdaGP);
        printf("Primary CPU path: M/M/c/NCP/NCP with c=%d NCP=%d and lambdaeCP=%.6e\n", coresP, NCP, result.lambdaCP);
        if (confOptS == 0 && coresS > 0) {
            printf("Secondary CPU path: M/M/c/NCS/NCS with c=%d and NCS=%d and lambdaeCS=%.6e\n", coresS, NCS, result.lambdaCS);
        } else if (confOptS == 1) {
            printf("Secondary GPU path: M/M/1/NGS/NGS with NGS=%d and lambdaeGS=%.6e\n", NGS, result.lambdaGS);
        }
        printf("lambdae=%.6e and ntokens=%d\n", result.lambdae, nTokens);
        printf("\n");
    }

    const auto &best_result = results.front();

    // Configuración de los dispositivos
    auto deviceCPU = inputArgs.resourcesManager->getDevice(Acc::CPU);
    auto deviceGPU = inputArgs.resourcesManager->getDevice(Acc::GPU);

    // Esperar a que todos los cores estén libres
    while (deviceCPU->getUsedCores() != 0 || deviceGPU->getUsedCores() != 0) {
        printf("Waiting for all cores to be free\n");
        printf("CPU cores in use: %d\n", deviceCPU->getUsedCores());
        printf("GPU cores in use: %d\n", deviceGPU->getUsedCores());
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    // Modificamos las etapas actuales
    auto stageCPU = deviceCPU->getStage(0);
    auto stageGPU = deviceGPU->getStage(0);

    // Modificamos la CPU
    if constexpr (LIMITCORES_ENABLED) {
        stageCPU->setTotalCores(best_result.cP);
        stageCPU->setMaxQueueSize(best_result.NCP - best_result.cP);
    } else {
        stageCPU->setTotalCores((SYCL_ENABLED ? 1 : inputArgs.nThreads));
        stageCPU->setMaxQueueSize(best_result.NCP);
    }

    // Modificamos la GPU
    stageGPU->setTotalCores(1);
    if constexpr (LIMITCORES_ENABLED) {
        stageGPU->setMaxQueueSize(best_result.NGP - 1);
    } else {
        stageGPU->setMaxQueueSize(best_result.NGP);
    }

    if (best_result.confOptS == -1) {
        std::unordered_map<int, int> stageMap{{0, 0}, {1, 0}, {2, 0}};
        deviceCPU->updateStageMapping(stageMap);
        deviceGPU->updateStageMapping(stageMap);
    }

    if (best_result.confOptS == 1) {
        // Añadimos una etapa extra a GPU
        if constexpr (LIMITCORES_ENABLED) {
            deviceGPU->addStage(1, 1, best_result.NGS);
        } else {
            deviceGPU->addStage(1, 1, best_result.NGS - 1);
        }

        // Mapeamos las etapas
        for (size_t i = 0; i < best_result.confOptP.size(); ++i) {
            if (best_result.confOptP[i] == '1') {
                deviceGPU->mapStageIndex(i, 0);
            } else {
                deviceGPU->mapStageIndex(i, 1);
            }
            deviceCPU->mapStageIndex(i, 0);
        }
    }

    if (best_result.confOptS == 0) {
        if (best_result.cS > 0) {
            // Añadimos una etapa extra a CPU
            if constexpr (LIMITCORES_ENABLED) {
                deviceCPU->addStage(1, best_result.cS, best_result.NCS - best_result.cS);
            } else {
                deviceCPU->addStage(1, (SYCL_ENABLED ? 1 : inputArgs.nThreads), best_result.NCS);
            }

            // Mapeamos las etapas
            for (size_t i = 0; i < best_result.confOptP.size(); ++i) {
                if (best_result.confOptP[i] == '0') {
                    deviceCPU->mapStageIndex(i, 0);
                } else {
                    deviceCPU->mapStageIndex(i, 1);
                }
                deviceGPU->mapStageIndex(i, 0);
            }
        } else {
            // Mapeamos las etapas
            for (size_t i = 0; i < best_result.confOptP.size(); ++i) {
                deviceCPU->mapStageIndex(i, 0);
                deviceGPU->mapStageIndex(i, 0);
            }
        }
    }

    // Impresión de la configuración de los dispositivos
    std::cout << "Configuración de dispositivos:\n";
    std::cout << "Configuración del CPU:\n";
    std::cout << "Total de Cores: " << deviceCPU->getTotalCores() << "\n";
    std::cout << "Cores en uso: " << deviceCPU->getUsedCores() << "\n";
    for (size_t i = 0; i < NUM_STAGES; ++i) {
        const auto &stage = deviceCPU->getStage(i);
        std::cout << "  Etapa " << i << ":\n";
        std::cout << "    Máximo de Cores: " << stage->getTotalCores() << "\n";
        std::cout << "    Tamaño máximo de la cola: " << stage->getMaxQueueSize() << "\n";
    }
    std::cout << "\n";

    std::cout << "Configuración del GPU:\n";
    std::cout << "Total de Cores: " << deviceGPU->getTotalCores() << "\n";
    std::cout << "Cores en uso: " << deviceGPU->getUsedCores() << "\n";
    for (size_t i = 0; i < NUM_STAGES; ++i) {
        const auto &stage = deviceGPU->getStage(i);
        std::cout << "  Etapa " << i << ":\n";
        std::cout << "    Máximo de Cores: " << stage->getTotalCores() << "\n";
        std::cout << "    Tamaño máximo de la cola: " << stage->getMaxQueueSize() << "\n";
    }
    std::cout << "\n";

    // Si confOptP es '111' se ejecuta en modo decoupled, todas las etapas son CPU_GPU
    if (best_result.confOptP == "111") {
        inputArgs.selectedPath = PathSelection::Decoupled;
        for (int i = 0; i < NUM_STAGES; i++) {
            inputArgs.stageExecutionState[i] = StageState::CPU_GPU;
        }
    } else {
        // Configuraciones Coupled
        inputArgs.selectedPath = PathSelection::Coupled;
        // Ajustar los estados de ejecución de las etapas según la mejor configuración
        if (best_result.confOptS == -1) {
            for (int i = 0; i < NUM_STAGES; i++) {
                inputArgs.stageExecutionState[i] = (best_result.confOptP[i] == '1') ? StageState::GPU : StageState::CPU;
            }
        } else if (best_result.confOptS == 1) {
            for (int i = 0; i < NUM_STAGES; i++) {
                inputArgs.stageExecutionState[i] = (best_result.confOptP[i] == '1') ? StageState::GPU : StageState::CPU_GPU;
            }
        } else if (best_result.confOptS == 0) {
            for (int i = 0; i < NUM_STAGES; i++) {
                inputArgs.stageExecutionState[i] = (best_result.confOptP[i] == '0') ? StageState::CPU : StageState::CPU_GPU;
            }
        }
    }

    // Imprimir stageExecutionState
    for (int i = 0; i < NUM_STAGES; i++) {
        std::cout << "Stage " << i << " is executed in " << (inputArgs.stageExecutionState[i] == StageState::CPU ? "CPU" : (inputArgs.stageExecutionState[i] == StageState::GPU ? "GPU" : "CPU_GPU")) << std::endl;
    }

    // Modificamos executionDevicePriority según la mejor configuración
    for (size_t i = 0; i < best_result.confOptP.size(); ++i) {
        inputArgs.executionDevicePriority[i] = (best_result.confOptP[i] == '1') ? Acc::GPU : Acc::CPU;
    }

    for (size_t i = 0; i < NUM_STAGES; ++i) {
        std::cout << "Stage " << i << " is prioritary in " << (inputArgs.executionDevicePriority[i] == Acc::GPU ? "GPU" : "CPU") << std::endl;
    }

    // Cambiamos el número de tokens
    inputArgs.inFlightFrames = best_result.ntokens;

    // Guardamos el nombre de la configuración
    inputArgs.configStagesStr = best_result.confOptP + "-A";

    // Start the timer if needed
    inputArgs.sampleFrames = appData.id;
    std::cout << "We have sampled " << inputArgs.sampleFrames << " frames" << std::endl;

    // 4.6: Calculate the throughput balance
    appData.sampleTime = (appData.sample_end - appData.pipeline_start).seconds() * 1000; // Time to sample the system
    appData.throughputBalance = (inputArgs.sampleFrames * 1000) / (appData.sampleTime);  // Throughput obtained during system optimization
    appData.throughputSystemExpected = best_result.lambdae;

    // if (!inputArgs.hasDuration()) {
    //     // Aumentamos el número de frames a procesar
    //     std::cout << "Adding " << appData.id << " frames to the total number of frames to process" << std::endl;
    //     inputArgs.numFrames += appData.id;
    // } else {
    //     startTimerIfNeeded(appData, inputArgs);
    // }

    if (inputArgs.hasDuration()) {
        startTimerIfNeeded(appData, inputArgs);
    }

    // 4.7: Initialize the timer to measure the total time to process once optimized
    appData.system_start = tbb::tick_count::now();
}

void PipelineInterface::debugAndTrace(ViVidItem *item, ApplicationData &appData, Tracer &traceFile) {
    if constexpr (DEBUG_ENABLED) {
        Comparer::compare(item, appData);
    }
    if constexpr (TRACE_ENABLED) {
        traceFile.frame_end(item);
    }
}

void PipelineInterface::recycleItem(circular_buffer &bufferItems, ViVidItem *item) {
    bufferItems.recycle(item);
}

void PipelineInterface::publicReduceCountersAfterProcessing(const InputArgs &inputArgs, const ApplicationData &appData, Acc accelerator, int index, sycl::queue *Q, sycl::event *event, std::vector<sycl::event> *vectorEvents) {
    reduceCountersAfterProcessing(inputArgs, appData, accelerator, index, Q, event, vectorEvents);
}