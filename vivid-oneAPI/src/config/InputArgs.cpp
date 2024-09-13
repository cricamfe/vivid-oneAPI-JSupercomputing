#include "InputArgs.hpp"
#include "Device.hpp"
#include "GlobalParameters.hpp"
#include "PipelineFactory.hpp"
#include "Stage.hpp"
#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

InputArgs::InputArgs(int argc, char *argv[]) {
    parseArguments(argc, argv);
}

bool InputArgs::parseConfigStages() {
    try {
        if constexpr (TIMESTAGES_ENABLED) {
            GPUactive = true;
            configStagesStr = "TIMESTAGES";
            std::fill(stageExecutionState.begin(), stageExecutionState.end(), StageState::CPU_GPU);
            return true;
        } else if constexpr (AUTOMODE_ENABLED) {
            GPUactive = true;
            configStagesStr = "AUTO";
            std::fill(stageExecutionState.begin(), stageExecutionState.end(), StageState::CPU_GPU);
            return true;
        } else if (configStagesStr == "CPU" || std::all_of(configStagesStr.begin(), configStagesStr.end(), [](char c) { return c == '0'; })) {
            configStagesStr = "CPU-only";
            std::fill(stageExecutionState.begin(), stageExecutionState.end(), StageState::CPU);
            return true;
        } else if (configStagesStr == "GPU" || std::all_of(configStagesStr.begin(), configStagesStr.end(), [](char c) { return c == '2'; })) {
            GPUactive = true;
            configStagesStr = "GPU-only";
            std::fill(stageExecutionState.begin(), stageExecutionState.end(), StageState::GPU);
            return true;
        } else if (configStagesStr == "DECOUPLED") {
            GPUactive = true;
            configStagesStr = "Decoupled";
            std::fill(stageExecutionState.begin(), stageExecutionState.end(), StageState::CPU_GPU);
            selectedPath = PathSelection::Decoupled;
            return true;
        } else if (configStagesStr.length() == NUM_STAGES) {
            GPUactive = true;
            for (int i = 0; i < NUM_STAGES; i++) {
                if (configStagesStr[i] == '2') {
                    stageExecutionState[i] = StageState::GPU;
                } else if (configStagesStr[i] == '1') {
                    stageExecutionState[i] = StageState::CPU_GPU;
                } else if (configStagesStr[i] == '0') {
                    stageExecutionState[i] = StageState::CPU;
                } else {
                    std::cout << "Invalid configuration of the stages" << std::endl;
                    return false;
                }
            }
            return true;
        } else {
            return false;
        }
    } catch (const std::invalid_argument &e) {
        return false;
    } catch (const std::out_of_range &e) {
        return false;
    }
    return false;
}

void InputArgs::setResources(const std::vector<int> &size, const std::vector<int> &cores, Acc acc) {
    auto num_cores = (acc == Acc::CPU ? nThreads : DEFAULT_CORES_GPU);
    // Si estamos en Acc::GPU y todos los valores de stageExecutionState son GPU, entonces los cores de CPU son 0
    if (acc == Acc::CPU && std::all_of(stageExecutionState.begin(), stageExecutionState.end(), [](StageState s) { return s == StageState::GPU; })) {
        num_cores = 0;
    }
    // Si estamos en Acc::CPU y todos los valores de stageExecutionState son CPU, entonces los cores de GPU son 0
    if (acc == Acc::GPU && std::all_of(stageExecutionState.begin(), stageExecutionState.end(), [](StageState s) { return s == StageState::CPU; })) {
        num_cores = 0;
    }
    auto device = std::make_unique<Device>(acc, num_cores);

    auto getVectorValue = [](const std::vector<int> &vec, int index, int default_value) {
        if (vec.empty()) {
            return default_value;
        } else if (vec.size() == 1) {
            return vec[0];
        } else if (vec.size() == NUM_STAGES) {
            return vec[index];
        } else {
            throw std::invalid_argument("Invalid vector size.");
        }
    };

    if (selectedPath == PathSelection::Decoupled) {
        // Check the size and cores vectors
        if (size.size() != 1 && !size.empty()) {
            throw std::invalid_argument("Invalid size vector configuration for the decoupled path.");
        }
        if (cores.size() != 1 && !cores.empty()) {
            throw std::invalid_argument("Invalid cores vector configuration for the decoupled path.");
        }

        // Default values
        int default_cores = (acc == Acc::CPU) ? ((SYCL_ENABLED) ? 1 : nThreads) : DEFAULT_CORES_GPU;
        int default_size = 0;

        // Add the stages to the device
        int _cores = getVectorValue(cores, 0, default_cores);
        int _size = getVectorValue(size, 0, default_size);
        if constexpr (LOG_ENABLED) {
            std::cout << "Adding device " << device->getAccStr() << " with cores: " << _cores << ", size: " << _size << std::endl;
        }
        device->addStage(0, _cores, _size);
        device->mapStageIndex(0, 0);
    } else {
        // Check the size and cores vectors
        if (size.size() > 1 && size.size() != NUM_STAGES) {
            throw std::invalid_argument("Invalid size vector configuration for the coupled path.");
        }
        if (cores.size() > 1 && cores.size() != NUM_STAGES) {
            throw std::invalid_argument("Invalid cores vector configuration for the coupled path.");
        }

        for (int i = 0; i < NUM_STAGES; i++) {
            int _cores = 0;
            int _size = 0;

            if (stageExecutionState[i] == StageState::CPU_GPU) {
                if constexpr (__ACQMODE__ != 2) {
                    _size = getVectorValue(size, i, DEFAULT_CORES_GPU);
                }
                _cores = getVectorValue(cores, i, (acc == Acc::CPU) ? ((SYCL_ENABLED) ? 1 : nThreads) : DEFAULT_CORES_GPU);
            } else if (stageExecutionState[i] == StageState::GPU) {
                if (acc == Acc::GPU) {
                    _cores = getVectorValue(cores, i, DEFAULT_CORES_GPU);
                    if constexpr (__ACQMODE__ != 2) {
                        _size = getVectorValue(size, i, inFlightFrames);
                    }
                }
            } else if (stageExecutionState[i] == StageState::CPU) {
                if (acc == Acc::CPU) {
                    _cores = getVectorValue(cores, i, (SYCL_ENABLED) ? 1 : nThreads);
                    if constexpr (__ACQMODE__ != 2) {
                        _size = getVectorValue(size, i, inFlightFrames);
                    }
                }
            }
            if constexpr (LOG_ENABLED) {
                std::cout << "Adding stage " << i << " to device " << device->getAccStr() << " with _cores: " << _cores << ", _size: " << _size << std::endl;
            }

            // Add the stage to the device
            device->addStage(i, _cores, _size);
            // Map the stage index
            device->mapStageIndex(i, i);
        }
    }

    // Verificación antes de agregar el dispositivo
    if constexpr (LOG_ENABLED) {
        std::clog << "Adding device " << device->getAccStr() << " to the resources manager" << std::endl;
    }
    resourcesManager->addDevice(acc, std::move(device));

    return;
}

// Function to set execution device priority based on the given conditions
void InputArgs::setExecutionDevicePriority(const std::vector<int> &exeDevPriority, std::vector<Acc> &executionDevicePriority) {
    if (exeDevPriority.empty()) {
        // If exeDevPriority is empty, determine the priority based on stageExecutionState
        for (int i = 0; i < NUM_STAGES; i++) {
            Acc acc = ((stageExecutionState[i] == StageState::GPU) || (stageExecutionState[i] == StageState::CPU_GPU)) ? Acc::GPU : Acc::CPU;
            executionDevicePriority[i] = acc;
        }
    } else {
        // Determine the default accelerator based on the first value of exeDevPriority
        Acc defaultAcc = (exeDevPriority[0] == 1) ? Acc::GPU : Acc::CPU;
        if (exeDevPriority.size() == 1) {
            // If only one priority is given, apply it to all stages
            std::fill(executionDevicePriority.begin(), executionDevicePriority.end(), defaultAcc);
        } else if (exeDevPriority.size() == NUM_STAGES) {
            // If priorities are given for all stages, transform them accordingly
            std::transform(exeDevPriority.begin(), exeDevPriority.end(), executionDevicePriority.begin(),
                           [](int v) { return (v == 2) ? Acc::GPU : Acc::CPU; });
        } else {
            throw std::invalid_argument("Invalid number of execution device priorities provided.");
        }
    }
}

void InputArgs::setThroughput(const std::vector<double> &th, std::vector<double> &throughput) {
    if (th.size() == 1) {
        // If only one throughput value is given, apply it to all stages
        std::fill(throughput.begin(), throughput.end(), th[0]);
    } else if (th.size() == NUM_STAGES) {
        // If throughput values are given for all stages, copy them to throughput
        std::copy(th.begin(), th.end(), throughput.begin());
    }
}

void InputArgs::parseArguments(int argc, char *argv[]) {
    auto printSeparator = []() {
        std::cout << "---------------------------------------------------------------------------------------\n";
    };

    CLI::App app{"vivid-OneAPI: An example of a pipeline using oneAPI"};

    // Variables temporales
    std::string pipelineStr;
    std::string durationStr;
    std::string timeSamplingStr;
    std::vector<int> sizeGPU;
    std::vector<int> sizeCPU;
    std::vector<int> coresCPU;
    std::vector<int> coresGPU;
    std::vector<int> exeDevPriority;
    std::vector<double> th_CPU;
    std::vector<double> th_GPU;

    app.add_option("--api", pipelineStr, "Name of the API")->required()->check(CLI::IsMember({"pipeline", "fgfn", "fgan", "syclevents", "taskflow", "serie"}))->default_val("pipeline");
    app.add_option("--numframes", numFrames, "Number of frames to process")->check(CLI::PositiveNumber);
    app.add_option("--resolution", imageResolution, "Image resolution (0: 1280x720, 1: 1080p, 2: 1440p, 3: 2160p, 4: 2880p, 5: 4320p)")->check(CLI::Range(0, 5));
    app.add_option("--duration", durationStr, "Duration of the execution");
    app.add_option("--threads", nThreads, "Number of cores to use in the CPU")->check(CLI::PositiveNumber);
    app.add_option("--iff", inFlightFrames, "Number of frames in flight")->check(CLI::PositiveNumber);
    app.add_option("--config", configStagesStr, "Configuration of the stages as a string (0: CPU, 1: CPU+GPU, 2: GPU)");
    app.add_option("--buffersize", sizeCircularBuffer, "Size of the circular buffer")->check(CLI::PositiveNumber);
    app.add_option("--sizegpu", sizeGPU, "Size of the general GPU queue")->expected(1, NUM_STAGES);
    app.add_option("--sizecpu", sizeCPU, "Size of the general CPU queue")->expected(1, NUM_STAGES);
    app.add_option("--corescpu", coresCPU, "Number of cores per stage in the CPU")->expected(1, NUM_STAGES);
    app.add_option("--coresgpu", coresGPU, "Number of cores per stage in the GPU")->expected(1, NUM_STAGES);
    app.add_option("--prefdevice", exeDevPriority, "Preferred device per stage (0: CPU, 2: GPU)")->expected(1, NUM_STAGES);
    app.add_flag("--dependson", useDependsOnSerial, "Flag that uses sycl::events on --api being 'serie'");
    app.add_option("--thcpu", th_CPU, "Throughput of the CPU in stage 1")->expected(1, NUM_STAGES);
    app.add_option("--thgpu", th_GPU, "Throughput of the GPU in stage 1")->expected(1, NUM_STAGES);

    if constexpr (AUTOMODE_ENABLED) {
        app.add_option("--timesampling", timeSamplingStr, "Time sampling for model")->default_val("10s");
    }

    // Procesamos los argumentos
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
    }

    // Obtenemos el tipo de pipeline
    pipelineName = PipelineFactory::getPipelineType(pipelineStr);

    // Validamos que el flag --dependson solo sea válido cuando el API es 'serie'
    if (useDependsOnSerial && pipelineStr != "serie") {
        throw std::invalid_argument("--usedependsonserial is only valid when --api is 'serie'");
    }
    // Validamos que no se especifiquen ambos flags --numframes y --duration
    if (numFrames != DEFAULT_NUM_FRAMES && !durationStr.empty()) {
        throw std::invalid_argument("Specify either --numframes or --duration, not both.");
    }

    // Si el usuario define el flag --duration, se convierte a segundos
    if (!durationStr.empty()) {
        std::cout << "Changing duration to seconds" << std::endl;
        std::regex durPattern(R"((\d+h)?(\d+m)?(\d+s)?)");
        std::smatch match;
        if (std::regex_match(durationStr, match, durPattern)) {
            int hours = match[1].length() ? std::stoi(match[1].str().substr(0, match[1].length() - 1)) : 0;
            int minutes = match[2].length() ? std::stoi(match[2].str().substr(0, match[2].length() - 1)) : 0;
            int seconds = match[3].length() ? std::stoi(match[3].str().substr(0, match[3].length() - 1)) : 0;
            duration = std::chrono::hours(hours) + std::chrono::minutes(minutes) + std::chrono::seconds(seconds);
        } else {
            throw std::invalid_argument("Invalid duration format. Should be like '1h2m3s'.");
        }
    }

    if constexpr (AUTOMODE_ENABLED) {
        if (!timeSamplingStr.empty()) {
            std::regex timePattern(R"((\d+h)?(\d+m)?(\d+s)?)");
            std::smatch match;
            if (std::regex_match(timeSamplingStr, match, timePattern)) {
                int hours = match[1].length() ? std::stoi(match[1].str().substr(0, match[1].length() - 1)) : 0;
                int minutes = match[2].length() ? std::stoi(match[2].str().substr(0, match[2].length() - 1)) : 0;
                int seconds = match[3].length() ? std::stoi(match[3].str().substr(0, match[3].length() - 1)) : 0;
                timeSampling = std::chrono::hours(hours) + std::chrono::minutes(minutes) + std::chrono::seconds(seconds);
                std::cout << "Time Sampling: " << timeSampling.count() << " seconds" << std::endl;
            } else {
                throw std::invalid_argument("Invalid time sampling format. Should be like '100'.");
            }
        }
    }

    printSeparator();
    std::cout << " \033[1m" << "API: " << "\033[91m" << PipelineFactory::getPipelineTypeString(pipelineName) << "\033[0m" << std::endl;
    printSeparator();
    std::cout << " INPUT ARGUMENTS" << std::endl;
    printSeparator();

    // Si estamos en la versión Serie
    if (pipelineName == PipelineType::Serie) {
        // Validar que solo se permita "GPU" o "CPU"
        if (configStagesStr != "CPU" && configStagesStr != "GPU") {
            throw std::invalid_argument("Only 'CPU' and 'GPU' configurations are allowed for the 'serie' API.");
        }
        // Establecer el número de threads a 1-core
        nThreads = 1;
        // Si el usuario NO define manualmente el tamaño del buffer circular
        if (sizeCircularBuffer == 0) {
            sizeCircularBuffer = DEFAULT_SIZE_CIRCULAR_BUFFER;
            if constexpr (VERBOSE_ENABLED) {
                printf(" Number of items on circular buffer: %lu\n", sizeCircularBuffer);
            }
        }
    }

    // Si el temporizador está activo, se imprime la duración en segundos, si no, se imprime el número de frames
    if (hasDuration()) {
        std::cout << " Duration: " << duration.count() << " seconds" << std::endl;
    } else {
        std::cout << " Number of Frames: " << numFrames << std::endl;
    }

    if constexpr (AUTOMODE_ENABLED) {
        std::cout << " Time Sampling: " << timeSampling.count() << " seconds" << std::endl;
    }

    if (pipelineName != PipelineType::Serie) {
        if (!parseConfigStages()) {
            throw std::invalid_argument("Error parsing the configuration of the stages.");
        }
        // Print number of threads AND configuration of the stages
        std::cout << " Number of Threads: " << nThreads << std::endl;
        std::cout << " Config Stages: " << configStagesStr << std::endl;

        // Si el usuario NO define manualmente el número de frames en vuelo
        int minFramesInFlightSYCL = 2;
        if (inFlightFrames == 0) {
            inFlightFrames = (SYCL_ENABLED) ? std::min(nThreads + GPUactive, minFramesInFlightSYCL) : nThreads + GPUactive;
            std::clog << " Number of frames in flight: " << inFlightFrames << std::endl;
        }
        // Si el usuario NO define manualmente el tamaño del buffer circular
        if (sizeCircularBuffer == 0) {
            sizeCircularBuffer = inFlightFrames * DEFAULT_SIZE_CIRCULAR_BUFFER;
            if constexpr (VERBOSE_ENABLED) {
                printf(" Number of items on circular buffer: %lu\n", sizeCircularBuffer);
            }
        }
        // Print the number of frames in flight
        std::cout << " In-Flight Frames: " << inFlightFrames << std::endl;

        if constexpr (TIMESTAGES_ENABLED || AUTOMODE_ENABLED) {
            coresGPU = {DEFAULT_CORES_GPU};
            coresCPU = {(SYCL_ENABLED) ? 1 : nThreads};
            sizeGPU = {0};
            sizeCPU = {0};
            exeDevPriority = {1};
        } else if (configStagesStr == "GPU-only") {
            coresGPU = {DEFAULT_CORES_GPU};
            coresCPU = {0};
            sizeGPU = {nThreads};
            sizeCPU = {0};
            exeDevPriority = {1};
        } else if (configStagesStr == "CPU-only") {
            coresGPU = {0};
            coresCPU = {(SYCL_ENABLED) ? 1 : nThreads};
            sizeGPU = {0};
            sizeCPU = {(SYCL_ENABLED) ? 2 : nThreads};
            exeDevPriority = {0};
        }

        if constexpr (__ACQMODE__ == 0) {
            std::cout << " Acquisition Mode: Default" << std::endl;
        } else if constexpr (__ACQMODE__ == 1) {
            std::cout << " Acquisition Mode: Acquire cores and queues (primary & secondary)" << std::endl;
        } else if constexpr (__ACQMODE__ == 2) {
            std::cout << " Acquisition Mode: No Queue" << std::endl;
        }

        // if constexpr (AUTOMODE_ENABLED) {
        //     if (this->hasDuration()) {
        //         sampleFrames = MAX_NFRAMES_QUEUE;
        //     } else {
        //         sampleFrames = numFrames * ((AVX_ENABLED || SIMD_ENABLED) ? PER_FRAMES_TO_PROCESS_VEC : PER_FRAMES_TO_PROCESS_BAS);
        //     }
        //     if constexpr (VERBOSE_ENABLED) {
        //         std::cout << " Automatic Mode: Enabled" << std::endl;
        //         std::cout << " - Number of frames used on sampling: " << sampleFrames << std::endl;
        //     }
        // }

        // Initialize the resources manager
        resourcesManager = std::make_unique<ResourcesManager>();
        setResources(sizeGPU, coresGPU, Acc::GPU);
        setResources(sizeCPU, coresCPU, Acc::CPU);

        // Define the preferred device for each stage
        setExecutionDevicePriority(exeDevPriority, executionDevicePriority);

        // Set the throughput of the CPU and GPU in each stage
        setThroughput(th_CPU, throughput_CPU);
        setThroughput(th_GPU, throughput_GPU);

        auto printAccInfo = [&](Acc acc, const std::vector<double> &throughput) {
            Device *device = resourcesManager->getDevice(acc);
            std::cout << " " << device->getAccStr() << ":" << std::endl;

            // Configurar los anchos de columna para la alineación
            const int labelWidth = 15;
            const int valueWidth = 10;

            if (selectedPath == PathSelection::Decoupled) {
                std::cout << std::left << std::setw(labelWidth) << "  - Cores:" << std::setw(valueWidth) << device->getStage(0)->getTotalCores() << std::endl;
                std::cout << std::left << std::setw(labelWidth) << "  - Q.Size:" << std::setw(valueWidth) << device->getStage(0)->getMaxQueueSize() << std::endl;
                if (throughput[0] != -1) {
                    std::cout << std::left << std::setw(labelWidth) << "  - Throug.:" << std::setw(valueWidth) << throughput[0] << std::endl;
                }
            } else {
                for (int i = 0; i < NUM_STAGES; i++) {
                    std::cout << "  - Stage " << (i + 1) << ":" << std::endl;
                    std::cout << std::left << std::setw(labelWidth) << "    - Cores:" << std::setw(valueWidth) << device->getStage(i)->getTotalCores() << std::endl;
                    std::cout << std::left << std::setw(labelWidth) << "    - Q.Size:" << std::setw(valueWidth) << device->getStage(i)->getMaxQueueSize() << std::endl;
                    if (throughput[i] != -1) {
                        std::cout << std::left << std::setw(labelWidth) << "    - Throug.:" << std::setw(valueWidth) << throughput[i] << std::endl;
                    }
                }
            }
        };

        std::cout << " Device Information:" << std::endl;
        printAccInfo(Acc::GPU, throughput_GPU);
        printAccInfo(Acc::CPU, throughput_CPU);

        std::cout << " Pref. Device:" << std::endl;
        if (selectedPath == PathSelection::Decoupled) {
            std::cout << "  - " << (executionDevicePriority[0] == Acc::GPU ? "GPU" : "CPU") << std::endl;
        } else {
            for (int i = 0; i < NUM_STAGES; i++) {
                std::cout << "  - Stage " << (i + 1) << ": " << (executionDevicePriority[i] == Acc::GPU ? "GPU" : "CPU") << std::endl;
            }
        }
    }
    if constexpr (DEBUG_ENABLED) {
        this->printArguments();
    }
}

void InputArgs::printArguments() const {
    // TODO : Implement this function
}

const std::string InputArgs::getImageTypeToString() const {
    switch (imageResolution) {
    case 1:
        return "1080p";
    case 2:
        return "1440p";
    case 3:
        return "2160p";
    case 4:
        return "2880p";
    case 5:
        return "4320p";
    default:
        return "640p";
    }
}