#include "jsonfile.hpp"

void JSONFile::saveToFile(const std::string &filename) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Attempting to save JSON to file: " << filename << std::endl;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    // Serialización incremental
    file << "{\n"; // Comienza el JSON

    bool first = true;
    for (const auto &[key, value] : m_json.items()) {
        if (!first) {
            file << ",\n";
        }
        file << "\"" << key << "\": " << value.dump(4); // Serializa cada parte del JSON
        first = false;
    }

    file << "\n}\n"; // Cierra el JSON
    file.close();

    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Successfully saved JSON to file: " << filename << std::endl;
    }
}

void JSONFile::loadFromFile(const std::string &filename) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Attempting to load JSON from file: " << filename << std::endl;
    }
    std::ifstream file(filename);
    if (file.is_open()) {
        file >> m_json;
        file.close();
        if constexpr (VERBOSE_ENABLED) {
            std::cout << "Successfully loaded JSON from file: " << filename << std::endl;
        }
    } else {
        m_json = nlohmann::json::object();
        if constexpr (VERBOSE_ENABLED) {
            std::cout << "File not found or unable to open. Initialized empty JSON object." << std::endl;
        }
    }
}

void JSONFile::setValue(const std::string &key, const nlohmann::json &value) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Setting value for key: " << key << std::endl;
    }
    m_json[key] = value;
}

nlohmann::json JSONFile::getValue(const std::string &key) const {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Getting value for key: " << key << std::endl;
    }
    return m_json.at(key);
}

std::string JSONFile::generateRandomKey() {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Generating random key." << std::endl;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(100000, 999999);
    return std::to_string(dis(gen));
}

std::string JSONFile::generateCommonKey(const ApplicationData &appData, const InputArgs &inputArgs) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Generating common key." << std::endl;
    }
    std::string commonKey;

    // Generate common key for the JSON file for Serie Pipeline
    if (inputArgs.pipelineName == PipelineType::Serie) {
        commonKey = PipelineFactory::getPipelineTypeAsShortString(inputArgs.pipelineName) + "_" +
                    (SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "SIMD" : "C++"))) + "_" +
                    inputArgs.configStagesStr + "_" +
                    std::to_string(inputArgs.nThreads) + "_" +
                    inputArgs.getImageTypeToString();
        return commonKey;
    }

    // Common key for the JSON file for Parallel Pipeline, FlowGraph and SYCL Events
    commonKey = PipelineFactory::getPipelineTypeAsShortString(inputArgs.pipelineName) + "_" +
                (SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "SIMD" : "C++"))) + "_" +
                ((__BACKEND__ == 0) ? "OpenCL" : ((__BACKEND__ == 1) ? "LevelZero" : "CUDA")) + "_" +
                inputArgs.configStagesStr + "_" +
                std::to_string(inputArgs.inFlightFrames) + "_" +
                std::to_string(inputArgs.nThreads) + "_" +
                inputArgs.getPrefDevice() + "_" +
                (inputArgs.preferGpu ? "1" : "0") + "_" +
                ((INORDER_QUEUE) ? "inOrder" : "outOfOrder") + "_" +
                inputArgs.getImageTypeToString() + "_";

    Device *deviceGPU = inputArgs.resourcesManager->getDevice(Acc::GPU);
    Device *deviceCPU = inputArgs.resourcesManager->getDevice(Acc::CPU);

    if (inputArgs.selectedPath == PathSelection::Decoupled) {
        if (deviceGPU != nullptr) {
            commonKey += std::to_string(deviceGPU->getStage(0)->getTotalCores()) + "_";
        }

        if (deviceCPU != nullptr) {
            commonKey += std::to_string(deviceCPU->getStage(0)->getTotalCores()) + "_";
        }
        return commonKey;
    }

    if (deviceGPU != nullptr) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            commonKey += std::to_string(deviceGPU->getStage(i)->getTotalCores()) + "_";
        }
        for (int i = 0; i < NUM_STAGES; ++i) {
            commonKey += std::to_string(deviceGPU->getStage(i)->getMaxQueueSize()) + "_";
        }
    }

    if (deviceCPU != nullptr) {
        for (int i = 0; i < NUM_STAGES; ++i) {
            commonKey += std::to_string(deviceCPU->getStage(i)->getTotalCores()) + "_";
        }
        for (int i = 0; i < NUM_STAGES; ++i) {
            commonKey += std::to_string(deviceCPU->getStage(i)->getMaxQueueSize()) + "_";
        }
    }
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Generated common key: " << commonKey << std::endl;
    }
    return commonKey;
}

std::string JSONFile::prepareDirectory(const InputArgs &inputArgs, char *executable_path) {
#if __cplusplus >= 202002L
    namespace fs = std::filesystem;
#else
    namespace fs = std::experimental::filesystem;
#endif

    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Preparing directory based on input arguments." << std::endl;
    }

    fs::path exe_path = fs::canonical(executable_path);

    // Get hostname
    char hostname[HOST_NAME_MAX + 1];

    if (gethostname(hostname, sizeof(hostname)) != 0) {
        throw std::system_error(errno, std::system_category(), "Error while getting hostname");
    }

    // Create the base path
    fs::path base_path = exe_path.parent_path() / "json" / std::string(hostname);
    base_path /= inputArgs.getImageTypeToString();
    base_path /= PipelineFactory::getPipelineTypeAsShortString(inputArgs.pipelineName);

    std::string subdir;
    if constexpr (TIMESTAGES_ENABLED) {
        subdir = "time_stages";
    } else if constexpr (ADVANCEDMETRICS_ENABLED) {
        subdir = "advanced_metrics";
    } else if constexpr (AUTOMODE_ENABLED) {
        subdir = "auto";
    } else if constexpr (ENERGYPCM_ENABLED) {
        subdir = "energy_pcm";
        // subdir += std::string("/") + (SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "SIMD" : "C++")));
    } else {
        subdir = "throughput";
    }

    base_path /= subdir;

    // Create the directories
    fs::create_directories(base_path);

    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Prepared directory: " << base_path.string() << std::endl;
    }

    return base_path.string();
}

std::pair<nlohmann::json, nlohmann::json> JSONFile::buildDataMap(const ApplicationData &appData, const InputArgs &inputArgs) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Building data map." << std::endl;
    }
    nlohmann::json commonData;
    nlohmann::json variableData;

    if (inputArgs.pipelineName == PipelineType::Serie) {
        commonData["API"] = PipelineFactory::getPipelineTypeString(inputArgs.pipelineName);
        commonData["Backend CPU"] = SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "SIMD" : "C++"));
        commonData["Resolution"] = inputArgs.getImageTypeToString();
        commonData["Num. Threads"] = inputArgs.nThreads;
        commonData["Config. Stages"] = inputArgs.configStagesStr;
        variableData["Num. Frames"] = inputArgs.numFrames;
        variableData["Throughput (FPS)"] = appData.throughput;
        variableData["Tot. Time (ms)"] = appData.totalTime;

        return {commonData, variableData};
    }

    // InputArgs variables
    commonData["API"] = PipelineFactory::getPipelineTypeString(inputArgs.pipelineName);
    commonData["Backend CPU"] = SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "SIMD" : "C++"));
    commonData["Backend GPU"] = (__BACKEND__ == 0) ? "OpenCL" : ((__BACKEND__ == 1) ? "Level Zero" : "CUDA");
    commonData["Config. Stages"] = inputArgs.configStagesStr;
    commonData["In-flight Frames"] = inputArgs.inFlightFrames;
    commonData["Num. Threads"] = inputArgs.nThreads;
    commonData["Pref. Device"] = inputArgs.getPrefDevice();
    commonData["Queue Order"] = (INORDER_QUEUE) ? "sycl::queue::in_order" : "sycl::queue::out_of_order";
    commonData["Resolution"] = inputArgs.getImageTypeToString();

    Device *deviceGPU = inputArgs.resourcesManager->getDevice(Acc::GPU);
    Device *deviceCPU = inputArgs.resourcesManager->getDevice(Acc::CPU);

    if (inputArgs.selectedPath == PathSelection::Decoupled) {
        if (deviceCPU != nullptr) {
            commonData["Size QCPU"] = std::to_string(deviceCPU->getStage(0)->getMaxQueueSize());
        }
        if (deviceGPU != nullptr) {
            commonData["Size QGPU"] = std::to_string(deviceGPU->getStage(0)->getMaxQueueSize());
        }
        commonData["Pref. GPU"] = std::to_string((inputArgs.executionDevicePriority[0] == Acc::GPU) ? 1 : 0);
    } else {
        for (int i = 0; i < NUM_STAGES; ++i) {
            if (deviceGPU != nullptr) {
                commonData["Size QGPU S" + std::to_string(i + 1)] = std::to_string(deviceGPU->getStage(i)->getMaxQueueSize());
                commonData["Num. Cores GPU S" + std::to_string(i + 1)] = std::to_string(deviceGPU->getStage(i)->getTotalCores());
            }
            if (deviceCPU != nullptr) {
                commonData["Size QCPU S" + std::to_string(i + 1)] = std::to_string(deviceCPU->getStage(i)->getMaxQueueSize());
                commonData["Num. Cores CPU S" + std::to_string(i + 1)] = std::to_string(deviceCPU->getStage(i)->getTotalCores());
            }
        }
    }

    // Info about the SYCL kernel used for the pairwise distance
    if constexpr (__PWDIST__ == 1) {
        commonData["Vectorization"] = "yes";
        commonData["Vec. Type GPU"] = "float";
        commonData["Vec. Type CPU"] = "float";
        commonData["Tile Size CPU"] = 64;
        commonData["Tile Size GPU"] = 16;
    } else if constexpr (__PWDIST__ == 2) {
        commonData["Vectorization"] = "yes";
        commonData["Vec. Type GPU"] = "sycl::float4";
        commonData["Vec. Type CPU"] = "sycl::float4";
        commonData["Tile Size CPU"] = 64;
        commonData["Tile Size GPU"] = 16;
    } else if constexpr (__PWDIST__ == 3) {
        commonData["Vectorization"] = "no";
        commonData["Vec. Type GPU"] = "float";
        commonData["Vec. Type CPU"] = "float";
    }

    // ApplicationData variables for variable data
    variableData["Num. Frames"] = inputArgs.numFrames;
    variableData["Throughput (FPS)"] = appData.throughput;
    variableData["Tot. Time (ms)"] = appData.totalTime;

    if constexpr (ADVANCEDMETRICS_ENABLED) {
        for (auto i = 0u; i < appData.numFiltersGPU.size(); ++i) {
            variableData["Num. Filters GPU S" + std::to_string(i + 1)] = appData.numFiltersGPU[i].load();
        }
        for (auto i = 0u; i < appData.numFiltersCPU.size(); ++i) {
            variableData["Num. Filters CPU S" + std::to_string(i + 1)] = appData.numFiltersCPU[i].load();
        }
    } else {
        // Filters processed by CPU and GPU
        int numFiltersGPU = appData.numFiltersGPU[0].load() + appData.numFiltersGPU[1].load() + appData.numFiltersGPU[2].load();
        int numFiltersCPU = appData.numFiltersCPU[0].load() + appData.numFiltersCPU[1].load() + appData.numFiltersCPU[2].load();
        if (numFiltersCPU > 0) {
            variableData["Num. Filters CPU"] = numFiltersCPU;
        }
        if (numFiltersGPU > 0) {
            variableData["Num. Filters GPU"] = numFiltersGPU;
        }
    }

    auto safe_divide = [](double numerator, int denominator) -> double {
        return (denominator == 0) ? 0 : numerator / denominator;
    };

    if constexpr (ADVANCEDMETRICS_ENABLED) {
        for (auto i = 0u; i < appData.time_GPU_S.size(); ++i) {
            variableData["Tot. Time GPU S" + std::to_string(i + 1) + " (ms)"] = appData.time_GPU_S[i];
            variableData["Avg. Time GPU S" + std::to_string(i + 1) + " (ms)"] = safe_divide(appData.time_GPU_S[i], appData.numFiltersGPU[i].load());
        }
        for (auto i = 0u; i < appData.time_CPU_S.size(); ++i) {
            variableData["Tot. Time CPU S" + std::to_string(i + 1) + " (ms)"] = appData.time_CPU_S[i];
            variableData["Avg. Time CPU S" + std::to_string(i + 1) + " (ms)"] = safe_divide(appData.time_CPU_S[i], appData.numFiltersCPU[i].load());
        }
    }

    if constexpr (TIMESTAGES_ENABLED) {
        for (auto i = 0u; i < appData.time_GPU_S.size(); ++i)
            variableData["Time GPU S" + std::to_string(i + 1) + " (ms)"] = safe_divide(appData.time_GPU_S[i], appData.numFiltersGPU[i].load());
        for (auto i = 0u; i < appData.time_CPU_S.size(); ++i)
            variableData["Time CPU S" + std::to_string(i + 1) + " (ms)"] = safe_divide(appData.time_CPU_S[i], appData.numFiltersCPU[i].load());
    }

    if constexpr (AUTOMODE_ENABLED) {
        variableData["Num. Frames Sampled"] = inputArgs.sampleFrames;
        variableData["Th. Balance (FPS)"] = appData.throughputBalance;
        variableData["Th. System (FPS)"] = appData.throughputSystem;
        variableData["Tot. Time Balance (ms)"] = appData.sampleTime;
        variableData["Tot. Time System (ms)"] = appData.systemTime;
        variableData["Th. System Expected (FPS)"] = appData.throughputSystemExpected;
    }

    if constexpr (ENERGYPCM_ENABLED) {
        variableData["CPU Energy (J)"] = appData.energyCPU;
        variableData["GPU Energy (J)"] = appData.energyGPU;
        variableData["Uncore Energy (J)"] = appData.energyUncore;
        variableData["Total Energy (J)"] = appData.energyTotal;
        variableData["Total Power (W)"] = appData.avgPower_W;
        variableData["Total Kilowatt-hours consumed (kWh)"] = appData.totalKilowattHours;
    }

    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Built data map successfully." << std::endl;
    }

    return {commonData, variableData};
}

void JSONFile::writeVariablesToJSON(const ApplicationData &appData, const InputArgs &inputArgs, int argc, char *argv[]) {
    if constexpr (VERBOSE_ENABLED) {
        std::cout << "Starting to write variables to JSON." << std::endl;
    }
    // Prepare directory and filename
    std::string directory = prepareDirectory(inputArgs, argv[0]);
    std::string acquisitionMode = "";
    if constexpr (__ACQMODE__ == 1) {
        acquisitionMode = "_fillcompletely";
    } else if constexpr (__ACQMODE__ == 2) {
        acquisitionMode = "_noqueue";
    }

    std::filesystem::create_directories(directory);
    std::string filename = directory + "/" + PipelineFactory::getPipelineTypeAsShortString(inputArgs.pipelineName) + "_" + inputArgs.getImageTypeToString() + acquisitionMode + ".json";

    // Load existing JSON if it exists
    loadFromFile(filename);

    // Build data map
    auto [commonData, variableData] = buildDataMap(appData, inputArgs);

    // Generate common key for this set of data
    std::string commonKey = generateCommonKey(appData, inputArgs);

    // Create nested structure
    std::string api = commonData["API"];
    std::string randomKey = generateRandomKey();

    if (m_json[api].contains(commonKey)) {
        m_json[api][commonKey][randomKey] = variableData;
    } else {
        m_json[api][commonKey] = commonData;
        m_json[api][commonKey][randomKey] = variableData;
    }

    // Save JSON to file
    saveToFile(filename);

    if constexpr (VERBOSE_ENABLED) {
        std::cout << "JSON file created: " << filename << std::endl;
    }
}