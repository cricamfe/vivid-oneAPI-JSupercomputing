#include "ApplicationData.hpp"
#include "Comparer.hpp"
#include "DataBuffers.hpp"
#if ENERGYPCM_ENABLED
#include "EnergyPCM.hpp"
#endif
#include "GlobalParameters.hpp"
#include "ImageUtils.hpp"
#include "InputArgs.hpp"
#include "PipelineFactory.hpp"
#include "Results.hpp"
#include "SYCLUtils.hpp"
#include "Timer.hpp"
#include "Tracer.hpp"
#include "circular-buffer.hpp"
#include "jsonfile.hpp"
#include "pipeline_template.hpp"
#include <sycl/sycl.hpp>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {
    // ____________________________________________________________________________________________________________________
    // 0. Define the variables that will be used in the pipeline and check the arguments
    // ____________________________________________________________________________________________________________________
    // Create the application data
    ApplicationData appData;
    // Parse the input arguments
    InputArgs inputArgs(argc, argv);

    // ____________________________________________________________________________________________________________________
    // 2. Configure the pipeline
    // ____________________________________________________________________________________________________________________
    // Create Image object
    Image imageData;
    imageData.loadImageData(inputArgs.imageResolution, appData.height, appData.width);

    // Configure the SYCL queue if we are using SYCL as backend for filters or the pipeline
    sycl::queue Q_GPU, Q_CPU;
    SYCLUtils::configureSYCLQueues(Q_GPU, Q_CPU, inputArgs.nThreads, (inputArgs.pipelineName == PipelineType::SYCLEvents) ? true : false);
    appData.selectUSMQueue(Q_GPU); // We need to use this when use USM queue

    // Configure all the buffers (GlobalFrame, FilterBank, GlobalCla)
    DataBuffers::createAllBuffers(appData, imageData.getImageData());
    // Create the circular buffer for the items of the pipeline (default: 8*inFlightFrames)
    circular_buffer bufferItems{inputArgs.sizeCircularBuffer, appData.globalFrame, appData.globalCla, appData.numFilters, appData.USM_queue};

    // ____________________________________________________________________________________________________________________
    // 3. Configure some output variables
    // ____________________________________________________________________________________________________________________
    // Calculate the golden output used for debugging
    if constexpr (DEBUG_ENABLED) {
        Comparer::createGoldenFrame(appData);
    }
    // Create the tracer object for the pipeline
    Tracer traceFile;
    if constexpr (TRACE_ENABLED) {
        createTraceFile(argc, argv, traceFile);
    }

    // ____________________________________________________________________________________________________________________
    // 4. Execute the pipeline
    // ____________________________________________________________________________________________________________________
    // Determinar el tipo de pipeline a ejecutar
    // Create the pipeline
    auto pipeline = PipelineFactory::createPipeline(inputArgs.pipelineName);
    if constexpr (VERBOSE_ENABLED) {
        std::cout << " Running " << PipelineFactory::getPipelineTypeString(inputArgs.pipelineName) << " version..." << std::endl;
    }

#if ENERGYPCM_ENABLED
    EnergyPCM *energyPCM;
    energyPCM = new EnergyPCM();
    energyPCM->startEnergyMeasurement(inputArgs, appData);
#endif

    // Execute the pipeline
    pipeline->executePipeline(appData, inputArgs, bufferItems, traceFile, Q_GPU, Q_CPU);

// // Stop the energy measurement
#if ENERGYPCM_ENABLED
    energyPCM->stopEnergyMeasurement(inputArgs, appData);
#endif

    // ____________________________________________________________________________________________________________________
    // 5. Print the results
    // ____________________________________________________________________________________________________________________
    // Compute and display the results
    calculateAndDisplayResults(appData, inputArgs);

    // ____________________________________________________________________________________________________________________
    // 6. Export the results to a file (JSON)
    // ____________________________________________________________________________________________________________________
    // Write the results to a CSV file if the flag is set
    if constexpr (JSON_ENABLED) {
        JSONFile jsonFile;
        jsonFile.writeVariablesToJSON(appData, inputArgs, argc, argv);
    }

    std::exit(EXIT_SUCCESS);

    return 0;
}