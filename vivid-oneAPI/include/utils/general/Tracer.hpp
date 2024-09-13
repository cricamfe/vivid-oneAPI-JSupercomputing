/**
 * @file tracer.hpp
 * @brief This file contains the definition of the Tracer class.
 *
 * This header file contains the Tracer class used for performance tracing and
 * analysis. The class generates output in the Paje format which can be visualized
 * using tools such as Vite (http://vite.gforge.inria.fr/).
 */

#pragma once
#ifndef TRACER_TEMPLATE_H
#define TRACER_TEMPLATE_H

#include "GlobalParameters.hpp"
#include "oneapi/tbb.h"
#include "pipeline_template.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>

using namespace oneapi::tbb;
using namespace std;

using namespace Pipeline_template;

struct SyclEventInfo; // forward declaration

/**
 * @class Tracer
 * @brief A performance tracer for monitoring and analyzing code execution.
 *
 * This class generates trace files in the Paje format. The trace files contain
 * information about the execution of the code, such as the start and end times
 * of frames, CPU and GPU usage, and other performance-related data.
 */
class Tracer {
  private:
    tbb::queuing_mutex qmtx;         ///< Mutex for thread-safe access to the tracefile.
    std::ofstream tracefile;         ///< Output tracefile stream.
    tbb::tick_count start;           ///< Starting timestamp for the trace.
    std::set<thread::id> thread_set; ///< Set of thread IDs.
    bool threads;                    ///< Flag to indicate if threads are being used.

    /**
     * @brief Write the intro header to the tracefile.
     *
     * Writes the necessary Paje format header and definitions to the tracefile.
     */
    void write_intro_header();

  public:
    /**
     * @brief Constructor for Tracer with thread flag.
     * @param th Indicates if threads are being used (default: false).
     */
    Tracer(bool th = false);

    /**
     * @brief Constructor for Tracer with filename and thread flag.
     * @param filename The name of the output tracefile.
     * @param th Indicates if threads are being used (default: false).
     */
    Tracer(string filename, bool th = false);

    /**
     * @brief Destructor for the Tracer class.
     *
     * Closes the trace file and removes the thread.
     */
    ~Tracer();

    /**
     * @brief Open and initialize the tracefile.
     * @param filename The name of the output tracefile.
     */
    void open(string filename);

    /**
     * @brief Record the start of a frame.
     * @param[in] frame The frame ID.
     */
    void frame_start(ViVidItem *item);

    /**
     * @brief Record the end of a frame.
     * @param[in] frame The frame ID.
     */
    void frame_end(ViVidItem *item);

    /**
     * @brief Record the start of CPU execution.
     * @param[in] frame The frame ID.
     */
    void cpu_start(ViVidItem *item);

    /**
     * @brief Record the start of GPU execution.
     * @param[in] frame The frame ID.
     */
    void gpu_start(ViVidItem *item);

    /**
     * @brief Record the end of CPU execution.
     * @param[in] frame The frame ID.
     * @param[in] execution_time The execution time in seconds.
     */
    void cpu_end(ViVidItem *item, double execution_time);

    /**
     * @brief Record the end of GPU execution.
     * @param[in] frame The frame ID.
     * @param[in] execution_time The execution time in seconds.
     */
    void gpu_end(ViVidItem *item, double execution_time);

    /**
     * @brief Record the start of a wait
     *
     * @param item
     */
    void wait_start(ViVidItem *item);

    /**
     * @brief Record the end of a wait
     * @param[in] frame The frame ID.
     */
    void wait_end(ViVidItem *item);
};

/**
 * @brief Create a tracefile and initialize the Tracer object.
 * @param[in] argc The number of command line arguments.
 * @param[in] argv The command line arguments.
 * @param[in] my_tracer The Tracer object to initialize passed by reference.
 */
void createTraceFile(int argc, char *argv[], Tracer &my_tracer);

#endif // TRACER_TEMPLATE_H
