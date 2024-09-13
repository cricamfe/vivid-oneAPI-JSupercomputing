/**
 * @file sycl_utils.hpp
 * @brief Contains utilities for SYCL-based operations.
 *
 * This file contains utilities for SYCL-based operations, such as generating ranges, configuring the SYCL queues,
 * selecting the device backend and warming up the device before running the pipeline.
 */
#pragma once
#ifndef _SYCL_UTILS_HPP
#define _SYCL_UTILS_HPP

#include "GlobalParameters.hpp"
#include <sycl/sycl.hpp>

/**
 * @class SyclEventInfo
 * @brief Stores information about a SYCL event.
 */
class SyclEventInfo {
  public:
    sycl::event event;

    /**
     * @brief Default constructor.
     */
    SyclEventInfo() = default;

    /**
     * @brief Constructor with parameters.
     * @param syclEvent The SYCL event.
     * @param kernelExecTime The time taken by the kernel.
     * @param accType The type of accelerator used.
     */
    SyclEventInfo(sycl::event syclEvent, double kernelExecTime, Acc accType);

    /**
     * @brief Copy constructor.
     * @param other The SyclEventInfo object to copy from.
     */
    SyclEventInfo(const SyclEventInfo &other);

    /**
     * @brief Returns the SYCL event.
     * @return The SYCL event.
     */
    sycl::event getEvent() const;

    /**
     * @brief Returns the kernel execution time.
     * @return The time taken by the kernel.
     */
    double getKernelExecTime() const;

    /**
     * @brief Returns the accelerator type.
     * @return The type of accelerator used.
     */
    Acc getAcceleratorType() const;

    /**
     * @brief Sets the SYCL event.
     * @param newEvent The new SYCL event.
     */
    void setEvent(sycl::event newEvent);

  private:
    double kernelExecTime_;
    Acc accType_;
};

namespace SYCLUtils {
/**
 * @brief Generates a 2D range for a given tile size, row, and column.
 * @param tileSize The tile size.
 * @param numRows The number of rows.
 * @param numCols The number of columns.
 * @return A 2D range for the provided parameters.
 */
sycl::nd_range<2> generate2DRange(std::size_t tileSize, std::size_t numRows, std::size_t numCols);

/**
 * @brief Returns the name of the SYCL backend.
 * @param backend The backend to retrieve the name for.
 * @return A string representing the name of the backend.
 */
std::string getBackendName(sycl::backend backend);

/**
 * @brief Configures SYCL queues for GPU and CPU devices.
 * @param gpuQueue The SYCL queue for the GPU device.
 * @param cpuQueue The SYCL queue for the CPU device.
 * @param numThreads The number of threads to use for the CPU device.
 */
void configureSYCLQueues(sycl::queue &gpuQueue, sycl::queue &cpuQueue, int numThreads, bool syclEventsEnabled = false);

/**
 * @brief Warms up the SYCL device by running a simple vector addition kernel.
 * @param queue The SYCL queue on which to run the warm-up kernel.
 */
void warmupSYCLDevice(sycl::queue &queue);

/**
 * @brief Prints information about the SYCL queue.
 * @param queue The SYCL queue to print information about.
 */
void printQueueInfo(const sycl::queue &queue);

} // namespace SYCLUtils

#endif // _SYCL_UTILS_HPP
