#ifndef WORKLOAD_SIMULATOR_HPP
#define WORKLOAD_SIMULATOR_HPP

#include "GlobalParameters.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <thread>

/**
 * @class WorkloadSimulator
 * @brief Simulates a workload with a specified throughput distributed across a specified number of threads.
 */
class WorkloadSimulator {
  public:
    /**
     * @brief Constructs a WorkloadSimulator object.
     *
     * @param target_throughput The maximum throughput (frames per second or similar metric) that can be generated.
     * @param num_threads The number of threads (cores) over which the workload will be distributed.
     * @param variation_percentage The maximum percentage increase to introduce in the interval timing (default is 10%).
     *
     * @throws std::invalid_argument if target_throughput is less than or equal to zero, if num_threads is less than or equal to zero, or if variation_percentage is negative.
     */
    WorkloadSimulator(double target_throughput, int num_threads, double variation_percentage = 10.0);

    /**
     * @brief Simulates the workload according to the specified throughput and number of threads.
     */
    void simulate();

  private:
    double target_throughput;           ///< The target maximum throughput (e.g., frames per second).
    std::chrono::microseconds interval; ///< The base time interval between each simulated work unit.
    int num_threads;                    ///< The number of threads (cores) to be used in the simulation.
    double variation_percentage;        ///< The maximum percentage of increase to apply to the interval.

    /**
     * @brief Simulates the workload for a single core.
     *
     * This function simulates the workload for a single core by performing a busy-wait loop
     * for the duration of the calculated interval, applying an increase based on a percentage.
     */
    void simulate_core();
};

#endif // WORKLOAD_SIMULATOR_HPP
