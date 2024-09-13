#include "WorkloadSimulator.hpp"

/**
 * @brief Constructs a WorkloadSimulator object with the specified throughput, number of threads, and optional variation percentage.
 *
 * @param target_throughput The maximum throughput (frames per second or similar metric) that can be generated.
 * @param num_threads The number of threads (cores) over which the workload will be distributed.
 * @param variation_percentage The maximum percentage increase to introduce in the interval timing (default is 10%).
 *
 * @throws std::invalid_argument if target_throughput is less than or equal to zero, if num_threads is less than or equal to zero, or if variation_percentage is negative.
 */
WorkloadSimulator::WorkloadSimulator(double target_throughput, int num_threads, double variation_percentage)
    : target_throughput{target_throughput}, num_threads{num_threads}, variation_percentage{variation_percentage} {
    if (target_throughput <= 0.0) {
        throw std::invalid_argument("Target throughput must be greater than zero.");
    }
    if (num_threads <= 0) {
        throw std::invalid_argument("Number of threads must be greater than zero.");
    }
    if (variation_percentage < 0.0) {
        throw std::invalid_argument("Variation percentage must be non-negative.");
    }

    // Calculate the effective throughput per core by dividing the target throughput by the number of threads
    double throughput_per_core = target_throughput / num_threads;

    // Calculate the base interval for each core
    interval = std::chrono::microseconds(static_cast<int>(1000000.0 / throughput_per_core));
}

/**
 * @brief Simulates the workload according to the specified throughput and number of threads.
 *
 * This method initiates the simulation of the workload by calling the simulate_core method,
 * which handles the busy-wait loop for the specified interval.
 */
void WorkloadSimulator::simulate() {
    // Simulate the workload for a single core with the effective throughput adjusted for the number of threads
    simulate_core();
}

/**
 * @brief Simulates the workload for a single core with proportional variability in the interval.
 *
 * This method performs a busy-wait loop for the duration of the calculated interval to simulate workload processing.
 * The loop periodically yields control to other threads to allow multitasking.
 * Variability is introduced as an increase to the interval based on a percentage of the base interval.
 */
void WorkloadSimulator::simulate_core() {
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize random number generator with a seed based on current time to ensure different results in consecutive runs
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> distribution(0.0, variation_percentage / 100.0);

    // Busy-wait loop to simulate workload processing with increase-based interval variation
    while (true) {
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Calculate a random increase within the specified variation percentage
        double increase_factor = distribution(generator);
        auto varied_interval = interval + std::chrono::microseconds(static_cast<int>(interval.count() * increase_factor));

        if (elapsed >= varied_interval) {
            break;
        }
        std::this_thread::yield(); // Yield to allow other threads to run
    }
}
