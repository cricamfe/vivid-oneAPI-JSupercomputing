#ifndef STAGE_HPP
#define STAGE_HPP

#include "AcquisitionStatus.hpp"
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

/**
 * @class Stage
 * @brief Represents a stage in a multi-core device with queuing capabilities.
 *
 * This class manages the cores and the queue for a specific stage. It handles core acquisition,
 * queuing of tasks, and releasing of cores. It ensures threads can enqueue tasks and wait
 * efficiently for available cores.
 */
class Stage {
  private:
    int totalCores;                        ///< Total number of cores in this stage.
    std::atomic<int> usedCores;            ///< Number of cores currently in use.
    int maxQueueSize;                      ///< Maximum size of the wait queue.
    std::queue<std::thread::id> waitQueue; ///< Queue of threads waiting for cores.
    std::mutex mtx;                        ///< Mutex for synchronizing access.
    std::condition_variable cv;            ///< Condition variable for waiting threads.

  public:
    /**
     * @brief Constructor for the Stage class.
     * @param cores The total number of cores available in this stage.
     * @param queueSize The maximum number of threads that can wait in the queue.
     */
    Stage(int cores, int queueSize);

    /**
     * @brief Attempts to acquire a core for the stage.
     * @return AcquisitionStatus indicating the result of the core acquisition attempt.
     */
    AcquisitionStatus acquireCore();

    /**
     * @brief Attempts to enqueue a task and wait for a core.
     * @return AcquisitionStatus indicating the result of the queuing attempt.
     */
    AcquisitionStatus acquireQueue();

    /**
     * @brief Releases a core back to the stage, making it available for other tasks.
     */
    void release();

    /**
     * @brief Gets the number of cores currently in use.
     * @return The number of used cores.
     */
    int getUsedCores();

    /**
     * @brief Gets the current size of the wait queue.
     * @return The number of threads in the wait queue.
     */
    int getQueueSize();

    /**
     * @brief Gets the total number of cores in this stage.
     * @return The total number of cores.
     */
    int getTotalCores();

    /**
     * @brief Gets the maximum size of the wait queue.
     * @return The maximum queue size.
     */
    int getMaxQueueSize();

    /**
     * @brief Sets the total number of cores in this stage.
     * @param cores The new total number of cores.
     */
    void setTotalCores(int cores);

    /**
     * @brief Sets the maximum size of the wait queue.
     * @param queueSize The new maximum size of the wait queue.
     */
    void setMaxQueueSize(int queueSize);
};

#endif // STAGE_HPP
