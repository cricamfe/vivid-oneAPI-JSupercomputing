#ifndef DEVICE_HPP
#define DEVICE_HPP

#include "AcquisitionStatus.hpp"
#include "GlobalParameters.hpp"
#include "Stage.hpp"
#include <condition_variable>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * @class Device
 * @brief Represents a computing device with multiple stages and cores.
 *
 * This class manages the stages and cores of a computing device, handling core
 * acquisition, task queuing, and core release across different stages. It
 * supports both CPU and GPU devices.
 */
class Device {
  private:
    Acc acc;                                               ///< Type of accelerator (CPU or GPU).
    std::vector<std::unique_ptr<Stage>> stages;            ///< List of stages in the device.
    std::unordered_map<int, int> stageIndexMap;            ///< Map from stage index to actual stage.
    int totalCores;                                        ///< Total number of cores in the device.
    std::atomic<int> usedCores;                            ///< Number of cores currently in use.
    std::atomic<int> totalQueuedTasks;                     ///< Total number of tasks currently enqueued across all stages.
    const int maxTotalQueuedTasks = 16;                    ///< Maximum total number of enqueued tasks allowed for the device.
    std::mutex mtx;                                        ///< Mutex for synchronizing access to cores and stages.
    std::condition_variable cv;                            ///< Condition variable for managing core availability.
    std::queue<std::pair<int, std::thread::id>> waitQueue; ///< Queue for waiting tasks, storing stage index and thread ID.

  public:
    /**
     * @brief Constructor for the Device class.
     * @param acc Type of accelerator (CPU or GPU).
     * @param cores Total number of cores available in this device.
     */
    Device(Acc acc, int cores);

    /**
     * @brief Removes a stage from the device.
     * @param stage_ID Identifier for the stage to be removed.
     */
    void removeStage(int stage_ID);

    /**
     * @brief Adds a stage to the device.
     * @param stage_ID Identifier for the stage.
     * @param maxCores Maximum number of cores the stage can use.
     * @param maxQueueSize Maximum number of tasks that can be enqueued in the stage.
     */
    void addStage(int stage_ID, int maxCores, int maxQueueSize);

    /**
     * @brief Maps a stage index to an actual stage.
     * @param virtualIndex The virtual index to be mapped.
     * @param actualStageIndex The actual stage index to map to.
     */
    void mapStageIndex(int virtualIndex, int actualStageIndex);

    /**
     * @brief Updates the mapping of stage indices.
     * @param newMapping The new mapping from virtual indices to actual stage indices.
     */
    void updateStageMapping(const std::unordered_map<int, int> &newMapping);

    /**
     * @brief Gets a specific stage by index.
     * @param stageIndex Index of the stage to retrieve.
     * @return Pointer to the Stage object.
     */
    Stage *getStage(int stageIndex);

    /**
     * @brief Attempts to acquire a core for a specific stage.
     * @param stageIndex Index of the stage to acquire a core for.
     * @return AcquisitionStatus indicating the result of the core acquisition attempt.
     */
    AcquisitionStatus acquireCore(int stageIndex);

    /**
     * @brief Attempts to enqueue a task for a specific stage and wait for a core.
     * @param stageIndex Index of the stage to enqueue the task for.
     * @return AcquisitionStatus indicating the result of the queuing attempt.
     */
    AcquisitionStatus acquireQueue(int stageIndex);

    /**
     * @brief Releases a core for a specific stage.
     * @param stageIndex Index of the stage to release the core for.
     */
    void release(int stageIndex);

    /**
     * @brief Gets the list of stages in the device.
     * @return Reference to the vector of unique pointers to Stage objects.
     */
    std::vector<std::unique_ptr<Stage>> &getStages();

    /**
     * @brief Gets the number of cores currently in use.
     * @return The number of used cores.
     */
    int getUsedCores();

    /**
     * @brief Gets the current size of the wait queue for a specific stage.
     * @param stageIndex Index of the stage to check the queue size for.
     * @return The number of tasks in the wait queue.
     */
    int getQueueSize(int stageIndex);

    /**
     * @brief Gets the total number of cores in the device.
     * @return The total number of cores.
     */
    int getTotalCores();

    /**
     * @brief Gets the maximum queue size for a specific stage.
     * @param stageIndex Index of the stage to check the maximum queue size for.
     * @return The maximum queue size.
     */
    int getMaxQueueSize(int stageIndex);

    /**
     * @brief Gets the type of accelerator.
     * @return The type of accelerator (CPU or GPU).
     */
    Acc getAcc();

    /**
     * @brief Gets the type of accelerator as a string.
     * @return The type of accelerator as a string ("CPU" or "GPU").
     */
    std::string getAccStr();
};

#endif // DEVICE_HPP
