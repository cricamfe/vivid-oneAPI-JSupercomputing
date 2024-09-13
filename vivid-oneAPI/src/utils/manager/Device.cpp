#include "Device.hpp"
#include "GlobalParameters.hpp"
#include <stdexcept>

// Constructor
Device::Device(Acc acc, int cores)
    : acc(acc), totalCores(cores), usedCores(0), totalQueuedTasks(0) {
    if constexpr (LOG_ENABLED) {
        std::clog << "Device " << getAccStr() << " created with " << cores << " cores.\n";
    }
}

// Remove a stage from the device
void Device::removeStage(int stage_ID) {
    std::unique_lock<std::mutex> lock(mtx);
    if (stage_ID < stages.size() && stages[stage_ID]) {
        stages[stage_ID].reset();
        stageIndexMap.erase(stage_ID); // Remove any mapping related to this stage
        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << getAccStr() << " stage " << stage_ID << " removed.\n";
        }
    } else {
        if constexpr (LOG_ENABLED) {
            std::clog << "Attempt to remove non-existing stage " << stage_ID << ".\n";
        }
    }
}

// Add a stage to the device
void Device::addStage(int stage_ID, int maxCores, int maxQueueSize) {
    std::unique_lock<std::mutex> lock(mtx);
    if (stage_ID >= stages.size()) {
        stages.resize(stage_ID + 1);
    }
    stages[stage_ID] = std::make_unique<Stage>(maxCores, maxQueueSize);
    if constexpr (LOG_ENABLED) {
        std::clog << "Device " << getAccStr() << " stage " << stage_ID << " added with " << maxCores << " maximum cores and " << maxQueueSize << " maximum queue size.\n";
    }
}

// Map a stage index to an actual stage
void Device::mapStageIndex(int virtualIndex, int actualStageIndex) {
    std::unique_lock<std::mutex> lock(mtx);
    if (actualStageIndex >= stages.size()) {
        throw std::out_of_range("Actual stage index out of range");
    }
    stageIndexMap[virtualIndex] = actualStageIndex;
    if constexpr (LOG_ENABLED) {
        std::clog << "Mapped virtual index " << virtualIndex << " to actual stage index " << actualStageIndex << ".\n";
    }
}

// Update the mapping of stage indices
void Device::updateStageMapping(const std::unordered_map<int, int> &newMapping) {
    std::unique_lock<std::mutex> lock(mtx);
    for (const auto &pair : newMapping) {
        if (pair.second >= stages.size()) {
            throw std::out_of_range("Actual stage index out of range");
        }
    }
    stageIndexMap = newMapping;
    if constexpr (LOG_ENABLED) {
        std::clog << "Updated stage mapping.\n";
    }
}

// Get a specific stage by index
Stage *Device::getStage(int stageIndex) {
    std::unique_lock<std::mutex> lock(mtx);
    auto it = stageIndexMap.find(stageIndex);
    if (it != stageIndexMap.end()) {
        return stages[it->second].get();
    }
    return nullptr;
}

// Attempt to acquire a core for a specific stage
AcquisitionStatus Device::acquireCore(int stageIndex) {
    auto stagePtr = getStage(stageIndex);
    if (!stagePtr) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Invalid stage index: " << stageIndex << ".\n";
        }
        return AcquisitionStatus::Failed;
    }

    // Verificar si el total de cores es 0, en cuyo caso, no proceder
    if (totalCores == 0) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << getAccStr() << " has 0 total cores. Acquisition failed.\n";
        }
        return AcquisitionStatus::Failed;
    }

    std::unique_lock<std::mutex> lock(mtx);
    if (usedCores.load(std::memory_order_relaxed) < totalCores) {
        AcquisitionStatus status = stagePtr->acquireCore();
        if (status == AcquisitionStatus::AcquiredCore) {
            usedCores.fetch_add(1, std::memory_order_relaxed);
            if constexpr (LOG_ENABLED) {
                std::clog << "Device " << getAccStr() << " acquired a core at stage " << stageIndex << " (" << getUsedCores() << "/" << totalCores << ") with thread " << std::this_thread::get_id() << ".\n";
            }
            return status;
        }
    }
    return AcquisitionStatus::Failed;
}

// Attempt to enqueue a task for a specific stage and wait for a core
AcquisitionStatus Device::acquireQueue(int stageIndex) {
    auto stagePtr = getStage(stageIndex);
    if (!stagePtr) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Invalid stage index: " << stageIndex << ".\n";
        }
        return AcquisitionStatus::Failed;
    }

    // Verificar si el total de cores es 0 o si el tamaño máximo de la cola es 0
    if (totalCores == 0 || maxTotalQueuedTasks == 0) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << getAccStr() << " has 0 total cores or queue is not allowed. Acquisition failed.\n";
        }
        return AcquisitionStatus::Failed;
    }

    if (totalQueuedTasks.load(std::memory_order_relaxed) >= maxTotalQueuedTasks) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << getAccStr() << " is saturated with queued tasks (" << totalQueuedTasks.load() << "/" << maxTotalQueuedTasks << "). Task rejected with thread " << std::this_thread::get_id() << ".\n";
        }
        return AcquisitionStatus::Failed;
    }

    if constexpr (LOG_ENABLED) {
        std::clog << "Thread " << std::this_thread::get_id() << " attempting to enqueue task on " << getAccStr() << " Stage " << stageIndex << "\n";
    }

    // Attempt to enqueue the task on the `Stage` without blocking on the `Device`
    AcquisitionStatus status = stagePtr->acquireQueue();
    if (status == AcquisitionStatus::Enqueued) {
        if constexpr (LOG_ENABLED) {
            std::clog << getAccStr() << "> Stage " << stageIndex << " enqueued a task " << stagePtr->getQueueSize() << "/" << stagePtr->getMaxQueueSize() << " with thread " << std::this_thread::get_id() << ".\n";
        }
        std::unique_lock<std::mutex> lock(mtx);

        // Increment the total queued tasks count
        totalQueuedTasks.fetch_add(1, std::memory_order_relaxed);

        // Add the current thread and stage index to the device's wait queue
        waitQueue.push({stageIndex, std::this_thread::get_id()});

        // Wait until a core is available and this thread is first in the device's wait queue
        cv.wait(lock, [this]() {
            return (usedCores.load(std::memory_order_relaxed) < totalCores) && (waitQueue.front().second == std::this_thread::get_id());
        });

        // Now that we are the first thread in the queue and cores are available
        usedCores.fetch_add(1, std::memory_order_relaxed);
        totalQueuedTasks.fetch_sub(1, std::memory_order_relaxed); // Decrement total queued tasks count
        waitQueue.pop();

        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << getAccStr() << " acquired a core after waiting at stage " << stageIndex << " (" << getUsedCores() << "/" << totalCores << ") with thread " << std::this_thread::get_id() << ".\n";
        }
        return status;
    }

    if constexpr (LOG_ENABLED) {
        std::clog << "Thread " << std::this_thread::get_id() << " failed to enqueue task on " << getAccStr() << " Stage " << stageIndex << "\n";
    }

    return AcquisitionStatus::Failed;
}

// Release a core for a specific stage
void Device::release(int stageIndex) {
    auto stagePtr = getStage(stageIndex);
    if (stagePtr) {
        std::unique_lock<std::mutex> lock(mtx);
        if (usedCores.load(std::memory_order_relaxed) > 0) {
            stagePtr->release();
            usedCores.fetch_sub(1, std::memory_order_relaxed);
            cv.notify_all(); // Notify one waiting thread
            if constexpr (LOG_ENABLED) {
                std::clog << "Device " << getAccStr() << " released a core at stage " << stageIndex << " (" << getUsedCores() << "/" << totalCores << ") with thread " << std::this_thread::get_id() << ".\n";
            }
        } else {
            if constexpr (LOG_ENABLED) {
                std::clog << "Device " << getAccStr() << " tried to release a core at stage " << stageIndex << " but no cores were in use with thread " << std::this_thread::get_id() << ".\n";
            }
        }
    }
}

// Get the list of stages
std::vector<std::unique_ptr<Stage>> &Device::getStages() {
    std::unique_lock<std::mutex> lock(mtx);
    return stages;
}

// Get the number of cores currently in use at the device level
int Device::getUsedCores() {
    return usedCores.load();
}

// Get the queue size for a specific stage
int Device::getQueueSize(int stageIndex) {
    auto stagePtr = getStage(stageIndex);
    if (stagePtr) {
        return stagePtr->getQueueSize();
    }
    return 0;
}

// Get the total number of cores at the device level
int Device::getTotalCores() {
    return totalCores;
}

// Get the maximum queue size for a specific stage
int Device::getMaxQueueSize(int stageIndex) {
    auto stagePtr = getStage(stageIndex);
    if (stagePtr) {
        return stagePtr->getMaxQueueSize();
    }
    return 0;
}

// Get the type of accelerator
Acc Device::getAcc() {
    return acc;
}

// Get the type of accelerator as a string
std::string Device::getAccStr() {
    return ((acc == Acc::GPU) ? "GPU" : "CPU");
}
