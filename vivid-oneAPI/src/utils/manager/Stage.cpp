#include "Stage.hpp"
#include "GlobalParameters.hpp"

// Constructor
Stage::Stage(int cores, int queueSize)
    : totalCores(cores), usedCores(0), maxQueueSize(queueSize) {
    if constexpr (LOG_ENABLED) {
        std::clog << "Stage created with " << cores << " cores and a max queue size of " << queueSize << ".\n";
    }
}

// Attempt to acquire a core for this stage.
AcquisitionStatus Stage::acquireCore() {
    std::unique_lock<std::mutex> lock(mtx);
    if (totalCores > 0 && usedCores.load(std::memory_order_relaxed) < totalCores) {
        usedCores.fetch_add(1, std::memory_order_relaxed);
        lock.unlock(); // Unlock as soon as possible to minimize contention
        if constexpr (LOG_ENABLED) {
            std::clog << "Stage acquired a core. " << usedCores.load(std::memory_order_relaxed) << "/" << totalCores << " cores in use with thread " << std::this_thread::get_id() << ".\n";
        }
        return AcquisitionStatus::AcquiredCore;
    } else {
        if constexpr (LOG_ENABLED) {
            std::clog << "No cores available in the stage or totalCores is 0.\n";
        }
        return AcquisitionStatus::Failed;
    }
}

// Attempt to enqueue a task and wait for a core.
AcquisitionStatus Stage::acquireQueue() {
    std::unique_lock<std::mutex> lock(mtx);

    // Verificar si el total de cores es 0 o si el tamaño máximo de la cola es 0
    if (totalCores == 0 || maxQueueSize == 0) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Stage has 0 total cores or queue is not allowed. Acquisition failed.\n";
        }
        return AcquisitionStatus::Failed;
    }

    if (waitQueue.size() < maxQueueSize) {
        waitQueue.push(std::this_thread::get_id());
        if constexpr (LOG_ENABLED) {
            std::clog << "Thread " << std::this_thread::get_id() << " enqueued, waiting for a core. Queue size is now " << waitQueue.size() << "/" << maxQueueSize << ".\n";
        }

        cv.wait(lock, [this]() {
            return (usedCores.load(std::memory_order_relaxed) < totalCores) && (waitQueue.front() == std::this_thread::get_id());
        });

        if (usedCores.load(std::memory_order_relaxed) < totalCores) {
            usedCores.fetch_add(1, std::memory_order_relaxed);
            waitQueue.pop();
            if constexpr (LOG_ENABLED) {
                std::clog << "Thread " << std::this_thread::get_id() << " acquired a core after waiting. " << usedCores.load(std::memory_order_relaxed) << "/" << totalCores << " cores in use.\n";
                std::clog << "Queue size after dequeue is " << waitQueue.size() << "/" << maxQueueSize << ".\n";
            }
            return AcquisitionStatus::Enqueued;
        } else {
            if constexpr (LOG_ENABLED) {
                std::clog << "Thread " << std::this_thread::get_id() << " failed to acquire a core after waiting.\n";
            }
            return AcquisitionStatus::Failed;
        }
    } else {
        if constexpr (LOG_ENABLED) {
            std::clog << "Queue is full or maxQueueSize is 0, cannot enqueue thread. Queue size is " << waitQueue.size() << "/" << maxQueueSize << ".\n";
        }
        return AcquisitionStatus::Failed;
    }
}

// Release a core, making it available for other tasks.
void Stage::release() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (usedCores.load(std::memory_order_relaxed) > 0) {
            usedCores.fetch_sub(1, std::memory_order_relaxed);
            if constexpr (LOG_ENABLED) {
                std::clog << "Stage released a core. " << usedCores.load(std::memory_order_relaxed) << "/" << totalCores << " cores in use with thread " << std::this_thread::get_id() << ".\n";
            }
        }
    }
    cv.notify_all(); // Notify all waiting threads
}

// Get the number of cores currently in use.
int Stage::getUsedCores() {
    return usedCores.load(std::memory_order_relaxed);
}

// Get the current size of the wait queue.
int Stage::getQueueSize() {
    std::lock_guard<std::mutex> lock(mtx);
    return waitQueue.size();
}

// Get the total number of cores in this stage.
int Stage::getTotalCores() {
    return totalCores;
}

// Get the maximum size of the wait queue.
int Stage::getMaxQueueSize() {
    return maxQueueSize;
}

// Set the total number of cores in this stage.
void Stage::setTotalCores(int cores) {
    std::lock_guard<std::mutex> lock(mtx);
    totalCores = cores;
    if constexpr (LOG_ENABLED) {
        std::clog << "Stage total cores set to " << totalCores << ".\n";
    }
}

// Set the maximum size of the wait queue.
void Stage::setMaxQueueSize(int queueSize) {
    std::lock_guard<std::mutex> lock(mtx);
    maxQueueSize = queueSize;
    if constexpr (LOG_ENABLED) {
        std::clog << "Stage max queue size set to " << maxQueueSize << ".\n";
    }
}