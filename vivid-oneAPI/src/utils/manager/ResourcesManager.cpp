#include "ResourcesManager.hpp"
#include "GlobalParameters.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <unordered_map>

enum class AcquisitionMode {
    DEFAULT,
    PRIMARY_SECONDARY,
    NO_QUEUE
};

// Constructor
ResourcesManager::ResourcesManager(const std::string &name)
    : name(name), monitoring(false) {
    if constexpr (LOG_ENABLED) {
        std::clog << name << " creado.\n";
    }
}

// Agregar un dispositivo al administrador de recursos
void ResourcesManager::addDevice(const Acc &acc, std::unique_ptr<Device> device) {
    assert(device != nullptr); // Chequeo inicial de nulo

    Device *existingDevice = getDevice(acc);
    if (existingDevice != nullptr) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << device->getAccStr() << " already exists in " << this->name << ".\n";
        }
        return;
    }

    devices.push_back(std::move(device));

    if (device == nullptr) {
        if constexpr (LOG_ENABLED) {
            std::clog << "The device has been moved and is now nullptr.\n";
        }
    } else {
        if constexpr (LOG_ENABLED) {
            std::clog << "Error: the device still has a valid pointer after moving.\n";
        }
    }

    if (devices.back() == nullptr) {
        if constexpr (LOG_ENABLED) {
            std::clog << "Error: the moved device in the list is nullptr.\n";
        }
    } else {
        if constexpr (LOG_ENABLED) {
            std::clog << "Device " << devices.back()->getAccStr() << " added to " << this->name << ".\n";
        }
    }
}

// Destructor
ResourcesManager::~ResourcesManager() {
    stopMonitoring();
    if (monitorThread.joinable()) {
        monitorThread.join();
    }
}

std::tuple<AcquisitionStatus, Acc> ResourcesManager::acquireForStage(int stageIndex, StageState stageState, Acc preferredAcc) {
    static std::unordered_map<std::thread::id, Acc> lastUsedDevice; // Map to store the last used device for each thread
    std::mutex lastUsedDeviceMutex;                                 // Mutex to protect access to lastUsedDevice map

    // Helper function to acquire core or enqueue task on a device
    auto tryAcquireCore = [&](Device *device) -> std::tuple<AcquisitionStatus, Acc> {
        // Verificar si el dispositivo tiene cores disponibles
        if (device->getTotalCores() == 0) {
            if constexpr (LOG_ENABLED) {
                std::clog << "Device " << device->getAccStr() << " has 0 total cores. Skipping acquisition.\n";
            }
            return std::make_tuple(AcquisitionStatus::Failed, Acc::OTHER);
        }

        AcquisitionStatus status = device->acquireCore(stageIndex);
        if (status == AcquisitionStatus::AcquiredCore) {
            if constexpr (LOG_ENABLED) {
                std::clog << name << " > " << device->getAccStr() << " stage " << stageIndex << " acquired a core.\n";
            }
            std::lock_guard<std::mutex> lock(lastUsedDeviceMutex);
            lastUsedDevice[std::this_thread::get_id()] = device->getAcc(); // Record the last used device
            return std::make_tuple(status, device->getAcc());
        }
        return std::make_tuple(AcquisitionStatus::Failed, Acc::OTHER);
    };

    auto tryEnqueueTask = [&](Device *device) -> std::tuple<AcquisitionStatus, Acc> {
        // Verificar si el dispositivo tiene cores disponibles o colas habilitadas
        if (device->getTotalCores() == 0 || device->getMaxQueueSize(stageIndex) == 0) {
            if constexpr (LOG_ENABLED) {
                std::clog << "Device " << device->getAccStr() << " has 0 total cores or queues not enabled. Skipping enqueue.\n";
            }
            return std::make_tuple(AcquisitionStatus::Failed, Acc::OTHER);
        }

        AcquisitionStatus status = device->acquireQueue(stageIndex);
        if (status == AcquisitionStatus::Enqueued) {
            if constexpr (LOG_ENABLED) {
                std::clog << name << " > " << device->getAccStr() << " stage " << stageIndex << " enqueued a task.\n";
            }
            std::lock_guard<std::mutex> lock(lastUsedDeviceMutex);
            lastUsedDevice[std::this_thread::get_id()] = device->getAcc(); // Record the last used device
            return std::make_tuple(status, device->getAcc());
        }
        return std::make_tuple(AcquisitionStatus::Failed, Acc::OTHER);
    };

    // Function to try acquiring resources in a specific order
    auto acquireResources = [&](Device *firstDevice, Device *secondDevice, bool useQueue) -> std::tuple<AcquisitionStatus, Acc> {
        {
            std::lock_guard<std::mutex> lock(lastUsedDeviceMutex);
            Acc lastDeviceUsed = lastUsedDevice[std::this_thread::get_id()];

            // Only attempt to swap devices if the stageState is CPU_GPU
            if (stageState == StageState::CPU_GPU && firstDevice && firstDevice->getAcc() == lastDeviceUsed && secondDevice) {
                std::swap(firstDevice, secondDevice);
            }
        }

        if (firstDevice) {
            auto [status, acc] = tryAcquireCore(firstDevice);
            if (status == AcquisitionStatus::AcquiredCore) {
                return std::make_tuple(status, acc);
            }
        }

        if (secondDevice) {
            auto [status, acc] = tryAcquireCore(secondDevice);
            if (status == AcquisitionStatus::AcquiredCore) {
                return std::make_tuple(status, acc);
            }
        }

        if (useQueue) {
            if (firstDevice) {
                auto [status, acc] = tryEnqueueTask(firstDevice);
                if (status == AcquisitionStatus::Enqueued) {
                    return std::make_tuple(status, acc);
                }
            }

            if (secondDevice) {
                auto [status, acc] = tryEnqueueTask(secondDevice);
                if (status == AcquisitionStatus::Enqueued) {
                    return std::make_tuple(status, acc);
                }
            }
        }

        return std::make_tuple(AcquisitionStatus::Failed, Acc::OTHER);
    };

    // Determine the devices to use based on stageState
    Device *primaryDevice = getDevice(preferredAcc);
    Device *secondaryDevice = nullptr;
    for (const auto &device : devices) {
        if (device->getAcc() != preferredAcc) {
            secondaryDevice = device.get();
            break;
        }
    }

    AcquisitionMode mode = AcquisitionMode::DEFAULT;
    if constexpr (__ACQMODE__ == 1) {
        mode = AcquisitionMode::PRIMARY_SECONDARY;
    } else if constexpr (__ACQMODE__ == 2) {
        mode = AcquisitionMode::NO_QUEUE;
    }

    // Attempt to acquire resources based on AcquisitionMode
    switch (mode) {
    case AcquisitionMode::DEFAULT: {
        if (stageState == StageState::CPU) {
            return acquireResources(primaryDevice, nullptr, DEVICE_QUEUE_ENABLED);
        } else if (stageState == StageState::GPU) {
            return acquireResources(primaryDevice, nullptr, DEVICE_QUEUE_ENABLED);
        } else if (stageState == StageState::CPU_GPU) {
            return acquireResources(primaryDevice, secondaryDevice, DEVICE_QUEUE_ENABLED);
        }
        break;
    }
    case AcquisitionMode::PRIMARY_SECONDARY: {
        if (stageState == StageState::CPU) {
            return acquireResources(primaryDevice, nullptr, DEVICE_QUEUE_ENABLED);
        } else if (stageState == StageState::GPU) {
            return acquireResources(primaryDevice, nullptr, DEVICE_QUEUE_ENABLED);
        } else if (stageState == StageState::CPU_GPU) {
            // Try primary device first, then secondary
            if (primaryDevice) {
                auto [status, acc] = tryAcquireCore(primaryDevice);
                if (status == AcquisitionStatus::AcquiredCore) {
                    return std::make_tuple(status, acc);
                }
                if (DEVICE_QUEUE_ENABLED) { // Verificación adicional
                    auto [enqueueStatus, enqueueAcc] = tryEnqueueTask(primaryDevice);
                    if (enqueueStatus == AcquisitionStatus::Enqueued) {
                        return std::make_tuple(enqueueStatus, enqueueAcc);
                    }
                }
            }
            if (secondaryDevice) {
                auto [status, acc] = tryAcquireCore(secondaryDevice);
                if (status == AcquisitionStatus::AcquiredCore) {
                    return std::make_tuple(status, acc);
                }
                if (DEVICE_QUEUE_ENABLED) { // Verificación adicional
                    auto [enqueueStatus, enqueueAcc] = tryEnqueueTask(secondaryDevice);
                    if (enqueueStatus == AcquisitionStatus::Enqueued) {
                        return std::make_tuple(enqueueStatus, enqueueAcc);
                    }
                }
            }
            break;
        }
    }
    case AcquisitionMode::NO_QUEUE: {
        if (stageState == StageState::CPU) {
            return acquireResources(primaryDevice, nullptr, false);
        } else if (stageState == StageState::GPU) {
            return acquireResources(primaryDevice, nullptr, false);
        } else if (stageState == StageState::CPU_GPU) {
            return acquireResources(primaryDevice, secondaryDevice, false);
        }
        break;
    }
    }

    // If no resources could be acquired or queued
    if constexpr (LOG_ENABLED) {
        std::clog << name << " > No resources available for stage " << stageIndex << ".\n";
    }
    return std::make_tuple(AcquisitionStatus::Failed, Acc::OTHER);
}

// Liberar recursos para una etapa específica
void ResourcesManager::releaseForStage(int stageIndex, Acc acc) {
    Device *device = getDevice(acc);
    device->release(stageIndex);
    if constexpr (LOG_ENABLED) {
        std::clog << name << " > " << device->getAccStr() << " stage " << stageIndex << " ha liberado recursos.\n";
    }
}

// Monitor de recursos
void ResourcesManager::monitorResources() {
    while (monitoring) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::cout << "Monitoring Resources:\n";
        for (const auto &dev : devices) {
            std::cout << dev->getAccStr() << " > Global: " << dev->getUsedCores() << "/" << dev->getTotalCores() << " cores used.\n";
            for (size_t i = 0; i < dev->getStages().size(); ++i) {
                std::cout << dev->getAccStr() << " > Stage " << i << ": "
                          << dev->getStages()[i]->getUsedCores() << "/" << dev->getStages()[i]->getTotalCores() << " cores used, "
                          << "Queue size: " << dev->getStages()[i]->getQueueSize() << "\n";
            }
        }
    }
}

// Iniciar monitoreo
void ResourcesManager::startMonitoring() {
    monitoring = true;
    monitorThread = std::thread(&ResourcesManager::monitorResources, this);
    if constexpr (LOG_ENABLED) {
        std::clog << name << " > Monitoreo de recursos iniciado.\n";
    }
}

// Detener monitoreo
void ResourcesManager::stopMonitoring() {
    monitoring = false;
    if constexpr (LOG_ENABLED) {
        std::clog << name << " > Monitoreo de recursos detenido.\n";
    }
}

// Obtener dispositivo basado en el tipo de acelerador
Device *ResourcesManager::getDevice(const Acc &selectedAcc) const {
    for (const auto &dev : devices) {
        if (dev->getAcc() == selectedAcc) {
            return dev.get();
        }
    }
    return nullptr;
}
