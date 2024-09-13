#ifndef RESOURCES_MANAGER_HPP
#define RESOURCES_MANAGER_HPP

#include "Device.hpp"
#include "GlobalParameters.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

class ResourcesManager {
  private:
    std::vector<std::unique_ptr<Device>> devices;
    std::string name;
    bool monitoring;
    std::thread monitorThread;

    void monitorResources();

  public:
    ResourcesManager(const std::string &name = "Manager");
    void addDevice(const Acc &acc, std::unique_ptr<Device> device);
    ~ResourcesManager();

    std::tuple<AcquisitionStatus, Acc> acquireForStage(int stageIndex, StageState stageState, Acc preferedAcc = Acc::OTHER);
    void releaseForStage(int stageIndex, Acc selectedAcc);
    void startMonitoring();
    void stopMonitoring();
    Device *getDevice(const Acc &selectedAcc) const;
};

#endif // RESOURCES_MANAGER_HPP