#include "SYCLUtils.hpp"

using namespace SYCLUtils;

SyclEventInfo::SyclEventInfo(sycl::event syclEvent, double kernelExecTime, Acc accType)
    : event(syclEvent), kernelExecTime_(kernelExecTime), accType_(accType) {}

SyclEventInfo::SyclEventInfo(const SyclEventInfo &other)
    : event(other.event), kernelExecTime_(other.kernelExecTime_), accType_(other.accType_) {}

sycl::event SyclEventInfo::getEvent() const {
    return event;
}

double SyclEventInfo::getKernelExecTime() const {
    return kernelExecTime_;
}

Acc SyclEventInfo::getAcceleratorType() const {
    return accType_;
}

void SyclEventInfo::setEvent(sycl::event newEvent) {
    event = newEvent;
}

sycl::nd_range<2> SYCLUtils::generate2DRange(std::size_t tileSize, std::size_t numRows, std::size_t numCols) {
    const std::size_t factorRows = (numRows + tileSize - 1) / tileSize;
    const std::size_t globalRows = factorRows * tileSize;

    const std::size_t factorCols = (numCols + tileSize - 1) / tileSize;
    const std::size_t globalCols = factorCols * tileSize;

    return {sycl::range<2>{globalRows, globalCols}, sycl::range<2>{tileSize, tileSize}};
}

void SYCLUtils::configureSYCLQueues(sycl::queue &gpuQueue, sycl::queue &cpuQueue, int numThreads, bool syclEventsEnabled) {
    // Select the properties of the queue
    sycl::property_list props;
    std::string propsStr = " SYCL Properties: ";
    if constexpr ((TRACE_ENABLED || TIMESTAGES_ENABLED || AUTOMODE_ENABLED || ADVANCEDMETRICS_ENABLED) && INORDER_QUEUE) {
        props = sycl::property_list{sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}};
        propsStr += "\n - sycl::property::queue::in_order\n - sycl::property::queue::enable_profiling";
    } else if constexpr ((TRACE_ENABLED || TIMESTAGES_ENABLED || AUTOMODE_ENABLED || ADVANCEDMETRICS_ENABLED) && !INORDER_QUEUE) {
        props = sycl::property_list{sycl::property::queue::enable_profiling{}};
        propsStr += "\n - sycl::property::queue::out_of_order\n - sycl::property::queue::enable_profiling";
    } else if constexpr (INORDER_QUEUE) {
        props = sycl::property_list{sycl::property::queue::in_order{}};
        propsStr += "\n - sycl::property::queue::in_order";
    } else {
        props = sycl::property_list{};
        propsStr += "\n - sycl::property::queue::out_of_order";
    }
    propsStr += " \n\n";

    // Select the backend to use for the GPU queue
    gpuQueue = sycl::queue{[](const sycl::device &d) {
                               sycl::backend desiredBackend;
                               const bool isGpu = d.get_info<sycl::info::device::device_type>() == sycl::info::device_type::gpu;

                               switch (__BACKEND__) {
                               case 0:
                                   desiredBackend = sycl::backend::opencl;
                                   break;
                               case 1:
                                   desiredBackend = sycl::backend::ext_oneapi_level_zero;
                                   break;
                               case 2:
                                   desiredBackend = sycl::backend::ext_oneapi_cuda;
                                   break;
                               default:
                                   throw std::runtime_error("Invalid backend specified.");
                               }

                               return isGpu && (d.get_platform().get_backend() == desiredBackend);
                           },
                           props};

    warmupSYCLDevice(gpuQueue);

    auto cpuName = sycl::device(sycl::cpu_selector_v).get_info<sycl::info::device::name>();
    if (SYCL_ENABLED || syclEventsEnabled) {
        sycl::device cpuDevice = sycl::device{sycl::cpu_selector_v};
        std::vector<size_t> maxComputeUnits(1, numThreads);
        std::vector<sycl::device> cpuSubDevices = cpuDevice.create_sub_devices<sycl::info::partition_property::partition_by_counts>(maxComputeUnits);
        cpuQueue = sycl::queue{cpuSubDevices[0], props};
        warmupSYCLDevice(cpuQueue);
    }

    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << " DEVICES" << std::endl;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::string kernelType = SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "std::simd" : "C++"));
    std::cout << " CPU(" << kernelType << "): " << cpuName << std::endl;
    std::cout << " GPU(" << getBackendName(gpuQueue.get_backend()) << "): " << gpuQueue.get_device().get_info<sycl::info::device::name>() << std::endl
              << std::endl;
    std::cout << propsStr;

    if constexpr (VERBOSE_ENABLED) {
        printQueueInfo(gpuQueue);
        if (SYCL_ENABLED || syclEventsEnabled)
            printQueueInfo(cpuQueue);
    }
}

void SYCLUtils::warmupSYCLDevice(sycl::queue &queue) {
    const size_t dataSize = 1024;
    std::vector<float> a(dataSize, 1.0f);
    std::vector<float> b(dataSize, 2.0f);
    std::vector<float> c(dataSize, 0.0f);
    sycl::buffer<float, 1> aBuf(a.data(), sycl::range<1>(dataSize));
    sycl::buffer<float, 1> bBuf(b.data(), sycl::range<1>(dataSize));
    sycl::buffer<float, 1> cBuf(c.data(), sycl::range<1>(dataSize));
    queue.submit([&](sycl::handler &cgh) {
        auto aAcc = aBuf.get_access<sycl::access::mode::read>(cgh);
        auto bAcc = bBuf.get_access<sycl::access::mode::read>(cgh);
        auto cAcc = cBuf.get_access<sycl::access::mode::write>(cgh);
        cgh.parallel_for<class warmupKernel>(sycl::range<1>(dataSize), [=](sycl::id<1> idx) {
            cAcc[idx] = aAcc[idx] + bAcc[idx];
        });
    });
    queue.wait();
}

std::string SYCLUtils::getBackendName(sycl::backend backend) {
    switch (backend) {
    case sycl::backend::opencl:
        return "OpenCL";
    case sycl::backend::ext_oneapi_level_zero:
        return "Level Zero";
    case sycl::backend::ext_oneapi_cuda:
        return "CUDA";
    default:
        return "Unknown";
    }
}

void SYCLUtils::printQueueInfo(const sycl::queue &queue) {
    const auto device = queue.get_device();
    const auto platform = device.get_platform();

    std::string deviceTypeStr;
    const auto deviceType = device.get_info<sycl::info::device::device_type>();
    switch (deviceType) {
    case sycl::info::device_type::cpu:
        deviceTypeStr = "CPU\n";
        break;
    case sycl::info::device_type::gpu:
        deviceTypeStr = "GPU\n";
        break;
    case sycl::info::device_type::accelerator:
        deviceTypeStr = "Accelerator\n";
        break;
    default:
        deviceTypeStr = "Other\n";
        break;
    }

    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << " QUEUE PROPERTIES " << deviceTypeStr;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
    std::cout << " Queue Order: " << (queue.has_property<sycl::property::queue::in_order>() ? "sycl::queue::in_order" : "sycl::queue::out_of_order") << "\n";
    std::cout << " Profiling: " << (queue.has_property<sycl::property::queue::enable_profiling>() ? "True" : "False") << "\n";
    std::cout << " Maximum number of compute units of the " + deviceTypeStr + " queue: " << queue.get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
    std::cout << "---------------------------------------------------------------------------------------" << std::endl;
}
