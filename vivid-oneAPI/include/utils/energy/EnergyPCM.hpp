#ifndef ENERGY_PCM_H
#define ENERGY_PCM_H

#include "../../../../pcm/src/bw.h"
#include "../../../../pcm/src/cpucounters.h"
#include "../../../../pcm/src/debug.h"
#include "../../../../pcm/src/mmio.h"
#include "../../../../pcm/src/msr.h"
#include "../../../../pcm/src/pci.h"
#include "../../../../pcm/src/pcm-accel-common.h"
#include "../../../../pcm/src/resctrl.h"
#include "../../../../pcm/src/threadpool.h"
#include "../../../../pcm/src/topology.h"
#include "../../../../pcm/src/uncore_pmu_discovery.h"
#include "../../../../pcm/src/utils.h"
#include "ApplicationData.hpp"
#include "InputArgs.hpp"
#include <iomanip>
#include <iostream>
#include <oneapi/tbb.h>
#include <string>
#include <vector>

using namespace pcm;
using namespace std;

class EnergyPCM {
  public:
    pcm::PCM *m_pcm;
    std::vector<pcm::CoreCounterState> cstates1, cstates2;
    std::vector<pcm::SocketCounterState> sktstate1, sktstate2;
    pcm::SystemCounterState sstate1, sstate2;
    tbb::tick_count start, end;

    FILE *file_temperatue_CPU;
    std::vector<double> temperature_CPU;
    float averageTemperature_CPU;
    double minTemperature_CPU;
    double maxTemperature_CPU;

    EnergyPCM();
    ~EnergyPCM();
    void startEnergyMeasurement(InputArgs &inputArgs, ApplicationData &appData);
    void stopEnergyMeasurement(InputArgs &inputArgs, ApplicationData &appData);
};

#endif