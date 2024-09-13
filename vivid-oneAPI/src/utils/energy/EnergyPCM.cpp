#include "EnergyPCM.hpp"

using namespace pcm;

EnergyPCM::~EnergyPCM() {
    // Limpia los recursos de PCM
    if (m_pcm) {
        m_pcm->cleanup();
    }
}

EnergyPCM::EnergyPCM() : m_pcm{PCM::getInstance()} {
    m_pcm->resetPMU();
    if (m_pcm->program() != PCM::Success) {
        std::cerr << "Error in PCM library initialization" << std::endl;
        exit(-1);
    }
};

/*Sets the start mark of energy and time*/
void EnergyPCM::startEnergyMeasurement(InputArgs &inputArgs, ApplicationData &appData) {
    m_pcm->getAllCounterStates(sstate1, sktstate1, cstates1);
    start = tbb::tick_count::now();
}

void EnergyPCM::stopEnergyMeasurement(InputArgs &inputArgs, ApplicationData &appData) {
    end = tbb::tick_count::now();                             // Marca el tiempo de finalización
    m_pcm->getAllCounterStates(sstate2, sktstate2, cstates2); // Obtiene los estados finales

    // Energía de la CPU (dominio PP0, o power plane 0)
    appData.energyCPU = getConsumedJoules(0, sktstate1[0], sktstate2[0]);

    // Energía de la GPU integrada (dominio PP1, o power plane 1)
    // Si no hay GPU integrada, esto debería devolver 0
    appData.energyGPU = getConsumedJoules(1, sktstate1[0], sktstate2[0]);

    // Energía total del socket (incluye CPU, GPU y Uncore)
    appData.energyTotal = getConsumedJoules(sktstate1[0], sktstate2[0]);

    // Energía Uncore = Energía Total - Energía CPU - Energía GPU
    appData.energyUncore = appData.energyTotal - appData.energyCPU - appData.energyGPU;

    // Cálculo de la potencia promedio en vatios (W)
    appData.avgPower_W = appData.energyTotal / (end - start).seconds();

    // Conversión a kilovatios-hora (kWh)
    appData.totalKilowattHours = (appData.avgPower_W * ((end - start).seconds() / 3600.0)) / 1000.0;

    // Imprimir resultados
    std::cout << "Energy CPU (J): " << appData.energyCPU << " J" << std::endl;
    std::cout << "Energy GPU (J): " << appData.energyGPU << " J" << std::endl;
    std::cout << "Energy Total (J): " << appData.energyTotal << " J" << std::endl;
    std::cout << "Energy Uncore (J): " << appData.energyUncore << " J" << std::endl;
    std::cout << "Average Power (W): " << appData.avgPower_W << " W" << std::endl;
    std::cout << "Total Kilowatt-hours consumed: " << appData.totalKilowattHours << " kWh" << std::endl;
}