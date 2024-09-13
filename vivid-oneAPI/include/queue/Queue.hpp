#ifndef QUEUE_HPP
#define QUEUE_HPP

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// Estructura para almacenar los resultados de la función `calculateWaitTime`
struct WaitTimeResults {
    double Lq;
    double Wq;
    double rate;
    double prob_occupancy;
    double ro;
};

class MMcKKModel {
  public:
    static WaitTimeResults calculateWaitTime(double arrival, double active, int nstream, int nterm);
};

struct OptConfResults {
    double lambdaOpt;
    std::string confOptP;
    int confOptS;
    double lambdae;
    int ntokens;
    int cP;
    int cS;
    double lambdaGP;
    double lambdaCP;
    double lambdaGS;
    double lambdaCS;
    int NGP;
    int NCP;
    int NGS;
    int NCS;
    int stageBotl;
    std::string devBot;
};

class PipelineOptimizer {
  public:
    static std::vector<OptConfResults> findOptimalConfiguration(int nstages, std::vector<double> thC, std::vector<double> thG, int nc, bool debug = false);

  private:
    static std::pair<double, int> findMinWithIndex(const std::vector<double> &v);
    static void calculateTser(const std::string &conf, int nstages, int nc, const std::vector<double> &thC, const std::vector<double> &thG,
                              double &TserCP, double &TserGP, double &TserCS, double &TserGS);
    static void evaluateConfiguration(int nc, double TserGP, double TserCP, double TserCS, double TserGS,
                                      double lambdaGP, double lambdaCP, double &p, int &Sdev, double &lambda_val, double &rhoG, double &rhoC);
    static OptConfResults calculateOptimalConfiguration(int nstages, int configCount, const std::vector<double> &lambda_vals,
                                                        const std::vector<double> &p, const std::vector<int> &Sdev,
                                                        const std::vector<double> &TserGP, const std::vector<double> &TserCP,
                                                        const std::vector<double> &TserCS, const std::vector<double> &TserGS,
                                                        int nc, const std::vector<double> &rhoG, const std::vector<double> &rhoC,
                                                        int stageBotl, std::string devBot,
                                                        bool debug);
};

#endif // QUEUE_HPP
