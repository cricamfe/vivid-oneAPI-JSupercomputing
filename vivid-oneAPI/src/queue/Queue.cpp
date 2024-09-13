#include "Queue.hpp"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// Implementación de la función `calculateWaitTime`
WaitTimeResults MMcKKModel::calculateWaitTime(double arrival, double active, int nstream, int nterm) {
    double Lq = 0, Wq = 0, rate = 0, prob_occupancy = 0, ro = 0;
    double z = active / arrival;

    // Calculando f
    std::vector<double> f(nterm);
    for (int i = 0; i < nterm; ++i) {
        if (i <= nstream - 1) {
            f[i] = (nterm - i) * active / (arrival * (i + 1));
        } else {
            f[i] = (nterm - i) * active / (arrival * nstream);
        }
    }
    // Calculando fp
    std::vector<double> fp(nterm);
    for (int i = 0; i < nterm; ++i) {
        fp[i] = std::accumulate(f.begin(), f.begin() + i + 1, 1.0, std::multiplies<double>());
    }

    double sum_fp = std::accumulate(fp.begin(), fp.end(), 0.0);
    if (sum_fp == 0) {
        return {0, 0, 0, 0, 0};
    }
    double p0 = 1 / (1 + sum_fp);

    // Calculando pn
    std::vector<double> pn(nterm);
    for (int i = 0; i < nterm; ++i) {
        pn[i] = std::accumulate(f.begin(), f.begin() + i + 1, 1.0, std::multiplies<double>()) * p0;
    }

    prob_occupancy = std::accumulate(pn.begin() + std::min(nstream - 1, (int)pn.size()), pn.end(), 0.0);

    // Calculando L
    std::vector<double> p1(nterm);
    for (int i = 0; i < nterm; ++i) {
        p1[i] = (i + 1) * pn[i];
    }

    double L = std::accumulate(p1.begin(), p1.end(), 0.0);
    Lq = std::max(0.0, L - (z * (nterm - L)));
    Wq = (Lq * arrival) / (nterm - L);
    rate = (nterm - L) / arrival;
    ro = rate * active / nstream;

    return {Lq, Wq, rate, prob_occupancy, ro};
}

std::vector<OptConfResults> PipelineOptimizer::findOptimalConfiguration(int nstages, std::vector<double> thC, std::vector<double> thG, int nc, bool debug) {
    std::vector<char> confP(nstages, '0');

    auto [botlC, stC] = findMinWithIndex(thC);
    auto [botlG, stG] = findMinWithIndex(thG);

    int stageBotl = (botlC < botlG) ? stC : stG;
    confP[stageBotl] = (botlC < botlG) ? '1' : '0';

    int configCount = pow(2, nstages);
    std::vector<double> TserCP(configCount, 0), TserGP(configCount, 0), TserCS(configCount, 0), TserGS(configCount, 0);
    std::vector<double> p(configCount, 1), lambda_vals(configCount, 0);
    std::vector<int> Sdev(configCount, -1);
    std::vector<double> rhoG(configCount, 1), rhoC(configCount, 1);

    for (int i = 1; i < configCount; i++) {
        std::string conf = std::bitset<32>(i).to_string().substr(32 - nstages);
        if (i == configCount - 1) {
            for (int k = 0; k < nstages; ++k) {
                TserGP[i] += 1 / thG[k];
                TserCP[i] += 1 / (thC[k] / nc);
            }
            lambda_vals[i] = 1 / TserGP[i] + (1 / TserCP[i]) * nc;
        } else {
            if (conf[stageBotl] == confP[stageBotl]) {
                calculateTser(conf, nstages, nc, thC, thG, TserCP[i], TserGP[i], TserCS[i], TserGS[i]);

                if (TserGP[i] == 0 || TserCP[i] == 0) {
                    lambda_vals[i] = 0;
                    continue;
                }

                double lambdaGP = 1 / TserGP[i];
                double lambdaCP = (1 / TserCP[i]) * nc;
                double temp_rhoG, temp_rhoC;
                evaluateConfiguration(nc, TserGP[i], TserCP[i], TserCS[i], TserGS[i], lambdaGP, lambdaCP, p[i], Sdev[i], lambda_vals[i], temp_rhoG, temp_rhoC);
                rhoG[i] = temp_rhoG;
                rhoC[i] = temp_rhoC;
            }
        }
    }

    std::vector<OptConfResults> results;
    int top_configs = 5;
    for (int k = 0; k < top_configs; k++) {
        auto max_iter = max_element(lambda_vals.begin(), lambda_vals.end());
        if (*max_iter == 0) {
            break;
        }

        results.push_back(calculateOptimalConfiguration(nstages, configCount, lambda_vals, p, Sdev, TserGP, TserCP, TserCS, TserGS, nc, rhoG, rhoC, stageBotl, (botlC < botlG) ? "CPU" : "GPU", debug));

        // Invalidate the selected optimal configuration by setting its lambda value to 0
        *max_iter = 0;
    }

    // Ordenar los resultados por lambdae de mayor a menor
    std::sort(results.begin(), results.end(), [](const OptConfResults &a, const OptConfResults &b) {
        return a.lambdae > b.lambdae;
    });

    return results;
}

std::pair<double, int> PipelineOptimizer::findMinWithIndex(const std::vector<double> &v) {
    auto min_it = std::min_element(v.begin(), v.end());
    return {*min_it, static_cast<int>(std::distance(v.begin(), min_it))};
}

void PipelineOptimizer::calculateTser(const std::string &conf, int nstages, int nc, const std::vector<double> &thC, const std::vector<double> &thG,
                                      double &TserCP, double &TserGP, double &TserCS, double &TserGS) {
    for (int k = 0; k < nstages; k++) {
        if (conf[k] == '1') {
            TserGP += 1 / thG[k];
            TserCS += 1 / (thC[k] / nc);
        } else {
            TserCP += 1 / (thC[k] / nc);
            TserGS += 1 / thG[k];
        }
    }
}

void PipelineOptimizer::evaluateConfiguration(int nc, double TserGP, double TserCP, double TserCS, double TserGS,
                                              double lambdaGP, double lambdaCP, double &p, int &Sdev, double &lambda_val, double &rhoG, double &rhoC) {
    double lambdaG = lambdaGP, lambdaC = lambdaCP;
    rhoG = 1;
    rhoC = 1;

    if (lambdaGP < 0.9 * lambdaCP) {
        rhoC = ((nc * TserGP + TserCS) / (TserGP * TserCS)) * ((TserCP * TserCS) / (TserCP + TserCS)) * (1.0 / nc);
        int cP = std::max(static_cast<int>(round(nc * rhoC)), 1);
        int cS = nc - cP;
        if (rhoC < 0.8) {
            Sdev = 0;
            lambdaG = lambdaGP + ((1 / TserCS) * nc) * (1 - rhoC);
            lambdaC = lambdaCP * rhoC;
            p = lambdaGP / lambdaG;
        }
    } else if (lambdaCP < 0.9 * lambdaGP) {
        rhoG = ((nc * TserGS + TserCP) / (TserGS * TserCP)) * ((TserGP * TserGS) / (TserGP + TserGS));
        if (rhoG < 0.8) {
            Sdev = 1;
            lambdaG = lambdaGP * rhoG;
            lambdaC = lambdaCP + (1 / TserGS) * (1 - rhoG);
            p = lambdaCP / lambdaC;
        }
    }
    lambda_val = std::min(lambdaG, lambdaC);
}

OptConfResults PipelineOptimizer::calculateOptimalConfiguration(int nstages, int configCount, const std::vector<double> &lambda_vals,
                                                                const std::vector<double> &p, const std::vector<int> &Sdev,
                                                                const std::vector<double> &TserGP, const std::vector<double> &TserCP,
                                                                const std::vector<double> &TserCS, const std::vector<double> &TserGS,
                                                                int nc, const std::vector<double> &rhoG, const std::vector<double> &rhoC,
                                                                int stageBotl, std::string devBot,
                                                                bool debug) {
    double lambdaOpt = *std::max_element(lambda_vals.begin(), lambda_vals.end());
    int id_opt = std::distance(lambda_vals.begin(), std::max_element(lambda_vals.begin(), lambda_vals.end()));
    std::string confOptP = std::bitset<32>(id_opt).to_string().substr(32 - nstages);
    int confOptS = Sdev[id_opt];

    double romax = 0.95;
    double Tarrive = (id_opt == configCount - 1) ? TserGP[id_opt] : ((confOptS == 0) ? 1 / (p[id_opt] * lambdaOpt) : 1 / lambdaOpt);
    if (confOptS == 1) {
        romax = 0.95 * rhoG[id_opt];
    }

    double TserG = TserGP[id_opt];
    int c = 1;
    int NGP = 0;
    double roG = 0;
    double lambdaeGP = 0;
    while (roG < romax) {
        NGP++;
        auto [LqG, WqG, lambdaeGP_temp, prob_occupancyG, roG_temp] = MMcKKModel::calculateWaitTime(Tarrive, TserG, c, NGP);
        lambdaeGP = lambdaeGP_temp;
        roG = roG_temp;
    }
    if (debug) {
        std::clog << "Primary GPU path: M/M/1/NGP/NGP with NGP= " << NGP << " and lambdaeGP= " << lambdaeGP << std::endl;
    }

    romax = 0.95;
    int cP = 0;
    if (id_opt == configCount - 1) {
        Tarrive = TserCP[id_opt];
        cP = nc;
    } else {
        if (confOptS == 0) {
            Tarrive = 1 / lambdaOpt;
            romax = 0.95 * rhoC[id_opt];
        } else if (confOptS == 1) {
            Tarrive = 1 / (p[id_opt] * lambdaOpt);
        } else {
            Tarrive = 1 / lambdaOpt;
        }
        cP = std::max(static_cast<int>(round(nc * rhoC[id_opt])), 1);
    }

    double TserC = TserCP[id_opt];
    c = nc;
    int NCP = 0;
    double roC = 0;
    double lambdaeCP = 0;
    while (roC < romax) {
        NCP++;
        auto [LqC, WqC, lambdaeCP_temp, prob_occupancyC, roC_temp] = MMcKKModel::calculateWaitTime(Tarrive, TserC, c, NCP);
        lambdaeCP = lambdaeCP_temp;
        roC = roC_temp;
    }
    if (debug) {
        std::clog << "Primary CPU path: M/M/c/NCP/NCP with NCP= " << NCP << " and lambdaeCP= " << lambdaeCP << std::endl;
    }

    int NGS = 0, NCS = 0;
    double lambdaeGS = 0, lambdaeCS = 0;
    int cS = 0;
    if (confOptS == 1) {
        Tarrive = 1 / ((1 - p[id_opt]) * lambdaOpt);
        romax = 0.95 * (1 - rhoG[id_opt]);
        TserG = TserGS[id_opt];
        c = 1;
        roG = 0;
        while (roG < romax) {
            NGS++;
            auto [LqG, WqG, lambdaeGS_temp, prob_occupancyG, roG_temp] = MMcKKModel::calculateWaitTime(Tarrive, TserG, c, NGS);
            lambdaeGS = lambdaeGS_temp;
            roG = roG_temp;
        }
        if (debug) {
            std::clog << "Secondary GPU path: M/M/1/NGS/NGS with NGS= " << NGS << " and lambdaeGS= " << lambdaeGS << std::endl;
        }
    }

    if (confOptS == 0) {
        Tarrive = 1 / ((1 - p[id_opt]) * lambdaOpt);
        romax = 0.95 * (1 - rhoC[id_opt]);
        TserC = TserCS[id_opt];
        cS = nc - cP;
        c = nc;
        roC = 0;
        while (roC < romax) {
            NCS++;
            auto [LqC, WqC, lambdaeCS_temp, prob_occupancyC, roC_temp] = MMcKKModel::calculateWaitTime(Tarrive, TserC, c, NCS);
            if (cS == 0) {
                break;
            }
            lambdaeCS = lambdaeCS_temp;
            roC = roC_temp;
        }
        if (debug) {
            std::clog << "Secondary CPU path: M/M/c/NCS/NCS with NCS= " << NCS << " and lambdaeCS= " << lambdaeCS << std::endl;
        }
    }

    double lambdae = (id_opt == configCount - 1) ? lambdaeGP + lambdaeCP : std::min(lambdaOpt, std::min(lambdaeGP + lambdaeCS, lambdaeCP + lambdaeGS));
    int ntokens = NGP + NCP + NGS + NCS;

    return {lambdaOpt, confOptP, confOptS, lambdae, ntokens, cP, cS, lambdaeGP, lambdaeCP, lambdaeGS, lambdaeCS, NGP, NCP, NGS, NCS, stageBotl, devBot};
}
