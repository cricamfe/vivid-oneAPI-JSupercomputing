#pragma once
#ifndef JSONFILE_HPP
#define JSONFILE_HPP

#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include "PipelineFactory.hpp"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <limits.h>
#include <map>
#include <random>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>
#if __cplusplus >= 202002L
#include <filesystem>
#else
#include <experimental/filesystem>
#endif

class JSONFile {
  public:
    void saveToFile(const std::string &filename);
    void loadFromFile(const std::string &filename);
    void setValue(const std::string &key, const nlohmann::json &value);
    nlohmann::json getValue(const std::string &key) const;
    std::pair<nlohmann::json, nlohmann::json> buildDataMap(const ApplicationData &appData, const InputArgs &inputArgs);
    void writeVariablesToJSON(const ApplicationData &appData, const InputArgs &inputArgs, int argc, char *argv[]);
    std::string prepareDirectory(const InputArgs &inputArgs, char *executable_path);

  private:
    nlohmann::json m_json;
    std::string generateRandomKey();
    std::string generateCommonKey(const ApplicationData &appData, const InputArgs &inputArgs);
};

#endif // JSONFILE_HPP
