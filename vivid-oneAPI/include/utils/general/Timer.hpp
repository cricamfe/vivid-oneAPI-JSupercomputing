#pragma once
#ifndef TIMER_HPP
#define TIMER_HPP

#include "ApplicationData.hpp"
#include "GlobalParameters.hpp"
#include "InputArgs.hpp"
#include <chrono>
#include <iostream>
#include <thread>

void startTimerIfNeeded(ApplicationData &appData, InputArgs &inputArgs);
void startTimeMeasurement(ApplicationData &appData, InputArgs &inputArgs);

#endif // TIMER_HPP
