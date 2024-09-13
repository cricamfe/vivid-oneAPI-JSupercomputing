#ifndef COMPARE_HPP
#define COMPARE_HPP

#include "ApplicationData.hpp"
#include "pipeline_template.hpp"
#include <sycl/sycl.hpp>

namespace Comparer {
    void compare(ViVidItem *item, ApplicationData &appData);
    void createGoldenFrame(ApplicationData &appData);
} // namespace Comparer

#endif // COMPARE_HPP