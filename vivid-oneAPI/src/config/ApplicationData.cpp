#include "ApplicationData.hpp"

void ApplicationData::selectUSMQueue(sycl::queue &Q) {
    USM_queue = Q;
}

ApplicationData::~ApplicationData() {
    if (goldenFrame != nullptr) {
        delete[] goldenFrame;
    }
    if (filterBank != nullptr) {
        sycl::free(filterBank, USM_queue);
    }
}
