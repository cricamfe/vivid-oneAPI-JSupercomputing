#include "Comparer.hpp"
#include "GlobalParameters.hpp"
#include "filters-CPP.hpp"

using namespace Comparer;

void Comparer::compare(ViVidItem *item, ApplicationData &appData) {
    int resultSize = item->out->Ne; // Number of elements in the result
    bool error = false;             // Error flag
    float tolerance = 10E-2;        // Tolerance for the comparison
    int max_print = 5;              // Number of values to print in case of error
    int index_error = 0;            // Index of the first error

    // Check all the values
    for (int j = 0; j < resultSize; j++) {
        float vabs = sycl::fabs(appData.goldenFrame[j] - item->out->data[j]);
        if (sycl::isnotequal(appData.goldenFrame[j], item->out->data[j]) && vabs >= tolerance) {
            if constexpr (VERBOSE_ENABLED)
                std::cout << "\tERROR (res = " << item->out->data[j] << " & ref = " << appData.goldenFrame[j] << ")" << std::endl;
            error = true;
            index_error = j;
            break;
        }
    }
    if (error) {
        std::cout << "ERROR: The result of item " << item->item_id << " on index " << index_error << " is not correct!" << std::endl;
    }
    if constexpr (VERBOSE_ENABLED) {
        for (int j = 0; j < max_print; j++) {
            std::cout << "\t" << item->out->data[j] << " ";
        }
        std::cout << std::endl;
    }
}

void Comparer::createGoldenFrame(ApplicationData &appData) {
    ViVidItem *item_dbg = appData.item_debug;
    sycl::queue Q_GPU = appData.USM_queue;
    if constexpr (VERBOSE_ENABLED) {
        printf(" Calculating the reference output...\n");
    }
    item_dbg = new ViVidItem{appData.globalFrame, appData.globalCla, appData.numFilters, Q_GPU};
    if constexpr (VERBOSE_ENABLED)
        printf(" Start of reference output calculation...\n");
    if constexpr (VERBOSE_ENABLED)
        printf("  - Filter 1...\n");
    cosine_filter_transpose(item_dbg->frame->get_HOST_PTR(BUF_READ), item_dbg->ind->get_HOST_PTR(BUF_WRITE), item_dbg->val->get_HOST_PTR(BUF_WRITE), appData.filterBank, appData.height, appData.width, appData.filterDim, appData.filterDim, appData.numFilters, item_dbg->val->pitch);
    if constexpr (VERBOSE_ENABLED)
        printf("  - Filter 2...\n");
    block_histogram(item_dbg->his->get_HOST_PTR(BUF_WRITE), item_dbg->ind->get_HOST_PTR(BUF_READ), item_dbg->val->get_HOST_PTR(BUF_READ), appData.cellSize, appData.height, appData.width, item_dbg->his->pitch / sizeof(float), item_dbg->ind->pitch / sizeof(float), item_dbg->val->pitch / sizeof(float));
    if constexpr (VERBOSE_ENABLED)
        printf("  - Filter 3...\n");
    pwdist_c(item_dbg->cla->get_HOST_PTR(BUF_READ), item_dbg->his->get_HOST_PTR(BUF_READ), item_dbg->out->get_HOST_PTR(BUF_WRITE), item_dbg->out->pitch / sizeof(float), item_dbg->cla->height, item_dbg->cla->pitch / sizeof(float), item_dbg->his->height, item_dbg->cla->width);
    if constexpr (VERBOSE_ENABLED)
        printf(" End of reference output calculation\n");
    int resultSize = item_dbg->out->Ne;
    appData.goldenFrame = (float *)malloc(resultSize * sizeof(float));
    memcpy(appData.goldenFrame, item_dbg->out->data, resultSize * sizeof(float)); // Copy the result to the golden array
    std::cout << " Golden (first values...): \n\t";
    for (int j = 0; j < 10; j++)
        std::cout << appData.goldenFrame[j] << " ";
    std::cout << "\n End of golden: \n\n";
}