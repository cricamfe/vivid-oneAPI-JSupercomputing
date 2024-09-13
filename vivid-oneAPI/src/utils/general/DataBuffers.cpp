#include "DataBuffers.hpp"

FloatBuffer *DataBuffers::createGlobalFrame(const std::unique_ptr<float[]> &f_imData, int height, int width, sycl::queue &Q) {
    FloatBuffer::set_ZCB(false);
    FloatBuffer::set_device_pitch(false);

    FloatBuffer *global_frame = new FloatBuffer(height, width, BUF_READ, Q);
    float *punt = global_frame->get_HOST_PTR(BUF_WRITE);
    memcpy(punt, f_imData.get(), global_frame->size);

    return global_frame;
}

float *DataBuffers::createFilterBank(const int num_filters, const int filter_dim, std::mt19937 &mte, sycl::queue &Q) {
    std::uniform_real_distribution<float> uniform_filter_bank{0.00000001, 0.00000099};

    float *filter_bank = sycl::malloc_shared<float>(num_filters * filter_dim * filter_dim, Q);
    for (int i = 0; i < num_filters * filter_dim * filter_dim; i++) {
        filter_bank[i] = uniform_filter_bank(mte);
    }

    return filter_bank;
}

void DataBuffers::createAllBuffers(ApplicationData &appData, const std::unique_ptr<float[]> &f_imData) {
    // Configure the buffers and copy the image to the buffer
    if constexpr (VERBOSE_ENABLED) {
        printf(" Configuring the buffers...\n");
        printf(" Image size: %d x %d\n", appData.height, appData.width);
    }
    appData.globalFrame = createGlobalFrame(f_imData, appData.height, appData.width, appData.USM_queue);

    // Create a random filter bank (filter_dim = 3)
    if constexpr (VERBOSE_ENABLED) {
        printf(" Creating the filter bank...\n");
    };
    appData.filterBank = createFilterBank(appData.numFilters, appData.filterDim, appData.mte, appData.USM_queue);

    // Create a random coefficients
    if constexpr (VERBOSE_ENABLED) {
        printf(" Creating the coefficients...\n");
    };
    appData.globalCla = createGlobalCla(appData.window_height, appData.windowWidth, appData.cellSize, appData.blockSize, appData.dictSize, appData.mte, appData.USM_queue);
}

FloatBuffer *DataBuffers::createGlobalCla(const int window_height, const int window_width, const int cell_size, const int block_size, const int dict_size, std::mt19937 &mte, sycl::queue &Q) {
    std::uniform_real_distribution<float> uniform_coefficients{0.05, 0.099};

    int n_cells_x = window_width / cell_size;
    int n_cells_y = window_height / cell_size;
    int n_blocks_x = n_cells_x - block_size + 1;
    int n_blocks_y = n_cells_y - block_size + 1;

    size_t n_total_coeff = block_size * block_size * n_blocks_x * n_blocks_y * dict_size;

    FloatBuffer *global_cla = new FloatBuffer{n_total_coeff / dict_size, static_cast<size_t>(dict_size), BUF_READ, Q};
    float *coefficients = global_cla->get_HOST_PTR(BUF_WRITE);

    for (int i = 0; i < dict_size; i++) {
        for (size_t j = 0; j < n_total_coeff / dict_size; j++) {
            coefficients[j * global_cla->pitch / sizeof(float) + i] = uniform_coefficients(mte);
        }
    }

    return global_cla;
}