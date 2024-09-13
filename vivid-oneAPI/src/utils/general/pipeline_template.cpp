/**
 * @file pipeline_template.cpp
 * @brief Contains the implementation of the class templates in the ViVid processing pipeline.
 *
 * This file contains the implementation of the Buffer_template, Item_template, and ViVidItem class templates,
 * which are used to manage buffers and items during the ViVid processing pipeline.
 */
#include "pipeline_template.hpp"

namespace Pipeline_template {
/**
 * @brief Construct a new ViVidItem with a common frame and common classification buffers.
 * @param global_frame A pointer to the frame buffer.
 * @param global_cla A pointer to the classification buffer.
 * @param num_filters The number of filters in the processing pipeline.
 * @param Q A SYCL queue object used for memory management.
 */
ViVidItem::ViVidItem(FloatBuffer *global_frame, FloatBuffer *global_cla, int num_filters, sycl::queue &Q) : frame{global_frame}, cla{global_cla}, ViVidItemQueue{Q} {
    ind = new FloatBuffer{global_frame->height, global_frame->width, BUF_READWRITE, ViVidItemQueue}; // create new buffers
    val = new FloatBuffer{global_frame->height, global_frame->width, BUF_READWRITE, ViVidItemQueue};
    his = new FloatBuffer{(global_frame->width / 8) * (global_frame->height / 8), static_cast<size_t>(num_filters), BUF_READWRITE, ViVidItemQueue};
    out = new FloatBuffer{global_cla->height, (global_frame->width / 8) * (global_frame->height / 8), BUF_READWRITE, ViVidItemQueue};

    for (auto &element : ptrSizeStage) {
        element = nullptr;
    }
}

/**
 * @brief Construct a new ViVidItem with a specific ID, global frame, and global classification buffer.
 * @param id The unique ID of the ViVidItem.
 * @param global_frame A pointer to the global frame buffer.
 * @param global_cla A pointer to the global classification buffer.
 * @param num_filters The number of filters in the processing pipeline.
 * @param Q A SYCL queue object used for memory management.
 */
ViVidItem::ViVidItem(size_t id, FloatBuffer *global_frame, FloatBuffer *global_cla, int num_filters, sycl::queue &Q) : item_id{id}, frame{global_frame}, cla{global_cla}, ViVidItemQueue{Q} {
    ind = new FloatBuffer{global_frame->height, global_frame->width, BUF_READWRITE, ViVidItemQueue}; // create new buffers
    val = new FloatBuffer{global_frame->height, global_frame->width, BUF_READWRITE, ViVidItemQueue};
    his = new FloatBuffer{(global_frame->width / 8) * (global_frame->height / 8), static_cast<size_t>(num_filters), BUF_READWRITE, ViVidItemQueue};
    out = new FloatBuffer{global_cla->height, (global_frame->width / 8) * (global_frame->height / 8), BUF_READWRITE, ViVidItemQueue};

    for (auto &element : ptrSizeStage) {
        element = nullptr;
    }
}

/**
 * @brief Destroy the ViVidItem object and free the memory allocated for the buffers.
 */
ViVidItem::~ViVidItem() {
    delete ind;
    delete val;
    delete his;
    delete out;
}

/**
 * @brief Recycles the ViVidItem object by clearing the buffers and the vector of events, setting the buffer to 0, pointers to the current stage to nullptr and setting the GPU_item flag to false.
 */
void ViVidItem::recycle() {
    // Este código es necesario cuando simulamos la ejecución de un kernel en un pipeline
    auto clear_buffer = [](FloatBuffer *buffer) {
        if (buffer && buffer->data && buffer->size > 0) {
            std::memset(buffer->data, 0, buffer->size);
        } /*else {
            if (VERBOSE_ENABLED)
                std::cerr << "Warning: Attempting to clear an invalid buffer. Data: " << buffer->data << ", Size: " << buffer->size << std::endl;
        }*/
    };
    // Clear buffers
    clear_buffer(ind);
    clear_buffer(val);
    clear_buffer(his);
    clear_buffer(out);

    // Clear vector of events
    stage_events.clear();

    // Clear vector of accelerators
    stage_acc.clear();

    // Clear the pointer to the current stage
    ptrSizeActualStage = nullptr;
    ptrCoreActualStage = nullptr;

    // Clear the pointers to the stages
    for (auto &element : ptrSizeStage) {
        element = nullptr;
    }

    // Reset time stages for CPU and GPU
    timeGPU_S = std::vector<double>(NUM_STAGES, 0.0);
    timeCPU_S = std::vector<double>(NUM_STAGES, 0.0);

    GPU_item = false;
}
} // namespace Pipeline_template