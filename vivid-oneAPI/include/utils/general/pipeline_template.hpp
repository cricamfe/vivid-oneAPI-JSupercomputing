/**
 * @file pipeline_template.hpp
 * @brief Contains the class templates and their methods for the ViVid processing pipeline.
 *
 * This file contains the Buffer_template, Item_template, and ViVidItem class templates,
 * which are used to manage buffers and items during the ViVid processing pipeline.
 */

#pragma once
#ifndef PIPELINE_TEMPLATE_H
#define PIPELINE_TEMPLATE_H
/************************************************************************************
 *       -------------------          CLASS TEMPLATES            ----------------------
 *************************************************************************************/
#include "GlobalParameters.hpp"
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <oneapi/tbb.h>
#include <sycl/sycl.hpp>
#include <vector>

using namespace std;
using namespace oneapi;

namespace Pipeline_template {
/**************************
 *
 * BUFFER TEMPLATE
 *
 * *********************/
class ViVidItem; // forward declaration

// BUFFER STATE
#define BUF_ST_NONE 0     // NOT INITIALIZATED
#define BUF_ST_UNMAPPED 1 // IN GPU
#define BUF_ST_MAPPED 2   // IN CPU

// RESULT
#define BUF_OK 0     //
#define BUF_ERROR -1 //

// BUFFER LOCATION
#define BUF_GPU 1
#define BUF_CPU 0

// BUFFER ACCESS MODE
#define BUF_READ 1
#define BUF_WRITE 2
#define BUF_READWRITE 0
#define BUF_UNDEFINED -1

#define MAX_BUFFERS 10

/**
 * @class Buffer_template
 * @tparam A_Type The type of the elements stored in the buffer (e.g. float, int)
 * @brief A template class for managing buffers in the ViVid processing pipeline
 *
 * The Buffer_template class manages the memory allocation, access modes, and other properties
 * of buffers used in the ViVid processing pipeline. The class provides methods to set and query
 * buffer properties, allocate and free memory, and get the host pointer for the buffer.
 */
template <typename A_Type>
class Buffer_template {
  public:
    static bool ZCB;       ///< Zero-copy buffer flag.
    static bool use_pitch; ///< Flag for using device pitch.

    int buffer_state = BUF_ST_NONE;

    size_t width;
    size_t height;
    size_t Ne; // number of elements

    size_t pitch; // size in bytes of the rows including padding
    size_t size;  // meassured in bytes

    bool dirty = false;
    bool d_dirty = false;
    int kernelaccess;
    int mapaccess = BUF_UNDEFINED;

    A_Type *data = nullptr;

    sycl::queue bufferTemplateQueue; // queue for USM allocation

    /**
     * @brief Constructor with width and height.
     * @param h The height of the buffer.
     * @param w The width of the buffer.
     * @param access Access mode for the buffer.
     * @param queue SYCL queue for USM allocation.
     */
    Buffer_template(size_t h, size_t w, int access, sycl::queue &queue);
    /**
     * @brief Constructor with width only.
     * @param w The width of the buffer.
     * @param access Access mode for the buffer.
     * @param queue SYCL queue for USM allocation.
     */
    Buffer_template(size_t w, int access, sycl::queue &queue);

    /**
     * @brief Copy constructor.
     * @param global Pointer to another Buffer_template object to copy from.
     * @param access Access mode for the buffer.
     * @param queue SYCL queue for USM allocation.
     */
    Buffer_template(Buffer_template *global, int access, sycl::queue &queue);

    /**
     * @brief Destructor for the Buffer_template class.
     */
    ~Buffer_template();

    /**
     * @brief Reuse the buffer without changing the zero-copy state.
     */
    void reuse_ZCB();

    /**
     * @brief Get the host pointer to the buffer data.
     * @param access Access mode for the buffer.
     * @return Pointer to the buffer data.
     */
    A_Type *get_HOST_PTR(int access);

    /**
     * @brief Flush the buffer data from the CPU.
     */
    void flush_from_CPU();

    /**
     * @brief Set the zero-copy buffer flag.
     * @param s The new value for the zero-copy buffer flag.
     */
    static void set_ZCB(bool s);

    /**
     * @brief Set the device pitch flag.
     * @param s The new value for the device pitch flag.
     */
    static void set_device_pitch(bool s);

  private:
    /**
     * @brief Set the pitch for the buffer.
     */
    void set_pitch();

    /**
     * @brief Allocate host Unified Shared Memory (USM).
     */
    void alloc_host_USM();

    /**
     * @brief Free host Unified Shared Memory (USM).
     */
    void free_host_USM();
};

template <class A_Type>
bool Buffer_template<A_Type>::ZCB = false;
template <class A_Type>
bool Buffer_template<A_Type>::use_pitch = false;

/*****************
 *  Typedefs
 * ****************/

using FloatBuffer = Buffer_template<float>;
using IntBuffer = Buffer_template<int>;

/**************************
 *
 * BUFFER TEMPLATE METHODS
 *
 * *********************/
template <typename A_Type>
void Buffer_template<A_Type>::reuse_ZCB() {
    d_dirty = false;
    dirty = false;
}
//---------------------------------------------------------
template <typename A_Type>
Buffer_template<A_Type>::Buffer_template(size_t h, size_t w, int access, sycl::queue &queue) : width{w}, height{h}, Ne{w * h}, kernelaccess{access}, bufferTemplateQueue{queue} {
    set_pitch();
}
//---------------------------------------------------------
template <typename A_Type>
Buffer_template<A_Type>::Buffer_template(size_t w, int access, sycl::queue &queue) : width{w}, height{1}, Ne{w}, kernelaccess{access}, bufferTemplateQueue{queue} {
    set_pitch();
}
//---------------------------------------------------------
template <typename A_Type>
Buffer_template<A_Type>::Buffer_template(Buffer_template *global, int access, sycl::queue &queue) : // does it make sense?
                                                                                                    width{global->width}, height{global->height}, Ne{global->Ne}, pitch{global->pitch}, size{global->size}, kernelaccess{access}, bufferTemplateQueue{queue} {}

//---------------------------------------------------------
template <typename A_Type>
Buffer_template<A_Type>::~Buffer_template() {
    if (!ZCB && data != NULL)
        free_host_USM();
}
//---------------------------------------------------------
template <typename A_Type>
void Buffer_template<A_Type>::set_ZCB(bool s) {
    ZCB = s;
}
//---------------------------------------------------------
template <typename A_Type>
void Buffer_template<A_Type>::set_device_pitch(bool s) {
    use_pitch = s;
}

//---------------------------------------------------------
template <typename A_Type>
void Buffer_template<A_Type>::flush_from_CPU() {
    if (buffer_state == BUF_ST_MAPPED) {
        this->unmap();
        this->buffer_state = BUF_ST_UNMAPPED;
    }
    dirty = false;
}
//---------------------------------------------------------
template <typename A_Type>
void Buffer_template<A_Type>::set_pitch() {
    /*The optimal pitch is computed by (1) getting the base address alignment
    preference for your card (CL_DEVICE_MEM_BASE_ADDR_ALIGN property with
    clGetDeviceInfo: note that the returned value is in bits, so you have
    to divide by 8 to get it in bytes);*/
    if (use_pitch) {

        int buffer;
        // cl_int prueba = clGetDeviceInfo(device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN , sizeof(buffer), &buffer, NULL);
        // buffer /= 8;
        buffer = 64;

        /*let's call this base (2) find the largest multiple of base
        that is no less than your natural
        data pitch (sizeof(type) times number of columns);*/

        pitch = ceil(float(sizeof(A_Type) * width) / buffer) * buffer;

        // You then allocate pitch times number of rows bytes, and pass the pitch information to kernels.

        size = height * pitch; // height + 16 why ???  Andrés erased it
    } else {
        pitch = sizeof(A_Type) * width;
        size = height * pitch;
    }
}

//---------------------------------------------------------
template <typename A_Type>
A_Type *Buffer_template<A_Type>::get_HOST_PTR(int access) {
    if (data == NULL)
        alloc_host_USM();
    d_dirty = false;

    dirty = dirty || (access == BUF_WRITE || access == BUF_READWRITE);
    return data;
}

//---------------------------------------------------------
template <typename A_Type>
void Buffer_template<A_Type>::alloc_host_USM() {
    data = sycl::malloc_shared<A_Type>(size / sizeof(A_Type), bufferTemplateQueue);
    if (data == NULL) {
        printf("Error I can't malloc host buffer. size: %zu\n", size);
        exit(0);
    }
}

//---------------------------------------------------------
template <typename A_Type>
void Buffer_template<A_Type>::free_host_USM() {
    if (data != NULL)
        sycl::free(data, bufferTemplateQueue);
}

/**
 * @class Item_template
 * @brief A base class template for managing items in the ViVid processing pipeline.
 *
 * The Item_template class should be used as a base class for items in the processing pipeline.
 */
class Item_template {
  public:
    bool sticked;
    size_t num_frame;
    std::vector<bool> usedGPU;
    Item_template() : sticked{false} {}
    Item_template(size_t n) : sticked{false}, num_frame{n} {}
    void set_num_item(size_t n) {
        num_frame = n;
    }
};

/**
 * @class ViVidItem
 * @brief A derived class template from Item_template for managing ViVid items in the processing pipeline.
 *
 * The ViVidItem class extends the Item_template class and manages additional properties specific
 * to the ViVid processing pipeline. This class manages FloatBuffer objects for input frames, indices,
 * values, histograms, classification, and output. It also handles the recycling of items in the pipeline.
 */
class ViVidItem : public Item_template {
  public:
    size_t item_id = 0;          //< The item ID
    bool GPU_item = false;       //< The item has been processed on GPU only.
    std::stringstream traceItem; //< The trace of the item.

    std::atomic<int> *ptrSizeActualStage = nullptr; //< Atomic pointer of integer type pointing to the current stage size.
    std::atomic<int> *ptrCoreActualStage = nullptr; //< Atomic pointer of integer type pointing to the current stage size.

    std::array<std::atomic<int> *, 3> ptrSizeStage; //< Array of atomic pointers of type integer pointing to the size of the stages.
    std::array<std::atomic<int> *, 3> ptrCoreStage; //< Array of atomic pointers of type integer pointing to the core of the stages.

    std::vector<sycl::event> stage_events; //< Create a vector of events to wait for the previous stage or save the event of the current stage.
    std::vector<Acc> stage_acc;            //< Create a vector of accelerators to save the accelerator used in each stage.
    sycl::queue &ViVidItemQueue;           //< Reference to the queue we use to allocate memory with malloc_shared

    // Variables for time measurement
    tbb::tick_count filter_start;           //< The filter used to start the timer
    std::vector<double> timeCPU_S{0, 0, 0}; //< The CPU execution time for each stage
    std::vector<double> timeGPU_S{0, 0, 0}; //< The GPU execution time for each stage
    double execution_time = 0;              //< The total execution time of the kernel

    // Buffers used in the ViVid pipeline
    FloatBuffer *frame; // Input                    //< The input frame buffer
    FloatBuffer *ind;   // F1                       //< The indices buffer
    FloatBuffer *val;   // F1                       //< The values buffer
    FloatBuffer *his;   // F2                       //< The histogram buffer
    FloatBuffer *cla;   // F2                       //< The classification buffer
    FloatBuffer *out;   // F3                       //< The output buffer

    ViVidItem(size_t id, FloatBuffer *global_frame, FloatBuffer *global_cla, int num_filters, sycl::queue &Q);
    ViVidItem(FloatBuffer *global_frame, FloatBuffer *global_cla, int num_filters, sycl::queue &Q);
    ~ViVidItem();
    void recycle();
};

} // namespace Pipeline_template

#endif