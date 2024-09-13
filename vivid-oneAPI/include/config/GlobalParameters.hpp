#pragma once
#ifndef CONFIG_TEMPLATE_H
#define CONFIG_TEMPLATE_H

#define DEFAULT_NUM_FRAMES 1000        //< Default number of frames to process
#define DEFAULT_IMAGE_RESOLUTION 1     //< Default image resolution (1: 1080p, 2: 1440p, 3: 2160p, 4: 2880p, 5: 4320p)
#define DEFAULT_NUM_THREADS 8          //< Default number of threads
#define DEFAULT_CONFIG_STAGES "000"    //< Default configuration of the stages
#define DEFAULT_SIZE_CIRCULAR_BUFFER 4 //< Default size of the circular buffer
#define DEFAULT_CORES_GPU 1            //< Default number of cores to use in the GPU
#define DEFAULT_SIZE_GPU 3             //< Default size of the GPU

#define MAX_NFRAMES_QUEUE 100          //< Maximum number of frames to process in the optimization if use duration
#define PER_FRAMES_TO_PROCESS_BAS 0.15 //< C++ and SYCL measure 10% of the frames
#define PER_FRAMES_TO_PROCESS_VEC 0.3  //< AVX and SIMD measure 20% of the frames
#define MIN_SPEEDUP 0.8                //< Minimum speedup to consider a Stage as accelerated

// Define Accelerator
enum class Acc {
    CPU = 0,
    GPU = 1,
    FPGA = 2,
    OTHER = 3
};

enum class PipelineType {
    Serie = 0,
    ParallelPipeline = 1,
    FlowGraphFunctionalNode = 2,
    FlowGraphAsyncNode = 3,
    SYCLEvents = 4,
    Taskflow = 5
};

enum class PathSelection {
    Coupled,
    Decoupled,
    CoupledCustom
};

enum class StageState {
    CPU = 0,
    CPU_GPU = 1,
    GPU = 2,
};

#ifdef __SYCL__
#define SYCL_ENABLED 1
#else
#define SYCL_ENABLED 0
#endif

#ifdef __MAVX__
#define AVX_ENABLED 1
#else
#define AVX_ENABLED 0
#endif

#ifdef __SIMD__
#define SIMD_ENABLED 1
#else
#define SIMD_ENABLED 0
#endif

#ifdef __DEBUG__
#define DEBUG_ENABLED 1
#else
#define DEBUG_ENABLED 0
#endif

#ifdef __VERBOSE__
#define VERBOSE_ENABLED 1
#else
#define VERBOSE_ENABLED 0
#endif

#ifdef __LOG__
#define LOG_ENABLED 1
#else
#define LOG_ENABLED 0
#endif

#ifdef __TRACE__
#define TRACE_ENABLED 1
#else
#define TRACE_ENABLED 0
#endif

#ifdef __CSV__
#define CSV_ENABLED 1
#else
#define CSV_ENABLED 0
#endif

#ifdef __JSON__
#define JSON_ENABLED 1
#else
#define JSON_ENABLED 0
#endif

#ifdef __TIMESTAGES__
#define TIMESTAGES_ENABLED 1
#else
#define TIMESTAGES_ENABLED 0
#endif

#ifdef __AUTO__
#define AUTOMODE_ENABLED 1
#else
#define AUTOMODE_ENABLED 0
#endif

#ifdef __ADVANCEDMETRICS__
#define ADVANCEDMETRICS_ENABLED 1
#else
#define ADVANCEDMETRICS_ENABLED 0
#endif

#ifdef __OLDCOMPILER__
#define USING_OLDCOMPILER 1
#else
#define USING_OLDCOMPILER 0
#endif

#ifdef __ENERGYPCM__
#define ENERGYPCM_ENABLED 1
#else
#define ENERGYPCM_ENABLED 0
#endif

#ifdef __NOQUEUE__
#define DEVICE_QUEUE_ENABLED 0
#else
#define DEVICE_QUEUE_ENABLED 1
#endif

#ifndef __ACQMODE__
#define __ACQMODE__ 0
#endif

#ifdef __QUEUE__
#define INORDER_QUEUE 1
#else
#define INORDER_QUEUE 0
#endif

#ifdef __LIMCORES__
#define LIMITCORES_ENABLED 1
#else
#define LIMITCORES_ENABLED 0
#endif

#ifndef __BACKEND__
#define __BACKEND__ 0
#endif

#ifndef __NODEPRIORITY__
#define __NODEPRIORITY__ 0
#endif

#ifndef __NUMSTAGES__
#define USE_VIVID_APP 1
#define NUM_STAGES 3
#else
#define USE_VIVID_APP 0
#define NUM_STAGES __NUMSTAGES__
#endif

#ifndef __PWDIST__
#define __PWDIST__ 0
#endif

#endif