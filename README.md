# ViVid Project

  

The ViVid project aims to improve the implementation and optimization of algorithms for handling massive data flows, using oneAPI in heterogeneous architectures, mainly with CPU+GPU, and eventually FPGA. The methodology consists of selecting appropriate algorithms, developing code and dependencies using oneAPI, identifying and solving performance bottlenecks, and maximizing parallelization. Our goal is to achieve significant advances in real-time processing of large volumes of data, efficiently leveraging the performance of state-of-the-art heterogeneous architectures.

  

## Table of Contents

  

- [Project Overview](#project-overview)

- [Building and Running](#building-and-running)

- [Build Configuration Flags](#build-configuration-flags)

- [Command Line Arguments](#command-line-arguments)

- [Previous Versions](#previous-versions)

  

## Project Overview

  

The current version of ViVid provides several implementations for image processing using different parallel programming models:

  

1.  **Serie Pipeline (serie)**: A serial implementation that processes images sequentially, supporting either CPU-only or GPU-only execution.

  

2.  **Parallel Pipeline (pipeline)**: Uses oneTBB's parallel_pipeline to create a heterogeneous pipeline that efficiently processes images using both CPU and GPU resources.

  

3.  **Flow Graph with Functional Nodes (fgfn)**: Implements the pipeline using oneTBB's Flow Graph with functional nodes, offering more flexibility in task execution.

  

4.  **Flow Graph with Async Nodes (fgan)**: Similar to fgfn but uses asynchronous nodes for GPU operations, freeing up oneTBB worker threads during GPU processing.

  

5.  **SYCL Events (syclevents)**: A heterogeneous pipeline implementation using SYCL events, providing direct control over task execution and synchronization without relying on oneTBB.

  

## Building and Running

  

### Prerequisites

- Intel oneAPI Base Toolkit

- Intel oneAPI HPC Toolkit

- A compatible C++ compiler (supporting C++20)

- CMake (3.x or higher)

  

### Building

  

```bash
cd  vivid-oneAPI
make
```

  

## Build Configuration Flags

  

The Makefile supports various configuration flags that can be enabled during compilation:

  

### Feature Flags

  

-  `DEBUG=1`: Enable debug mode

-  `VERBOSE=1`: Enable verbose output

-  `TIMESTAGES=1`: Enable time measurement for stages

-  `QUEUE=1`: Enable queue order tracking

-  `TRACE=1`: Enable execution tracing

-  `CSV=1`: Enable CSV output

-  `JSON=1`: Enable JSON output

-  `AUTO=1`: Enable automatic mode

-  `ADVANCEDMETRICS=1`: Enable advanced metrics collection

-  `LOG=1`: Enable logging

-  `NOQUEUE=1`: Disable queue usage

-  `NUMSTAGES=N`: Set number of stages

-  `ENERGYPCM=1`: Enable energy measurements with PCM

-  `LIMCORES=N`: Set core limit

-  `OLD_COMPILER=1`: Use older version of the Intel oneAPI compiler (2023.2.0)

  

### Kernel Optimization Settings

  

Configure pairwise distance calculation kernel with `PWDIST=N`:

-  `0`: Default (optimized for both CPU and GPU)

-  `1`: Optimized using sycl::float type

-  `2`: Optimized using sycl::float4 type

-  `3`: Unoptimized version

  

### Backend Settings

  

Select SYCL backend for GPU queue with `BACKEND=N`:

-  `0`: OpenCL (default)

-  `1`: Level Zero

-  `2`: CUDA (automatically detects and uses correct compute capability)

  

### Acquisition Mode Settings
Configure resource acquisition strategy with `ACQMODE=N`:

-  `0`: DEFAULT - Try to queue in preferred device cores first, then alternate device, following same order for queues

-  `1`: PRIMARY_SECONDARY - Try to acquire cores and queue from primary device first, then secondary device

-  `2`: NO_QUEUE - Only try to acquire cores without using queues

  

### Example Usage

  

```bash
# Build with debug and verbose output enabled
make  DEBUG=1  VERBOSE=1
# Build with CUDA backend and advanced metrics
make  BACKEND=2  ADVANCEDMETRICS=1
# Build with energy measurements and specific kernel optimization
make  ENERGYPCM=1  PWDIST=2
```

  

## Command Line Arguments
The application supports various command line arguments to configure its behavior:

### Required Arguments
 
-  `--api <type>`: Pipeline implementation to use [Required]

- Options: `serie`, `pipeline`, `fgfn`, `fgan`, `syclevents`, `taskflow`

### Optional Arguments

-  `--numframes <N>`: Number of frames to process

-  `--resolution <N>`: Image resolution [0-5]

	- 0: 1280x720 (720p)

	- 1: 1920x1080 (1080p)

	- 2: 2560x1440 (1440p)

	- 3: 3840x2160 (2160p)

	- 4: 5120x2880 (2880p)

	- 5: 7680x4320 (4320p)

-  `--duration <time>`: Duration of execution (format: XhYmZs, e.g., 1h30m)

-  `--threads <N>`: Number of CPU threads to use

-  `--iff <N>`: Number of frames in flight

-  `--config <string>`: Stage configuration

	- For **non-serie APIs**:

		- "CPU": All stages on CPU

		- "GPU": All stages on GPU

		- "DECOUPLED": Both CPU and GPU enabled with decoupled path

	- Custom string (e.g., "012"): **Per-stage configuration** where:

		- 0: CPU only

		- 1: CPU+GPU

		- 2: GPU only

	- For **serie API**:

		- Only "CPU" or "GPU" allowed

-  `--buffersize <N>`: Size of the circular buffer

-  `--sizegpu <N>`: Size of GPU queue (1 value or per stage)

-  `--sizecpu <N>`: Size of CPU queue (1 value or per stage)

-  `--corescpu <N>`: CPU cores per stage (1 value or per stage)

-  `--coresgpu <N>`: GPU cores per stage (1 value or per stage)

-  `--prefdevice <N>`: Preferred device per stage (0:CPU, 2:GPU)

-  `--dependson`: Enable SYCL event dependencies (serie API only)

-  `--thcpu <N>`: CPU throughput per stage

-  `--thgpu <N>`: GPU throughput per stage

  

### Examples

  

1. Run parallel pipeline with CPU only:

```bash
./main  --api  pipeline  --numframes  1000  --resolution  1  --threads  8  --config  CPU
```

  

2. Run SYCL events version with GPU only for 1 hour:

```bash
./main  --api  syclevents  --duration  1h  --resolution  2  --config  GPU
```

  

3. Run flow graph with custom stage configuration:
```bash
./main  --api  fgfn  --numframes  500  --resolution  1  --threads  4  --config  012
```