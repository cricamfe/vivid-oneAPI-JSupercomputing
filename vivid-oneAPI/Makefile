# Check for OLD_COMPILER=1 to use the old compiler versio (new version have some issues)
ifeq ($(OLD_COMPILER), 1)
    CXX := /opt/intel/oneapi/compiler/2023.2.0/linux/bin/icpx -fsycl
    DPCFLAGS := -std=c++20 -O3 -qtbb -pthread -L/opt/intel/oneapi/compiler/2023.2.0/linux/lib
    export LD_LIBRARY_PATH := /opt/intel/oneapi/compiler/2023.2.0/linux/lib:$(LD_LIBRARY_PATH)
else
    CXX := icpx -fsycl
    DPCFLAGS := -std=c++20 -O3 -qtbb -pthread
endif

# Get current directory
CURRENT_DIR := $(patsubst %/,%,$(dir $(realpath $(lastword $(MAKEFILE_LIST)))))
PARENT_DIR := $(abspath $(CURRENT_DIR)/..)

# Directories
SRC_DIR := $(CURRENT_DIR)/src
INCLUDE_DIR := $(CURRENT_DIR)/include
BIN_DIR := $(CURRENT_DIR)/bin
ENERDIR := $(PARENT_DIR)/pcm/src

CONFIG_SRC_DIR := $(SRC_DIR)/config
UTILS_GENERAL_SRC_DIR := $(SRC_DIR)/utils/general
UTILS_SPECIFIC_SRC_DIR := $(SRC_DIR)/utils/specific
UTILS_MANAGER_SRC_DIR := $(SRC_DIR)/utils/manager
FILTERS_SRC_DIR := $(SRC_DIR)/filters
PIPELINE_SRC_DIR := $(SRC_DIR)/pipeline
EXECUTORS_SRC_DIR := $(SRC_DIR)/executors
EXPORT_SRC_DIR := $(SRC_DIR)/export
QUEUE_SRC_DIR := $(SRC_DIR)/queue
ENERGY_SRC_DIR := $(SRC_DIR)/utils/energy

BIN_CONFIG_DIR := $(BIN_DIR)/config
BIN_UTILS_GENERAL_DIR := $(BIN_DIR)/utils/general
BIN_UTILS_SPECIFIC_DIR := $(BIN_DIR)/utils/specific
BIN_UTILS_MANAGER_DIR := $(BIN_DIR)/utils/manager
BIN_FILTERS_DIR := $(BIN_DIR)/filters
BIN_PIPELINE_DIR := $(BIN_DIR)/pipeline
BIN_EXECUTORS_DIR := $(BIN_DIR)/executors
BIN_EXPORT_DIR := $(BIN_DIR)/export
BIN_QUEUE_DIR := $(BIN_DIR)/queue
BIN_ENERGY_DIR := $(BIN_DIR)/energy

# Create 'bin' directory and necessary subdirectories if they do not exist
$(shell mkdir -p $(BIN_CONFIG_DIR))
$(shell mkdir -p $(BIN_UTILS_GENERAL_DIR))
$(shell mkdir -p $(BIN_UTILS_SPECIFIC_DIR))
$(shell mkdir -p $(BIN_UTILS_MANAGER_DIR))
$(shell mkdir -p $(BIN_EXECUTORS_DIR))
$(shell mkdir -p $(BIN_PIPELINE_DIR))
$(shell mkdir -p $(BIN_FILTERS_DIR))
$(shell mkdir -p $(BIN_EXPORT_DIR))
$(shell mkdir -p $(BIN_QUEUE_DIR))
$(shell mkdir -p $(BIN_ENERGY_DIR))

# Define include directories
INCLUDES := -I$(INCLUDE_DIR) \
            -I$(INCLUDE_DIR)/config \
            -I$(INCLUDE_DIR)/utils/general \
            -I$(INCLUDE_DIR)/utils/specific \
			-I$(INCLUDE_DIR)/utils/manager \
            -I$(INCLUDE_DIR)/filters \
            -I$(INCLUDE_DIR)/pipeline \
            -I$(INCLUDE_DIR)/executors \
			-I$(INCLUDE_DIR)/export \
			-I$(INCLUDE_DIR)/queue \
			-I$(PARENT_DIR)/taskflow

# Source files
CONFIG_SRC := $(wildcard $(CONFIG_SRC_DIR)/*.cpp)
PIPELINE_SRC := $(wildcard $(PIPELINE_SRC_DIR)/*.cpp)
EXECUTORS_SRC := $(wildcard $(EXECUTORS_SRC_DIR)/*.cpp)
COMMON_FILTERS_SRC := $(wildcard $(FILTERS_SRC_DIR)/filters-CPP.cpp $(FILTERS_SRC_DIR)/filters-SYCL.cpp $(FILTERS_SRC_DIR)/WorkloadSimulator.cpp)
UTILS_GENERAL_SRC := $(wildcard $(UTILS_GENERAL_SRC_DIR)/*.cpp)
UTILS_SPECIFIC_SRC := $(wildcard $(UTILS_SPECIFIC_SRC_DIR)/*.cpp)
UTILS_MANAGER_SRC := $(wildcard $(UTILS_MANAGER_SRC_DIR)/*.cpp)
EXPORT_SRC := $(wildcard $(EXPORT_SRC_DIR)/*.cpp)
QUEUE_SRC := $(wildcard $(QUEUE_SRC_DIR)/*.cpp)
MAIN_SRC := $(SRC_DIR)/main.cpp

# Extra sources for AVX, SIMD, SYCL
EXTRA_SRC :=

# -D__SYCL__ : Enables SYCL mode, which uses SYCL to execute the pipeline on the CPU.
ifeq ($(SYCL), 1)
    CXX += -xHost
    DPCFLAGS += -D__SYCL__
endif

# -D__AVX__ : Enables AVX mode, which uses AVX instructions to execute the pipeline on the CPU.
ifeq ($(AVX), 1)
    CXX += -xHost
    DPCFLAGS += -D__MAVX__
    EXTRA_SRC += $(FILTERS_SRC_DIR)/filters-AVX.cpp
endif

# -D__SIMD__ : Enables SIMD mode, which uses SIMD instructions to execute the pipeline on the CPU.
ifeq ($(SIMD), 1)
    CXX += -xHost
    DPCFLAGS += -D__SIMD__
    EXTRA_SRC += $(FILTERS_SRC_DIR)/filters-SIMD.cpp
endif


# Combine all source files
OBJ_FILES := $(patsubst $(CONFIG_SRC_DIR)/%.cpp, $(BIN_CONFIG_DIR)/%.o, $(CONFIG_SRC)) \
             $(patsubst $(PIPELINE_SRC_DIR)/%.cpp, $(BIN_PIPELINE_DIR)/%.o, $(PIPELINE_SRC)) \
             $(patsubst $(EXECUTORS_SRC_DIR)/%.cpp, $(BIN_EXECUTORS_DIR)/%.o, $(EXECUTORS_SRC)) \
             $(patsubst $(FILTERS_SRC_DIR)/%.cpp, $(BIN_FILTERS_DIR)/%.o, $(COMMON_FILTERS_SRC)) \
             $(patsubst $(FILTERS_SRC_DIR)/%.cpp, $(BIN_FILTERS_DIR)/%.o, $(EXTRA_SRC)) \
             $(patsubst $(UTILS_GENERAL_SRC_DIR)/%.cpp, $(BIN_UTILS_GENERAL_DIR)/%.o, $(UTILS_GENERAL_SRC)) \
             $(patsubst $(UTILS_SPECIFIC_SRC_DIR)/%.cpp, $(BIN_UTILS_SPECIFIC_DIR)/%.o, $(UTILS_SPECIFIC_SRC)) \
			 $(patsubst $(UTILS_MANAGER_SRC_DIR)/%.cpp, $(BIN_UTILS_MANAGER_DIR)/%.o, $(UTILS_MANAGER_SRC)) \
			 $(patsubst $(EXPORT_SRC_DIR)/%.cpp, $(BIN_EXPORT_DIR)/%.o, $(EXPORT_SRC)) \
			 $(patsubst $(QUEUE_SRC_DIR)/%.cpp, $(BIN_QUEUE_DIR)/%.o, $(QUEUE_SRC)) \
             $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%.o, $(MAIN_SRC))

# Flags for optional features
DEBUG_FLAGS := $(if $(filter 1,$(DEBUG)),-D__DEBUG__)
VERBOSE_FLAGS := $(if $(filter 1,$(VERBOSE)),-D__VERBOSE__)
TIMESTAGES_FLAGS := $(if $(filter 1,$(TIMESTAGES)),-D__TIMESTAGES__)
QUEUEORDER_FLAGS := $(if $(filter 1,$(QUEUE)),-D__QUEUE__)
TRACE_FLAGS := $(if $(filter 1,$(TRACE)),-D__TRACE__)
CSV_FLAGS := $(if $(filter 1,$(CSV)),-D__CSV__)
JSON_FLAGS := $(if $(filter 1,$(JSON)),-D__JSON__)
AUTOMODE_FLAGS := $(if $(filter 1,$(AUTO)),-D__AUTO__)
OLDCOMPILER_FLAGS := $(if $(filter 1,$(OLD_COMPILER)),-D__OLDCOMPILER__)
ADVANCEDMETRICS_FLAGS := $(if $(filter 1,$(ADVANCEDMETRICS)), -D__ADVANCEDMETRICS__)
LOG_FLAGS := $(if $(filter 1,$(LOG)), -D__LOG__)
NOQUEUE_FLAGS := $(if $(filter 1,$(NOQUEUE)), -D__NOQUEUE__)
NUMSTAGES_FLAGS := $(if $(filter-out 0,$(NUMSTAGES)),-D__NUMSTAGES__=$(NUMSTAGES))
ENERGYPCM_FLAGS := $(if $(filter 1,$(ENERGYPCM)),-D__ENERGYPCM__)
LIMCORES_FLAGS := $(if $(filter-out 0,$(LIMCORES)),-D__LIMCORES__=$(LIMCORES))

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Kernel optimizations settings
# --------------------------------------------------------------------------------------------------------------------------------------------------
# -D__PWDIST__ : Select the kernel used for the pairwise distance calculation
#	0.- Default - optimized both for CPU and GPU
# 	1.- Optimized(float) - use the kernel optimized with sycl::float type
#	2.- Optimized(sycl::float4) - use the kernel optimized with sycl::float4 type
#	3.- Unoptimized - use the kernel unoptimized
ifeq ($(PWDIST), 1)
    DPCFLAGS += -D__PWDIST__=1
else ifeq ($(PWDIST), 2)
    DPCFLAGS += -D__PWDIST__=2
else ifeq ($(PWDIST), 3)
    DPCFLAGS += -D__PWDIST__=3
endif

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Backend-specific settings
# --------------------------------------------------------------------------------------------------------------------------------------------------
# -D__BACKEND__ : Select the backend used by SYCL in the GPU queue 
#	0.- OpenCL (default) 
#	1.- Level Zero 
#	2.- CUDA
ifeq ($(BACKEND), 1)
    DPCFLAGS += -D__BACKEND__=1
else ifeq ($(BACKEND), 2)
    # Get the Compute Capability using deviceQuery and add it to the DPCFLAGS
    COMPUTE_CAPABILITY := $(shell /usr/local/cuda/extras/demo_suite/deviceQuery | grep 'CUDA Capability' | head -n 1 | awk '{print $$6}' | sed 's/\.//g')
    # Add the CUDA flags to the DPCFLAGS with the Compute Capability
    DPCFLAGS += -D__BACKEND__=2 -fsycl-targets=nvptx64-nvidia-cuda,spir64 -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_${COMPUTE_CAPABILITY} 
endif

# --------------------------------------------------------------------------------------------------------------------------------------------------
# Acquisitions settings
# --------------------------------------------------------------------------------------------------------------------------------------------------
# -D__ACQMODE__ : Select the acquisition mode
#	0.- DEFAULT: Intentar encolar en los cores del dispositivo preferido primero y luego en el otro, luego probar en las colas siguiendo el mismo orden.
#	1.- PRIMARY_SECONDARY: Intentar ocupar los cores y la cola del dispositivo primario primero y luego intentar con el dispositivo secundario.
#	2.- NO_QUEUE: Solo intentar adquirir cores sin utilizar las colas.
ifeq ($(ACQMODE), 0)
    DPCFLAGS += -D__ACQMODE__=0
else ifeq ($(ACQMODE), 1)
    DPCFLAGS += -D__ACQMODE__=1
else ifeq ($(ACQMODE), 2)
    DPCFLAGS += -D__ACQMODE__=2
endif

# --------------------------------------------------------------------------------------------------------------------------------------------------
# EnergyPCM settings
# --------------------------------------------------------------------------------------------------------------------------------------------------
# -D__ENERGYPCM__ : Enables EnergyPCM mode, which uses EnergyPCM to measure the energy consumption of the pipeline.
ifeq ($(filter 1,$(ENERGYPCM)),1)
    PCM_DIR := $(PARENT_DIR)/pcm
    INCLUDES += -I$(INCLUDE_DIR)/utils/energy -I$(PCM_DIR) -I$(PCM_DIR)/build/lib -I$(ENERGY_SRC_DIR)
    MAIN_LINK_FLAGS := -L$(PCM_DIR)/build/lib -lpcm -lpthread -Wl,-rpath,$(PCM_DIR)/build/lib
    OBJ_FILES += $(patsubst $(ENERGY_SRC_DIR)/%.cpp, $(BIN_ENERGY_DIR)/%.o, $(wildcard $(ENERGY_SRC_DIR)/*.cpp))
endif

MAIN_FLAGS := $(DPCFLAGS) \
			$(DEBUG_FLAGS) \
			$(VERBOSE_FLAGS) \
			$(TIMESTAGES_FLAGS) \
			$(QUEUEORDER_FLAGS) \
			$(TRACE_FLAGS) \
			$(CSV_FLAGS) \
			$(JSON_FLAGS) \
			$(AUTOMODE_FLAGS) \
			$(OLDCOMPILER_FLAGS) \
			$(ADVANCEDMETRICS_FLAGS) \
			$(LOG_FLAGS) \
			$(NOQUEUE_FLAGS) \
			$(NUMSTAGES_FLAGS) \
			$(ENERGYPCM_FLAGS) \
			$(LIMCORES_FLAGS)

# Rule for compiling and linking the main program
all: print_vars main

main: $(OBJ_FILES)
	$(CXX) $(MAIN_FLAGS) $(MAIN_LINK_FLAGS) $(OBJ_FILES) -o $@ -lstdc++fs -lsycl

# Compilation rules
$(BIN_CONFIG_DIR)/%.o: $(CONFIG_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_PIPELINE_DIR)/%.o: $(PIPELINE_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_EXECUTORS_DIR)/%.o: $(EXECUTORS_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_FILTERS_DIR)/%.o: $(FILTERS_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_UTILS_GENERAL_DIR)/%.o: $(UTILS_GENERAL_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_UTILS_SPECIFIC_DIR)/%.o: $(UTILS_SPECIFIC_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_UTILS_MANAGER_DIR)/%.o: $(UTILS_MANAGER_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_ENERGY_DIR)/%.o: $(ENERGY_SRC_DIR)/%.cpp  # Nueva regla para compilar EnergyPCM
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_EXPORT_DIR)/%.o: $(EXPORT_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_QUEUE_DIR)/%.o: $(QUEUE_SRC_DIR)/%.cpp
	$(CXX) $(MAIN_FLAGS) $(INCLUDES) -c $< -o $@

# Rule to clean up files generated during compilation removing the 'bin' directory
clean:
	rm -f $(OBJ_FILES) main $(BIN_DIR)/*.o $(BIN_CONFIG_DIR)/*.o $(BIN_PIPELINE_DIR)/*.o $(BIN_EXECUTORS_DIR)/*.o $(BIN_FILTERS_DIR)/*.o $(BIN_UTILS_GENERAL_DIR)/*.o $(BIN_UTILS_SPECIFIC_DIR)/*.o $(BIN_UTILS_MANAGER_DIR)/*.o $(BIN_EXPORT_DIR)/*.o $(BIN_QUEUE_DIR)/*.o $(BIN_ENERGY_DIR)/*.o

# print_vars: Prints the status of optional features during compilation.
print_vars:
	@BACKEND_CPU="C++"; \
	if [ -n "$(AVX)" ] && [ $(AVX) -eq 1 ]; then BACKEND_CPU="AVX"; fi; \
	if [ -n "$(SYCL)" ] && [ $(SYCL) -eq 1 ]; then BACKEND_CPU="SYCL"; fi; \
	if [ -n "$(SIMD)" ] && [ $(SIMD) -eq 1 ]; then BACKEND_CPU="SIMD"; fi; \
	\
	BACKEND_GPU="SYCL(OpenCL)"; \
	if [ -n "$(BACKEND)" ] && [ $(BACKEND) -eq 1 ]; then BACKEND_GPU="SYCL(Level_Zero)"; fi; \
	if [ -n "$(BACKEND)" ] && [ $(BACKEND) -eq 2 ]; then BACKEND_GPU="SYCL(CUDA)"; fi; \
	\
	QUEUE_ORDER="sycl::queue::in_order"; \
	if [ -n "$(QUEUE)" ] && [ $(QUEUE) -eq 1 ]; then QUEUE_ORDER="sycl::queue::out_of_order"; fi; \
	\
	EXTRA_FLAGS="{"; \
	if [ -n "$(DEBUG)" ] && [ $(DEBUG) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS DEBUG,"; fi; \
	if [ -n "$(VERBOSE)" ] && [ $(VERBOSE) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS VERBOSE,"; fi; \
	if [ -n "$(TRACE)" ] && [ $(TRACE) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS TRACE,"; fi; \
	if [ -n "$(TIMESTAGES)" ] && [ $(TIMESTAGES) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS TIMESTAGES,"; fi; \
	if [ -n "$(CSV)" ] && [ $(CSV) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS CSV,"; fi; \
	if [ -n "$(ENERGYPCM)" ] && [ $(ENERGYPCM) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS ENERGYPCM,"; fi; \
	if [ -n "$(PWDIST)" ] && [ $(PWDIST) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS PWDIST(float),"; fi; \
	if [ -n "$(PWDIST)" ] && [ $(PWDIST) -eq 2 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS PWDIST(sycl::float4),"; fi; \
	if [ -n "$(PWDIST)" ] && [ $(PWDIST) -eq 3 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS PWDIST(unoptimized),"; fi; \
	if [ -n "$(AUTO)" ] && [ $(AUTO) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS AUTO,"; fi; \
	if [ -n "$(OLD_COMPILER)" ] && [ $(OLD_COMPILER) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS OLD_COMPILER,"; fi; \
	if [ -n "$(ADVANCEDMETRICS)" ] && [ $(ADVANCEDMETRICS) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS ADVANCEDMETRICS,"; fi; \
	if [ -n "$(LOG)" ] && [ $(LOG) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS LOG,"; fi; \
	if [ -n "$(NOQUEUE)" ] && [ $(NOQUEUE) -eq 1 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS NOQUEUE,"; fi; \
	if [ -n "$(NUMSTAGES)" ] && [ $(NUMSTAGES) -ne 0 ]; then EXTRA_FLAGS="$$EXTRA_FLAGS NUMSTAGES=$(NUMSTAGES),"; fi; \
	EXTRA_FLAGS="$${EXTRA_FLAGS%,} }"; \
	\
	echo "· BACKEND_CPU: $$BACKEND_CPU"; \
	echo "· BACKEND_GPU: $$BACKEND_GPU"; \
	echo "· QUEUE_ORDER: $$QUEUE_ORDER"; \
	if [ "$$EXTRA_FLAGS" != "{}" ]; then \
		echo "· EXTRA_FLAGS: $$EXTRA_FLAGS"; \
	fi;
