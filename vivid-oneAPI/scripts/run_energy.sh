#!/bin/bash

# Ensure the script is running as root or with sudo
if [[ "$EUID" -ne 0 ]]; then
    echo "Please run this script as root or using sudo"
    exit 1
fi

# Define values for the --api, backend, and config parameters
api_values=("pipeline" "fgfn" "fgan" "syclevents")
backend_values=("C++" "SYCL" "AVX" "SIMD")  # "C++" represents the case without SIMD
config_values=("AUTO" "CPU" "GPU")  # Config options

# Path to the setvars.sh file
SETVARS_PATH="/opt/intel/oneapi/setvars.sh"

# Number of repetitions for energy measurements
repetitions=1

# Option to include throughput settings (set to true to enable, false to disable)
include_throughput=true

# Iterate over the values of the parameters
for api in "${api_values[@]}"; do
    for backend in "${backend_values[@]}"; do
        for config in "${config_values[@]}"; do
            # Define the backend_flags variable based on the current backend value
            if [[ "$backend" == "C++" ]]; then
                backend_flags=""
                thcpu="--thcpu 9 557 2.8"
            elif [[ "$backend" == "SYCL" ]]; then
                backend_flags="SYCL=1"
                thcpu="--thcpu 27.4 1090.5 5.7"
            elif [[ "$backend" == "AVX" ]]; then
                backend_flags="AVX=1"
                thcpu="--thcpu 20 554.1 24.3"
            elif [[ "$backend" == "SIMD" ]]; then
                backend_flags="SIMD=1"
                thcpu="--thcpu 50 499 26.1"
            fi

            # Define the config_flags variable based on the current config value
            if [[ "$config" == "AUTO" ]]; then
                config_flags="AUTO=1"
                time_flags="--timesampling 10s"
            else
                config_flags=""
                time_flags=""
            fi

            # Define throughput arguments
            if [[ "$include_throughput" == true ]]; then
                thgpu="--thgpu 37 673.8 6.0"
            else
                thcpu=""
                thgpu=""
            fi

            # Perform energy measurements three times
            for ((i=1; i<=repetitions; i++)); do
                echo "Running measurement $i with --api=$api, backend=$backend, config=$config"

                # Execute the command with the current parameters
                sudo bash -c "
                    . $SETVARS_PATH && \
                    export CPLUS_INCLUDE_PATH=/usr/include/x86_64-linux-gnu/c++/12:\$CPLUS_INCLUDE_PATH && \
                    make $config_flags $backend_flags ENERGYPCM=1 JSON=1 LIMCORES=1 -Bj && \
                    taskset -c 0,2,4,6,8,10,12,14,16 ./main --api $api --threads 8 --resolution 3 --numframes 1000 $timeflags --config $config $thcpu $thgpu
                "

                # Check if there was an error
                if [[ $? -ne 0 ]]; then
                    echo "Error running combination --api=$api, backend=$backend, and config=$config during measurement $i"
                    exit 1
                fi

                echo "Measurement $i with --api=$api, backend=$backend, and config=$config completed successfully."
            done
        done
    done
done