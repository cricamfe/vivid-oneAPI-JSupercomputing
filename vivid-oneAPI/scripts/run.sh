#!/bin/bash

# *********************************************************************************************************************************************************************************
# USAGE: ./run.sh [-s] [-o] [-h]
# *********************************************************************************************************************************************************************************
# Read the input parameters
# -s: launch main_serie
# -a: launch optimizations
# -h: help
# -d: default config.env
# -c: configuration with custom config.env

# *********************************************************************************************************************************************************************************
# COMPILATION FLAGS
# *********************************************************************************************************************************************************************************
declare -A flag_configs
flag_configs=(  ["CONFIG"]="CSV=1" ["TIMESTAGES"]="CSV=1 TIMESTAGES=1" )

# *********************************************************************************************************************************************************************************
# GLOBAL VARIABLES
# *********************************************************************************************************************************************************************************
# List of executables:
API_executables=("pipeline" "fgfn" "fgan" "syclevents" "taskflow")

# Number of frames and image resolution: 
declare -A img_res_and_frames
img_res_and_frames=( ["1"]="200" ["3"]="200" )

# Number of threads
num_threads=("8")

# Backend GPU: 0-OpenCL, 1-Level0, 2-CUDA
list_backend_GPU=("0")

# Backend CPU: C++, AVX, SYCL
list_backend_CPU=("C++" "AVX" "SYCL" "SIMD")

# Queue order (0-sycl::queue::out_of_order, 1-sycl::queue::in_order)
list_queue_order=("0")

# List of special configurations
special_config_stages=("DECOUPLED")

# Function to generate all possible configurations
generate_all_configs() {
    local configs=()
    local stages=("0" "1" "2") # 0-CPU, 1-GPU, 2-CPU+GPU
    for stage1 in "${stages[@]}"; do
        for stage2 in "${stages[@]}"; do
            for stage3 in "${stages[@]}"; do
                configs+=("${stage1}${stage2}${stage3}")
            done
        done
    done
    echo "${configs[@]}"
}

# Generate the default configuration stages including special configurations
default_config_stages=($(generate_all_configs) "${special_config_stages[@]}")

# LOOP PARAMETERS
default_num_elements_loop=5                         # Number of times to measure the same configuration
timestages_num_elements_loop=5                      # Number of times to measure the same configuration (TIMESTAGES)

# *********************************************************************************************************************************************************************************
# INTERRUPT HANDLER
# *********************************************************************************************************************************************************************************
# Function to handle the interrupt signal (Ctrl+C) and clean up the executables
interrupt_handler() {
    echo "Interrupt signal (Ctrl+C) detected. Cleaning up..."
    make clean
    exit 1
}

# Set up the interrupt handler for SIGINT (Ctrl+C)
trap interrupt_handler SIGINT

# *********************************************************************************************************************************************************************************
# READ INPUT PARAMETERS
# *********************************************************************************************************************************************************************************
# Get the absolute path of the scripts directory and partent dir from the script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PARENT_DIR="$(dirname "$SCRIPTS_DIR")"

# Function to display the help message
show_help() {
    echo "Usage: ./run.sh [OPTIONS]"
    echo
    echo "Options:"
    echo "  -s                          Launch only main_serie"
    echo "  -a                          Launch only APIs"
    echo "  -d                          Use the default configuration file 'config.env'"
    echo "  -c      FILE                Use a custom configuration file"
    echo "  -h                          Display this help message and exit"
    echo
    echo "Examples:"
    echo "  ./run.sh                    Launch both main_serie and APIs with the default configuration"
    echo "  ./run.sh -s                 Launch only main_serie"
    echo "  ./run.sh -a                 Launch only APIs"
    echo "  ./run.sh -s -a              Launch both main_serie and APIs"
    echo "  ./run.sh -c                 Use the default configuration file 'config.env'"
    echo "  ./run.sh -c config          Use the configuration file 'config'"
    echo "  ./run.sh -d                 Use the default configuration file 'config.env'"
    echo "  ./run.sh -d -s              Launch only main_serie with the default configuration"
    echo "  ./run.sh -d -a              Launch only APIs with the default configuration"
}

read_config_file() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        # Save the current IFS and set it to a newline for the 'source' command
        OLDIFS=$IFS
        IFS=$'\n'
        source "$config_file"
        # Restore the IFS
        IFS=$OLDIFS
    else
        echo "Configuration file '$config_file' not found."
        exit 1
    fi
}

# Function to parse the input parameters
parse_arguments() {
    custom_config_file="" # No default config file
    no_args=true
    while getopts ":hdc:" opt; do
        case $opt in
            h)
                show_help
                exit 0
                ;;
            d)
                custom_config_file="$SCRIPT_DIR/config.env"
                # Check if the default config file exists
                if [ ! -f "$custom_config_file" ]; then
                    echo "Default configuration file '$custom_config_file' not found."
                    exit 1
                fi
                ;;
            c)
                # If -c is provided with an argument, use the provided config file
                custom_config_file=${OPTARG}
                no_args=false
                ;;
            \?)
                echo "Invalid option: -$OPTARG" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

# Call the function to parse the input parameters
parse_arguments "$@"

# If a configuration file is provided, read it and update the variables
if [ -n "$custom_config_file" ]; then
    read_config_file "$custom_config_file"
    # If the flag CONFIG_FLAGS is defined in the configuration file, update the flag_configs array
    if [ -n "$CONFIG_FLAGS" ]; then
        eval "declare -A flag_configs=($CONFIG_FLAGS)"
    fi
    # If the flag IMG_RES_AND_FRAMES is defined in the configuration file, update the img_res_and_frames array
    if [ -n "$IMG_RES_AND_FRAMES" ]; then
        eval "declare -A img_res_and_frames=($IMG_RES_AND_FRAMES)"
    fi
fi

# *********************************************************************************************************************************************************************************
# Extra functions
# *********************************************************************************************************************************************************************************
compile_executable() {
    local API
    local BACKEND_CPU
    local BACKEND_GPU
    local QUEUE
    local CSV
    local JSON
    local TRACE
    local TIMESTAGES
    local VERBOSE
    local DEBUG
    local LOG
    local PWDIST
    local AUTO
    local OLD_COMPILER
    local ADVANCEDMETRICS
    local ACQMODE
    local ENERGYPCM

    local AVX=0
    local SYCL=0
    local SIMD=0

    while (( "$#" )); do
        case "$1" in
            --API)
                API=$2
                shift 2
                ;;
            --BACKEND_CPU)
                BACKEND_CPU=$2
                shift 2
                ;;
            --BACKEND_GPU)
                BACKEND_GPU=$2
                shift 2
                ;;
            --QUEUE)
                QUEUE=$2
                shift 2
                ;;
            --CSV)
                CSV=$2
                shift 2
                ;;
            --JSON)
                JSON=$2
                shift 2
                ;;
            --TRACE)
                TRACE=$2
                shift 2
                ;;
            --TIMESTAGES)
                TIMESTAGES=$2
                shift 2
                ;;
            --VERBOSE)
                VERBOSE=$2
                shift 2
                ;;
            --DEBUG)
                DEBUG=$2
                shift 2
                ;;
            --LOG)
                LOG=$2
                shift 2
                ;;
            --PWDIST)
                PWDIST=$2
                shift 2
                ;;
            --AUTO)
                AUTO=$2
                shift 2
                ;;
            --OLD_COMPILER)
                OLD_COMPILER=$2
                shift 2
                ;;
            --ADVANCEDMETRICS)
                ADVANCEDMETRICS=$2
                shift 2
                ;;
            --ACQMODE)
                ACQMODE=$2
                shift 2
                ;;
            --ENERGYPCM)
                ENERGYPCM=$2
                shift 2
                ;;
            *)
                echo "Error: Invalid argument"
                return 1
        esac
    done

    case $BACKEND_CPU in
        "AVX")
            AVX=1
            ;;
        "SYCL")
            SYCL=1
            ;;
        "SIMD")
            SIMD=1
            ;;
    esac

    local make_args=()
    make_args+=("AVX=$AVX")
    make_args+=("SYCL=$SYCL")
    make_args+=("SIMD=$SIMD")
    [ -n "$BACKEND_CPU" ] && make_args+=("BACKEND=$BACKEND_CPU")
    [ -n "$BACKEND_GPU" ] && make_args+=("BACKEND=$BACKEND_GPU")
    [ -n "$QUEUE" ] && make_args+=("QUEUE=$QUEUE")
    [ -n "$CSV" ] && make_args+=("CSV=$CSV")
    [ -n "$JSON" ] && make_args+=("JSON=$JSON")
    [ -n "$TIMESTAGES" ] && make_args+=("TIMESTAGES=$TIMESTAGES")
    [ -n "$TRACE" ] && make_args+=("TRACE=$TRACE")
    [ -n "$VERBOSE" ] && make_args+=("VERBOSE=$VERBOSE")
    [ -n "$DEBUG" ] && make_args+=("DEBUG=$DEBUG")
    [ -n "$LOG" ] && make_args+=("LOG=$LOG")
    [ -n "$PWDIST" ] && make_args+=("PWDIST=$PWDIST")
    [ -n "$AUTO" ] && make_args+=("AUTO=$AUTO")
    [ -n "$OLD_COMPILER" ] && make_args+=("OLD_COMPILER=$OLD_COMPILER")
    [ -n "$ADVANCEDMETRICS" ] && make_args+=("ADVANCEDMETRICS=$ADVANCEDMETRICS")
    [ -n "$ACQMODE" ] && make_args+=("ACQMODE=$ACQMODE")
    [ -n "$ENERGYPCM" ] && make_args+=("ENERGYPCM=$ENERGYPCM")

    (cd "$PARENT_DIR" && make -j -B "${make_args[@]}")
}

# *********************************************************************************************************************************************************************************
# Automatic selection of cores
# *********************************************************************************************************************************************************************************
# Function to get the list of physical cores
get_physical_core_list() {
  echo "$1" | awk -F, 'BEGIN{IGNORECASE=1} !/^#/ {print $1 " " $2}' | sort -u -k2,2n | awk '{print $1}' | tr '\n' ','
}
lscpu_output=$(lscpu -p)
# Get the list of physical cores
core_list=$(get_physical_core_list "$lscpu_output")
# Delete last comma
core_list=${core_list%,}
# Print the physical cores list
echo "Physical cores: ${core_list}"

# *********************************************************************************************************************************************************************************
# MAIN LOOP
# *********************************************************************************************************************************************************************************
for flag_config_key in "${!flag_configs[@]}"; do
    flag_config=${flag_configs[${flag_config_key}]}
    eval "${flag_config}" # Parse the flags from the current flag configuration
    if [[ "$AUTO" == "1" ]]; then
        num_elements_loop=${default_num_elements_loop}
        list_config_stages=("AUTO")
    elif [[ "$TIMESTAGES" == "1" ]]; then
        num_elements_loop=${timestages_num_elements_loop}
        list_config_stages=("TIMESTAGES")
    else
        num_elements_loop=${default_num_elements_loop}
        list_config_stages=("${default_config_stages[@]}")  # copia el array default_config_stages a list_config_stages
    fi
    # Loop over the list of executables
    for API in ${API_executables[@]}; do
        # Loop over the list of image resolutions
        for key in "${!img_res_and_frames[@]}"; do
            # Get the key of the current image resolution
            img_res="${key%_*}"
            # Get the number of frames
            numFrames="${img_res_and_frames[$key]}"
            echo "Processing image of resolution $img_res with $numFrames frames..."
            # Loop over the list of GPU backends
            for backend_GPU in ${list_backend_GPU[@]}; do
                # Loop over the list of CPU backends
                for backend_CPU in ${list_backend_CPU[@]}; do
                    # Loop over the list of queue orders
                    for queue in ${list_queue_order[@]}; do
                        echo "Procesando con queue order: $queue"
                        # Compile the executable
                        compile_executable --OLD_COMPILER "$OLD_COMPILER" --API "$API" --BACKEND_CPU "$backend_CPU" --BACKEND_GPU "$backend_GPU" --QUEUE "$queue" --CSV "$CSV" --JSON "$JSON" --TIMESTAGES "$TIMESTAGES" --TRACE "$TRACE" --VERBOSE "$VERBOSE" --DEBUG "$DEBUG" --LOG "$LOG" --PWDIST "$PWDIST" --AUTO "$AUTO" --ADVANCEDMETRICS "$ADVANCEDMETRICS" --ACQMODE "$ACQMODE" --ENERGYPCM "$ENERGYPCM"
                        # Loop over the list of configurations
                        for m_stage_config in ${list_config_stages[@]}; do
                            # Loop over the list of threads
                            for nthreads in ${num_threads[@]}; do
                                # Select the first N cores for "CPU" and "000" configurations
                                if [[ "$m_stage_config" == "CPU" ]] || [[ "$m_stage_config" == "000" ]]; then
                                    selected_cores=$(echo "$core_list" | cut -d',' -f1-"$nthreads")
                                    echo "Selected cores (${nthreads}): ${selected_cores}"
                                else
                                    # Select the first N+1 cores for other configurations
                                    selected_cores=$(echo "$core_list" | cut -d',' -f1-$(("$nthreads" + 1)))
                                    echo "Selected cores (${nthreads}+1): ${selected_cores}"
                                fi
                                if [[ "$API" == "serie" ]]; then
                                    for i in $(seq 1 ${num_elements_loop}); do
                                        taskset -c ${selected_cores} ./${PARENT_DIR}/main --api ${API} --duration ${numFrames} --resolution ${img_res} --config ${m_stage_config}
                                    done
                                else
                                    for i in $(seq 1 ${num_elements_loop}); do
                                        taskset -c ${selected_cores} ./${PARENT_DIR}/main --api ${API} --duration ${numFrames} --resolution ${img_res} --threads ${nthreads} --config ${m_stage_config}
                                    done
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done

# *********************************************************************************************************************************************************************************
# END MAIN LOOP
# *********************************************************************************************************************************************************************************

# Get host name
host_name=$(hostname)

# Send notification
curl -X POST -H "Content-Type: application/json" -d '{"value1":"'"$host_name"'"}' https://maker.ifttt.com/trigger/script_finished/with/key/cgEv3FKZXoq40I-bJYBOQb

# Empty line
echo