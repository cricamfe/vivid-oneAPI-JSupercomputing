/**
 * @file tracer.cpp
 * @brief This file contains the implementation of the Tracer class.
 */

#include "Tracer.hpp"

Tracer::Tracer(bool th) : threads{th} {}

Tracer::Tracer(string filename, bool th) : tracefile{filename, ios::out}, threads{th} {
    write_intro_header();
}

void Tracer::write_intro_header() {
    // write header
    tracefile <<
        R"(%EventDef PajeDefineContainerType 1
%   Alias string
%   ContainerType string
%   Name string
%EndEventDef
%EventDef	PajeDefineEventType	2
%   Alias	string
%	ContainerType	string
%	Name	string
%EndEventDef
%EventDef PajeDefineStateType 3
%   Alias string
%   ContainerType string
%   Name string
%EndEventDef
%EventDef	PajeDefineVariableType	4
%	Alias	string
%	ContainerType	string
%	Name	string
%EndEventDef
%EventDef	PajeDefineLinkType	5
%	Alias	string
%	ContainerType	string
%	SourceContainerType	string
%	DestContainerType	string
%	Name	string
%EndEventDef
%EventDef PajeDefineEntityValue 6
%   Alias string
%   EntityType string
%   Name string
%   Color	color
%EndEventDef
%EventDef PajeCreateContainer 7
%   Time date
%   Alias string
%   Type string
%   Container string
%   Name string
%EndEventDef
%EventDef PajeDestroyContainer 8
%   Time date
%   Name string
%   Type string
%EndEventDef
%EventDef	PajeNewEvent	9
%	Time	date
%	Type	string
%	Container	string
%	Value	string
%EndEventDef
%EventDef PajeSetState 10
%   Time date
%   Type string
%   Container string
%   Value string
%EndEventDef
%EventDef	PajePushState	11
%	Time	date
%	Type	string
%	Container	string
%	Value	string
%EndEventDef
%EventDef	PajePopState	12
%	Time	date
%	Type	string
%	Container	string
%EndEventDef
%EventDef	PajeSetVariable	13
%	Time	date
%	Container	string
%	Type	string
%	Value	double
%EndEventDef
%EventDef	PajeAddVariable	14
%	Time	date
%	Container	string
%	Type	string
%	Value	double
%EndEventDef
%EventDef	PajeSubVariable	15
%	Time	date
%	Container	string
%	Type	string
%	Value	double
%EndEventDef
1 P 0 Program
1 T P Task
3 S T State
4 TK P "Variable del Programa"
6 C S CPU "0.0 0.7 1.0"
6 G S GPU "0.2 1.0 0.5"
6 I S IDLE "0.5 0.5 0.5"
6 W S WAIT "1.0 0.75 0.8"
6 Q_CPU S QUEUE_CPU "0.7 1.0 1.0"
6 Q_GPU S QUEUE_GPU "0.7 1.0 0.7"
)";
    start = tick_count::now();
    tracefile << "7 " << (start - start).seconds() << " V P 0 \"Vivid\"" << std::endl;
    tracefile << "7 " << (start - start).seconds() << "  T1 T V \"Tokens\"" << std::endl;
    tracefile << "13 " << (start - start).seconds() << " T1 TK 0.0" << std::endl;
}

void Tracer::open(string filename) {
    tracefile.open(filename, ios::out);
    write_intro_header();
}

void Tracer::frame_start(ViVidItem *item) {
    tick_count now = tick_count::now();
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        tracefile << "7 " << (now - start).seconds() << " F" << m_id << " T V \"Frame " << m_id << "\"" << std::endl; // create frame
        tracefile << "10 " << (now - start).seconds() << " S F" << m_id << " I" << std::endl;                         // set state to idle
        tracefile << "14 " << (now - start).seconds() << " T1 TK 1.0" << std::endl;                                   // add n tokens +1
    }
}

void Tracer::frame_end(ViVidItem *item) {
    tick_count now = tick_count::now();
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        item->traceItem << "8 " << (now - start).seconds() << " F" << m_id << " T " << std::endl; // destroy frame
        item->traceItem << "15 " << (now - start).seconds() << " T1 TK 1.0" << std::endl;         // add n tokens -1

        tracefile << item->traceItem.rdbuf();
        item->traceItem.str("");
        item->traceItem.clear();
    }
}

void Tracer::cpu_start(ViVidItem *item) {
    tick_count now = tick_count::now();
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        if constexpr (SYCL_ENABLED) {
            item->traceItem << "11 " << (now - start).seconds() << " S F" << m_id << " Q_CPU " << std::endl; // push queue CPU
        } else {
            item->traceItem << "11 " << (now - start).seconds() << " S F" << m_id << " C " << std::endl; // push CPU
        }
    }
}

void Tracer::cpu_end(ViVidItem *item, double execution_time) {
    tick_count now = tick_count::now();
    tbb::tick_count::interval_t exec_time(((double)execution_time) / 1.0e9);
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        if constexpr (SYCL_ENABLED) {
            item->traceItem << "12 " << (now - start).seconds() - exec_time.seconds() << " S F" << m_id << std::endl;            // pull previous state (QUEUE)
            item->traceItem << "11 " << ((now - start).seconds() - exec_time.seconds()) << " S F" << m_id << " C " << std::endl; // push kernel
            item->traceItem << "12 " << (now - start).seconds() << " S F" << m_id << std::endl;                                  // pull previous state
        } else {
            item->traceItem << "12 " << (now - start).seconds() << " S F" << m_id << std::endl; // pull previous state
        }
    }
}

void Tracer::gpu_start(ViVidItem *item) {
    tick_count now = tick_count::now();
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        item->traceItem << "11 " << (now - start).seconds() << " S F" << m_id << " Q_GPU " << std::endl; // push GPU
    }
}

void Tracer::gpu_end(ViVidItem *item, double execution_time) {
    tick_count now = tick_count::now();
    tbb::tick_count::interval_t exec_time(((double)execution_time) / 1.0e9);
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        item->traceItem << "12 " << (now - start).seconds() - exec_time.seconds() << " S F" << m_id << std::endl;            // pull previous state (QUEUE)
        item->traceItem << "11 " << ((now - start).seconds() - exec_time.seconds()) << " S F" << m_id << " G " << std::endl; // push kernel
        item->traceItem << "12 " << (now - start).seconds() << " S F" << m_id << std::endl;                                  // pull previous state
    }
}

void Tracer::wait_start(ViVidItem *item) {
    tick_count now = tick_count::now();
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        tracefile << "11 " << (now - start).seconds() << " S F" << m_id << " W " << std::endl; // push CPU
    }
}

void Tracer::wait_end(ViVidItem *item) {
    tick_count now = tick_count::now();
    const size_t m_id = item->item_id;
    tbb::queuing_mutex::scoped_lock lock(qmtx);
    {
        tracefile << "12 " << (now - start).seconds() << " S F" << m_id << std::endl; // pull previous state
    }
}

// Destructor
Tracer::~Tracer() {
    tick_count now = tick_count::now();
    tracefile.close();
}

void createTraceFile(int argc, char *argv[], Tracer &my_tracer) {
    if constexpr (VERBOSE_ENABLED) {
        printf(" Creating the trace file...\n");
    }

    std::filesystem::path executable_path = std::filesystem::current_path() / std::filesystem::path(argv[0]);
    std::string api_value;
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == "--api") {
            if (i + 1 < argc) {          // Make sure we aren't at the end of argv!
                api_value = argv[i + 1]; // Increment 'i' so we get the argument after the '--api'
            }
            break;
        }
    }

    std::string executable_name = "Pipeline"; // Default name
    if (!api_value.empty()) {
        executable_name = api_value;
    }

    std::filesystem::path trace_folder = executable_path.parent_path() / "trace";

    if (!std::filesystem::exists(trace_folder)) {
        std::filesystem::create_directory(trace_folder);
        if constexpr (VERBOSE_ENABLED) {
            printf(" Created folder %s\n", trace_folder.c_str());
        }
    }

    // Create subfolder for the executable
    std::string exe_subfolder = (trace_folder / executable_name).string();
    if (!std::filesystem::exists(exe_subfolder)) {
        std::filesystem::create_directory(exe_subfolder);
        if constexpr (VERBOSE_ENABLED) {
            printf(" Created folder %s\n", exe_subfolder.c_str());
        }
    }

    // Create subfolder for the backend
    std::string backend = (__BACKEND__ == 2) ? "dGPU" : "iGPU";
    std::string backend_subfolder = (std::filesystem::path(exe_subfolder) / backend).string();
    if (!std::filesystem::exists(backend_subfolder)) {
        std::filesystem::create_directory(backend_subfolder);
        if constexpr (VERBOSE_ENABLED) {
            printf(" Created folder %s\n", backend_subfolder.c_str());
        }
    }

    // Create subfolder for the implementation
    std::string implementation = SYCL_ENABLED ? "SYCL" : (AVX_ENABLED ? "AVX" : (SIMD_ENABLED ? "SIMD" : "CPP"));
    std::string impl_subfolder = (std::filesystem::path(backend_subfolder) / implementation).string();
    if (!std::filesystem::exists(impl_subfolder)) {
        std::filesystem::create_directory(impl_subfolder);
        if constexpr (VERBOSE_ENABLED) {
            printf(" Created folder %s\n", impl_subfolder.c_str());
        }
    }

    std::stringstream filename;
    filename << executable_name + (TIMESTAGES_ENABLED ? "_stages" : "");

    std::string implemen = SYCL_ENABLED ? "_SYCL" : (AVX_ENABLED ? "_AVX" : (SIMD_ENABLED ? "_SIMD" : "_CPP"));

    for (int i = 1; i < argc; ++i) {
        filename << "_" << argv[i];
    }

    filename << ".trace";
    my_tracer.open((std::filesystem::path(impl_subfolder) / filename.str()).string());
}
