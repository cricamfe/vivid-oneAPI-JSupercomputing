# ViVid Project

The ViVid project aims to improve the implementation and optimization of algorithms for handling massive data flows, using oneAPI in heterogeneous architectures, mainly with CPU+GPU, and eventually FPGA. The methodology consists of selecting appropriate algorithms, developing code and dependencies using oneAPI, identifying and solving performance bottlenecks, and maximizing parallelization. Our goal is to achieve significant advances in real-time processing of large volumes of data, efficiently leveraging the performance of state-of-the-art heterogeneous architectures.

## Table of Contents

- [Previous Versions](#previous-versions)
- [Current Contributions](#current-contributions)

## Previous Versions

There are four previous versions of the project:

1. **FlowGraph-OpenCL-node:** This version of the project, developed by Jose Carlos Romero and Felipe Muñoz, is based on deprecated classes that were available in the 2020 release of TBB. Although these classes have been fundamental in the development of the application, they are currently marked as DEPRECATED in oneTBB (oneAPI), indicating the need to update or replace these classes to ensure long-term compatibility and performance.
2. **TBB_Pipeline_OldVivid:** The version developed by Antonio Vilches, Rafael Asenjo and  Andrés Rodriguez is based on a flat TBB pipeline, a parallel and efficient programming structure. In addition, it included a template for a heterogeneous pipeline that was created by Antonio during the course of his doctoral research. This template allows the combination and coordination of different types of processors in a single workflow.
3. **TBB_pipeline_CPU+GPU_noTemplate:** An older version, developed by Antonio Vilches and Rafael Asenjo, which represents an earlier stage of the project. Although it does not include the template for a heterogeneous pipeline, its simplified approach can facilitate the understanding of the pipeline fundamentals.
4. **vivid_series:** The original version of the ViVid project focuses on sequential image processing using three specific filters, which run exclusively on the CPU. This version laid the foundation for the project and provides a solid starting point for the rest of the optimizations.

## Current Contributions

Current contributions to the project:

5. **oneAPI_pipeline:** This version represents an evolution of the **vivid_series project**, incorporating improvements introduced in **TBB_Pipeline_OldVivid** and **FlowGraph-OpenCL-node**. The main differences from previous versions include:

   - **main_series:** This serial implementation of the Vivid project, similar to vivid_series, has been updated to include the latest C++17 and oneAPI features, allowing for improved performance and functionality.
   - **main_pipeline:** This implementation leverages the oneTBB parallel_pipeline to create a heterogeneous pipeline that processes images using a heterogeneous workflow, optimizing the use of compute resources.
   - **main_fgfn:** This version uses oneTBB's Flow Graph to design a graph that processes images using a heterogeneous flow, increasing flexibility and efficiency in task execution.
   - **main_fgan:** Very similar to main_fgfn, this implementation adds asynchronous nodes to the GPU, which allows freeing oneTBB worker threads while processing stages on the GPU, optimizing performance and scalability.
   - **main_syclevents:** In this version we simulate the generation of a heterogeneous pipeline using SYCL events. It arises as an alternative to the automatic management of oneTBB resources, offering greater control over the execution of tasks and the synchronization between them. This version does not use oneTBB.

This project update addresses and improves some of the aspects of image processing that we had in the ViVid project, while exploring different optimization strategies and adapting to the latest developments in the field of heterogeneous computing. For more information, click [here](oneAPI_pipeline/README.md).
