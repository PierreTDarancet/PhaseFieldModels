# TDGL Phase Field Simulator: Competing Orders under Ultrafast Quench

[cite_start]This repository contains a high-performance GPU-accelerated phase field code designed to simulate the Time-Dependent Ginzburg-Landau (TDGL) dynamics of a system with two coexisting and competing order parameters[cite: 5]. [cite_start]Specifically, it models the non-equilibrium dynamics of Charge Density Wave (CDW) and Superconductivity (SC) orders subjected to an ultrafast laser pulse[cite: 6, 9]. 

[cite_start]The simulation demonstrates the transient suppression of the dominant SC order and the subsequent anomalous enhancement and metastability of the CDW order[cite: 25, 26, 32].

## Features
* **GPU Acceleration:** Fully implemented in C++/CUDA for massive parallelization using NVIDIA GPUs.
* **Stochastic Dynamics:** Utilizes the Euler-Maruyama method with `cuRAND` to model thermal Langevin fluctuations and domain formation.
* **Momentum-Space Analysis:** Integrates `cuFFT` to dynamically calculate the structure factor and the momentum-space correlation length ($\xi_{CDW}$) of the order parameters.
* **Data-Driven Workflows:** Uses a clean `Input.dat` configuration file for parameter management and a standalone Python script for publication-ready visualizations.

## Prerequisites

To compile and run this simulation, your system must have:
* **NVIDIA GPU** with CUDA support.
* **CUDA Toolkit** (includes `nvcc`, `cuRAND`, `cuFFT`, and `Thrust`).
* **Python 3.x** (for the plotting script).
* Python Packages: `numpy`, `matplotlib`.

## File Structure
* `tdgl_simulator.cu`: The core C++/CUDA simulation engine.
* `Input.dat`: The configuration file where all grid, time, and physical parameters are defined.
* `plot_results.py`: A standalone Python script to parse the output data and generate summary figures.

## Quick Start Guide

### 1. Compilation
Compile the CUDA code using `nvcc`. You must link the `cufft` and `curand` libraries.

```bash
nvcc -O3 tdgl_simulator.cu -lcufft -lcurand -o tdgl_simulator



2. Configuration (Input.dat)Ensure the Input.dat file is in the same directory. You can edit this file to change the simulation parameters without recompiling. Key parameters include:Nx, Ny: Grid dimensions.dt: Simulation time step (must be small enough for SDE stability).a1_eq, a2_eq: Equilibrium mass terms for the CDW and SC orders.a2_prime: The pumped/excited mass term for the SC order during the laser pulse.c: The competition/coupling constant between the two orders.Tv: Normalized temperature driving the thermal fluctuations.3. Running the SimulationExecute the compiled binary and pipe the input file into it. We recommend redirecting the standard output to a log file.Bash./tdgl_simulator < Input.dat > terminal_output.log
Outputs generated during the run:[prefix]_timeseries.dat: A dense data file containing the time evolution of the mean fields ($\bar{\psi}_1$, $\bar{\psi}_2$), the spatial cross-correlation ($S^{12}$), and the CDW correlation length ($\xi_{CDW}$).terminal_output.log: The console output showing step-by-step progress.4. VisualizationOnce the simulation completes, run the Python script to generate a 3-panel summary plot.Bashpython plot_results.py
This script will automatically read Input.dat to find your specific output prefix, load the .dat file, and save a high-resolution image ([prefix]_summary.png) showing the trajectory of the mean fields, the cross-correlation, and the normalized correlation length.ReferenceThis code is based on the theoretical framework and parameter spaces presented in:
Masoumi, Y., de la Torre, A., & Fiete, G. A. (2025). Metastability in Coexisting Competing Orders. Physical Review Letters, 135, 066501. 

