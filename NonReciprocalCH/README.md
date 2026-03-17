# NRCH-ETDRK4: 2D Nonreciprocal Cahn-Hilliard Simulator

A high-performance, GPU-accelerated simulator for the Nonreciprocal Cahn-Hilliard (NRCH) model. This code simulates pattern formation in scalar active mixtures where pair interactions do not follow Newton's third law.

At high nonreciprocal activity, the system exhibits spontaneous time-reversal symmetry breaking, leading to the formation of traveling density waves (an active self-propelled smectic phase) or 2D moving micropatterns.

## Features
* **CUDA-Accelerated Pseudo-Spectral Solver:** Evaluates spatial derivatives exactly in Fourier space using `cuFFT`.
* **4th-Order ETD Runge-Kutta (ETDRK4):** Treats the stiff fourth-order surface tension terms exactly via Exponential Time Differencing, avoiding the crippling $\Delta t$ constraints of explicit explicit Euler methods. Coefficients are computed robustly on the host via Cauchy contour integration.
* **HDF5 I/O:** High-speed, highly compressed binary output of the 2D scalar fields directly from the GPU.
* **Dynamic Driving:** Supports linear time-dependent ramping of model parameters (e.g., activity $\alpha$) during the simulation.
* **Automated Post-Processing:** Python pipeline to instantly render 4-panel diagnostic visualizations (Species 1, Species 2, Modulus, Phase) and encode them into an MP4 video.

## Dependencies

**C++ / CUDA (Simulation):**
* NVIDIA CUDA Toolkit (provides `nvcc` and `libcufft`)
* HDF5 C Library (`libhdf5`)

**Python (Post-Processing):**
* Python 3.8+
* `numpy`, `matplotlib`, `h5py`
* `ffmpeg` (must be installed and available in your system's PATH)

## Compilation

Compile the CUDA source code using `nvcc`. You must link both the Fast Fourier Transform and HDF5 libraries:

`bash
nvcc mainRK4.cu -O3 -lcufft -lhdf5 -o nrch_sim`

## Note for HPC Users: If compiling on a cluster (e.g., ALCF Polaris), ensure the appropriate modules are loaded before compiling:
`bash
module load nvhpc
module load cray-hdf5`

## Configuration (Input.dat): The simulator reads configuration parameters via standard input. Create an Input.dat file in your working directory:

# --- Grid & Time ---
nx 128
ny 128
dx 0.5
dt 0.1
n_steps 10000
seed 42

# --- Output Controls ---
log_interval 100
field_interval 1000
save_fields 1
file_prefix run_alpha_45_

# --- Model Parameters ---
kappa 0.01
chi -0.2
chi_prime 0.2

# --- Time-Dependent Driving (Linear Ramp) ---
alpha_start 0.0
alpha_end 0.45
c11_start 0.2
c11_end 0.2
c12_start 0.5
c12_end 0.5
c21_start 0.1
c21_end 0.1
c22_start 0.5
c22_end 0.5



## Running the Simulation: Execute the compiled binary, piping the input file in and redirecting the scalar diagnostic output (Free Energy, Vorticity, Domain Size) to a log file:
`bash
./nrch_sim < Input.dat > metrics.log`

The program will periodically print progress and current free energy to stderr (visible in your terminal), while writing the heavy 2D grid data to .h5 files based on your file_prefix. 

## Post-Processing & Video Generation: 
Once the simulation completes, use the provided Python script to generate frames and stitch them into an MP4 video. Run the script with arguments matching your input file:
`bash
python postprocess.py --prefix run_alpha_45_ --fps 30 --video_name traveling_bands.mp4`

Output Structuremetrics.log: Tab-separated values tracking Step, Time, Alpha, Free_Energy, RMS_Vorticity, and Domain_Size_R.*.h5: Highly compressed binary files containing the raw $\phi_1$ and $\phi_2$ arrays at each saved interval.frames/: Directory containing the rendered .png plots for each time step.*.mp4: The final animated visualization of the phase separation and traveling waves.

References[1] Saha, S., Agudo-Canalejo, J., & Golestanian, R. (2020). Scalar Active Mixtures: The Nonreciprocal Cahn-Hilliard Model. Physical Review X, 10(4), 041009.
