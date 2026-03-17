#!/bin/bash
#PBS -A Your_Allocation_Name
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -N nrch_sim
#PBS -j oe

# Navigate to the directory where the job was submitted
cd ${PBS_O_WORKDIR}

# Load necessary modules (adjust paths/modules as required by your specific cluster)
module load nvhpc
module load cray-hdf5

# Compile the CUDA code (if not already compiled)
echo "Compiling code..."
nvcc main.cu -O3 -lcufft -lhdf5 -o nrch_sim

if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi

# Run the simulation, piping the input file and redirecting the output
echo "Starting simulation..."
./nrch_sim < Input.dat > Output.dat

echo "Simulation finished."
