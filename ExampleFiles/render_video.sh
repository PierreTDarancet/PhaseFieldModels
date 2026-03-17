#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "1. Generating PNG frames from HDF5 data..."
python plot_h5.py --prefix nrch_run_A_

echo "2. Stitching frames into MP4 video..."
# -y overwrites the output file if it already exists
ffmpeg -y -framerate 30 -i frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p -crf 18 nrch_simulation.mp4

echo "Done! Video saved as nrch_simulation.mp4."
