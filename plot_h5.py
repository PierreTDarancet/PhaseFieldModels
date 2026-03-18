import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

# Allow prefix to be passed as an argument
parser = argparse.ArgumentParser(description="Process NRCH HDF5 output.")
parser.add_argument("--prefix", type=str, default="nrch_run_A_", help="Prefix of the HDF5 files")
args = parser.parse_args()

files = sorted(glob.glob(f"{args.prefix}*.h5"))
os.makedirs("frames", exist_ok=True)

if not files:
    print(f"No files found matching {args.prefix}*.h5")
    exit()

for file in files:
    step = int(file.split('_')[-1].split('.')[0])
    
    with h5py.File(file, 'r') as f:
        phi1 = np.array(f['phi1'])
        phi2 = np.array(f['phi2'])
        
    modulus_sq = phi1**2 + phi2**2
    phase = np.arctan2(phi2, phi1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    
    im1 = axes[0].imshow(phi1, cmap='viridis', origin='lower')
    axes[0].set_title(r"Species $\phi_1$")
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(phi2, cmap='magma', origin='lower')
    axes[1].set_title(r"Species $\phi_2$")
    fig.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(modulus_sq, cmap='plasma', origin='lower')
    axes[2].set_title(r"Modulus $|\Phi|^2$")
    fig.colorbar(im3, ax=axes[2])
    
    im4 = axes[3].imshow(phase, cmap='hsv', origin='lower')
    axes[3].set_title(r"Phase $\arg(\Phi)$")
    fig.colorbar(im4, ax=axes[3])
    
    plt.suptitle(f"NRCH Simulation | Step {step}")
    plt.tight_layout()
    
    out_name = f"frames/frame_{step:10d}.png"
    plt.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"Rendered {out_name}")
