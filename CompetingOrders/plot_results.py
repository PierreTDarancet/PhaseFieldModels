import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_input_file(filename="Input.dat"):
    """Reads the prefix and tau_0 from the input file to structure the plots."""
    params = {'output_prefix': 'sim_out', 'tau_0': 0.1}
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.split('#')[0].strip()
                if '=' in line:
                    key, val = [x.strip() for x in line.split('=', 1)]
                    if key == 'output_prefix':
                        params[key] = val
                    elif key == 'tau_0':
                        params[key] = float(val)
    except FileNotFoundError:
        print(f"Warning: {filename} not found. Using default plot parameters.")
    return params

def create_plots():
    # 1. Get configurations
    params = parse_input_file("Input.dat")
    prefix = params['output_prefix']
    tau_0 = params['tau_0']
    
    data_file = f"{prefix}_timeseries.dat"
    
    # 2. Load data
    try:
        # skip_header=1 ignores the text columns
        data = np.genfromtxt(data_file, skip_header=1)
        t = data[:, 0]
        mean_psi1 = data[:, 1]
        mean_psi2 = data[:, 2]
        s12 = data[:, 3]
        xi = data[:, 4]
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
        sys.exit(1)

    # Normalize xi to its initial equilibrium value (t=0)
    xi_norm = xi / xi[0] if xi[0] > 0 else xi

    # 3. Create the 3-panel Figure
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    
    # --- Panel A: Mean Fields ---
    axes[0].plot(t, mean_psi1, color='indigo', label=r'$\bar{\psi}_1$ (CDW)')
    axes[0].plot(t, mean_psi2, color='teal', label=r'$\bar{\psi}_2$ (SC)')
    axes[0].set_ylabel('Mean Amplitude')
    axes[0].set_title('Mean Order Parameters')
    axes[0].legend()
    
    # --- Panel B: Cross-Correlation ---
    axes[1].plot(t, s12, color='crimson')
    axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[1].set_ylabel(r'$S^{12} = \langle \delta\psi_1 \delta\psi_2 \rangle$')
    axes[1].set_title('Spatial Cross-Correlation')

    # --- Panel C: Correlation Length ---
    axes[2].plot(t, xi_norm, color='darkorange')
    axes[2].set_ylabel(r'$\xi_{CDW}(t) / \xi_{CDW}^0$')
    axes[2].set_title('Normalized CDW Correlation Length')
    axes[2].set_xlabel('Time (ps)')

    # Add pump-on shading to all panels
    for ax in axes:
        ax.axvspan(0, tau_0, color='gray', alpha=0.3)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_img = f"{prefix}_summary.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved successfully as {output_img}")
    plt.show()

if __name__ == "__main__":
    create_plots()

