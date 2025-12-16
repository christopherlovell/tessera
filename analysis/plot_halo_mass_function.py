"""
Plot the FOF halo mass function from the parent SWIFT simulation.

This script reads FOF catalogue files from the parent simulation and plots
the halo mass function (dn/dlog10M) for all available snapshots.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import glob
from matplotlib.cm import viridis


def read_fof_masses(fof_file):
    """
    Read halo masses and metadata from a SWIFT FOF file.
    
    Parameters
    ----------
    fof_file : str or Path
        Path to the FOF HDF5 file.
        
    Returns
    -------
    masses : ndarray
        Halo masses in solar masses.
    redshift : float
        Redshift of the snapshot.
    boxsize : float
        Box size in Mpc.
    """
    with h5py.File(fof_file, 'r') as f:
        # Read masses (in internal units: 10^10 Msun)
        masses = f['Groups/Masses'][:]
        
        # Get cosmology/header info
        h = f['Cosmology'].attrs['h'][0]
        redshift = f['Header'].attrs['Redshift'][0]
        
        # Box size in Mpc (internal units are Mpc)
        boxsize = f['Header'].attrs['BoxSize'][0]  # Already in Mpc
        
        # Unit conversions
        unit_mass_cgs = f['Units'].attrs['Unit mass in cgs (U_M)'][0]
        msun_cgs = 1.98841e33  # Solar mass in grams
        
        # Convert to solar masses
        masses_msun = masses * (unit_mass_cgs / msun_cgs)
        
    return masses_msun, redshift, boxsize


def compute_mass_function(masses, boxsize, n_bins=30, mass_min=None, mass_max=None):
    """
    Compute the halo mass function dn/dlog10(M).
    
    Parameters
    ----------
    masses : ndarray
        Halo masses in solar masses.
    boxsize : float
        Box size in Mpc.
    n_bins : int
        Number of logarithmic mass bins.
    mass_min : float, optional
        Minimum mass for binning.
    mass_max : float, optional
        Maximum mass for binning.
        
    Returns
    -------
    bin_centers : ndarray
        Center of mass bins in log10(M/Msun).
    dn_dlogM : ndarray
        Number density per dlog10(M) in Mpc^-3.
    poisson_err : ndarray
        Poisson error on dn_dlogM.
    """
    if len(masses) == 0:
        return np.array([]), np.array([]), np.array([])
    
    log_masses = np.log10(masses)
    
    if mass_min is None:
        mass_min = log_masses.min()
    if mass_max is None:
        mass_max = log_masses.max()
    
    # Create logarithmic bins
    bin_edges = np.linspace(mass_min, mass_max, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    # Count halos in each bin
    counts, _ = np.histogram(log_masses, bins=bin_edges)
    
    # Volume of the box
    volume = boxsize**3  # Mpc^3
    
    # Compute dn/dlog10(M)
    dn_dlogM = counts / (volume * bin_width)
    
    # Poisson error
    poisson_err = np.sqrt(counts) / (volume * bin_width)
    
    return bin_centers, dn_dlogM, poisson_err


def plot_mass_functions(data_dir, output_file='halo_mass_function.png'):
    """
    Plot the halo mass function for all FOF snapshots.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing the FOF files.
    output_file : str
        Output filename for the plot.
    """
    data_dir = Path(data_dir)
    
    # Find all FOF files
    fof_files = sorted(glob.glob(str(data_dir / 'fof_*.hdf5')))
    
    if len(fof_files) == 0:
        raise FileNotFoundError(f"No FOF files found in {data_dir}")
    
    print(f"Found {len(fof_files)} FOF files")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for snapshots
    colors = viridis(np.linspace(0, 1, len(fof_files)))
    
    # Determine global mass range for consistent binning
    all_masses = []
    for fof_file in fof_files:
        masses, _, _ = read_fof_masses(fof_file)
        if len(masses) > 0:
            all_masses.extend(masses)
    
    if len(all_masses) == 0:
        raise ValueError("No halos found in any FOF file")
    
    log_mass_min = np.log10(min(all_masses))
    log_mass_max = np.log10(max(all_masses))
    
    # Add some padding
    log_mass_min -= 0.1
    log_mass_max += 0.1
    
    print(f"Mass range: 10^{log_mass_min:.1f} - 10^{log_mass_max:.1f} Msun")
    
    # Plot mass function for each snapshot
    for i, fof_file in enumerate(fof_files):
        masses, redshift, boxsize = read_fof_masses(fof_file)
        
        snap_num = Path(fof_file).stem.split('_')[1]
        
        if len(masses) < 10:
            print(f"  Snapshot {snap_num}: Only {len(masses)} halos, skipping")
            continue
        
        bin_centers, dn_dlogM, poisson_err = compute_mass_function(
            masses, boxsize, n_bins=25, 
            mass_min=log_mass_min, mass_max=log_mass_max
        )
        
        # Only plot bins with halos
        mask = dn_dlogM > 0
        
        print(f"  Snapshot {snap_num}: z={redshift:.2f}, {len(masses)} halos")
        
        ax.plot(
            bin_centers[mask], dn_dlogM[mask],
            color=colors[i], 
            label=f'z = {redshift:.2f}',
            linewidth=1.5,
            alpha=0.8
        )
        
        # Add error region for the last snapshot
        if i == len(fof_files) - 1:
            ax.fill_between(
                bin_centers[mask],
                (dn_dlogM - poisson_err)[mask],
                (dn_dlogM + poisson_err)[mask],
                color=colors[i],
                alpha=0.2
            )
    
    # Formatting
    ax.set_xlabel(r'$\log_{10}(M_{\rm halo} / M_\odot)$', fontsize=14)
    ax.set_ylabel(r'$dn/d\log_{10}M$ [Mpc$^{-3}$]', fontsize=14)
    ax.set_title('FOF Halo Mass Function - Parent Simulation', fontsize=14)
    
    ax.set_yscale('log')
    ax.set_xlim(log_mass_min, log_mass_max)
    
    # Set y-axis limits based on data
    ax.set_ylim(1e-8, None)
    
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")
    
    plt.close()


def main():
    """Main function to run the halo mass function plotting."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Plot FOF halo mass function from SWIFT simulation'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default='/snap7/scratch/dp276/dc-love2/tessera/parent',
        help='Directory containing FOF files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename for the plot (default: ../plots/halo_mass_function.png)'
    )
    
    args = parser.parse_args()
    
    # Default output to the top-level plots/ directory
    if args.output is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / 'plots'
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / 'halo_mass_function.png')
    
    print(f"Reading FOF data from: {args.data_dir}")
    print(f"Output file: {args.output}")
    print("-" * 50)
    
    plot_mass_functions(args.data_dir, args.output)


if __name__ == '__main__':
    main()
