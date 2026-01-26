"""
Plot the FOF halo mass function from the parent SWIFT simulation.

This script reads FOF catalogue files from the parent simulation and plots
the halo mass function (dn/dlog10M) for all available snapshots.
"""

from pathlib import Path
import glob

import h5py
import matplotlib as mpl
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap

# Apply local matplotlib defaults for analysis plots.
_MPLRC = Path(__file__).resolve().parent / "matplotlibrc.txt"
if _MPLRC.exists():
    mpl.rc_file(str(_MPLRC))

import matplotlib.pyplot as plt
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
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    
    # Color map for snapshots
    colors = viridis(np.linspace(0, 1, len(fof_files)))

    plotted_redshifts = []
    plotted_colors = []
    
    # Determine global mass range for consistent binning
    all_masses = []
    for fof_file in fof_files:
        masses, _, _ = read_fof_masses(fof_file)
        if len(masses) > 0:
            all_masses.extend(masses)
    
    if len(all_masses) == 0:
        raise ValueError("No halos found in any FOF file")
    
    log_mass_min = 14  # np.log10(min(all_masses))
    log_mass_max = np.log10(max(all_masses))
    
    # Add some padding
    log_mass_min -= 0.0
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

        plotted_redshifts.append(float(redshift))
        plotted_colors.append(colors[i])
        
        ax.plot(
            bin_centers[mask], dn_dlogM[mask],
            color=colors[i], 
            linewidth=1.5,
            alpha=0.8
        )
        
        # Add error region for the last snapshot
        if i == len(fof_files) - 1:
            ax.fill_between(
                bin_centers[mask],
                np.max([[1e-9]*np.sum(mask), (dn_dlogM - poisson_err)[mask]], axis=0),
                (dn_dlogM + poisson_err)[mask],
                color=colors[i],
                alpha=0.2
            )
    
    # Formatting
    ax.set_xlabel(r'${\rm log_{10}} \, M_{\rm halo} \,/\, {\rm M_\odot}$', fontsize=14)
    ax.set_ylabel(r'$\Phi \,/\, {\rm Mpc^{-3}} \, {\rm dex^{-1}}$', fontsize=14)
    ax.set_yscale('log')
    ax.set_xlim(log_mass_min, log_mass_max)
    
    # Set y-axis limits based on data
    ax.set_ylim(1e-9, None)
    
    # Discrete colorbar keyed by snapshot redshift.
    if plotted_redshifts:
        line_alpha = 0.8
        cmap_colors = np.asarray(plotted_colors, dtype=np.float64)
        if cmap_colors.ndim != 2 or cmap_colors.shape[1] not in (3, 4):
            raise RuntimeError(f"Unexpected plotted_colors array shape: {cmap_colors.shape}")
        if cmap_colors.shape[1] == 3:
            cmap_colors = np.concatenate([cmap_colors, line_alpha * np.ones((cmap_colors.shape[0], 1))], axis=1)
        else:
            cmap_colors = cmap_colors.copy()
            cmap_colors[:, 3] = line_alpha
        cmap = ListedColormap(cmap_colors)

        bounds = np.arange(len(plotted_colors) + 1, dtype=np.float64)
        norm = BoundaryNorm(bounds, cmap.N)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        ticks = np.arange(len(plotted_colors), dtype=np.float64) + 0.5
        # Place the discrete colorbar in the top-right corner of the axes, using an explicit
        # axes-fraction placement (more predictable than `borderpad`).
        cax = ax.inset_axes([0.92, 0.42, 0.035, 0.55])
        cax.patch.set_facecolor("white")
        cax.patch.set_alpha(0.75)
        cax.patch.set_edgecolor("none")
        cbar = fig.colorbar(sm, cax=cax, boundaries=bounds, ticks=ticks, spacing="proportional", drawedges=True)
        # Ensure alpha is applied even if the colorbar artist overrides colormap alpha.
        if getattr(cbar, "solids", None) is not None:
            cbar.solids.set_alpha(line_alpha)
        for coll in getattr(cbar.ax, "collections", []):
            coll.set_alpha(line_alpha)
        # Put ticks/labels outside the bar.
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.tick_params(which="major", direction="out", pad=2)
        cbar.ax.set_yticklabels([f"{z:.2f}" for z in plotted_redshifts], fontsize=11)
        cbar.ax.set_title("Redshift", pad=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Save figure
    output_path = Path(output_file)
    plt.savefig(output_path, bbox_inches='tight')
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
