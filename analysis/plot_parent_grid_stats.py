#!/usr/bin/env python
"""
Plot basic statistics of a gridder overdensity field for a parent simulation.

Inputs
------
- A gridder output HDF5 file containing:
  - `Grids/GridPointPositions` (regular grid points)
  - `Grids/Kernel_*/GridPointOverDensities` (delta = rho/rho_bar - 1)

Outputs
-------
- A figure with:
  - Histogram of log(1+delta) for each kernel
  - Power spectrum P(k) computed directly from the gridded delta field
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_boxsize(grid_path: Path) -> float:
    with h5py.File(grid_path, "r") as f:
        box = np.array(f["Header"].attrs["BoxSize"], dtype=np.float64)
    return float(box.item()) if box.ndim == 0 else float(box[0])


def load_gridder_sorted(grid_path: Path) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, float]]:
    """
    Load grid point positions and overdensities, sorted into a consistent (x,y,z) order.

    Returns
    -------
    pos : (N,3) positions sorted by x,y,z (z fastest)
    overd : dict kernel_index -> delta array sorted the same way
    radii : dict kernel_index -> KernelRadius
    """
    with h5py.File(grid_path, "r") as f:
        pos = np.array(f["Grids/GridPointPositions"], dtype=np.float64)
        sort = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
        pos = pos[sort]

        grids = f["Grids"]
        kernels: list[tuple[float | None, str]] = []
        overd_by_name: dict[str, np.ndarray] = {}
        for key in grids.keys():
            if not key.startswith("Kernel_"):
                continue
            grp = grids[key]
            rad = grp.attrs.get("KernelRadius")
            kernels.append((None if rad is None else float(rad), key))
            overd_by_name[key] = np.array(grp["GridPointOverDensities"], dtype=np.float64)[sort]

        if not kernels:
            raise RuntimeError(f"No Kernel_* groups found in {grid_path}")

        if any(r is None for (r, _) in kernels):
            kernels.sort(key=lambda t: t[1])
        else:
            kernels.sort(key=lambda t: float(t[0]))

        overd: dict[int, np.ndarray] = {}
        radii: dict[int, float] = {}
        for kidx, (rad, name) in enumerate(kernels):
            overd[kidx] = overd_by_name[name]
            if rad is not None:
                radii[kidx] = float(rad)

    return pos, overd, radii


def reshape_to_grid(pos: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    xs = np.unique(pos[:, 0])
    ys = np.unique(pos[:, 1])
    zs = np.unique(pos[:, 2])
    nx, ny, nz = int(xs.size), int(ys.size), int(zs.size)
    if nx * ny * nz != pos.shape[0]:
        raise RuntimeError(
            "GridPointPositions do not form a complete Cartesian product "
            f"(unique counts: {nx}×{ny}×{nz} = {nx*ny*nz}, N={pos.shape[0]})."
        )
    if field.shape[0] != pos.shape[0]:
        raise ValueError("field length does not match positions length")
    grid = field.reshape((nx, ny, nz))
    return grid, (nx, ny, nz)


def binned_pk(delta: np.ndarray, box_size: float, nbins: int = 60) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = delta.shape
    fft = np.fft.fftn(delta)
    power3d = (np.abs(fft) ** 2) / (nx * ny * nz)

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=box_size / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=box_size / ny)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=box_size / nz)
    kx3, ky3, kz3 = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(kx3**2 + ky3**2 + kz3**2).ravel()
    pflat = power3d.ravel()

    m = kmag > 0
    kmag = kmag[m]
    pflat = pflat[m]

    kmin = float(kmag.min())
    kmax = float(kmag.max())
    edges = np.logspace(np.log10(kmin), np.log10(kmax), nbins + 1)
    which = np.digitize(kmag, edges) - 1
    centers = np.sqrt(edges[:-1] * edges[1:])
    pk = np.full(nbins, np.nan, dtype=np.float64)
    for i in range(nbins):
        sel = which == i
        if np.any(sel):
            pk[i] = float(pflat[sel].mean())
    return centers, pk


def summarize(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return {
        "min": float(arr.min()),
        "p16": float(np.percentile(arr, 16)),
        "median": float(np.median(arr)),
        "p84": float(np.percentile(arr, 84)),
        "max": float(arr.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot log(1+delta) distribution and P(k) for a gridder output.")
    ap.add_argument("--grid", type=Path, required=True, help="Gridder output HDF5.")
    ap.add_argument("--out", type=Path, default=Path("plots/gridder_stats.png"), help="Output PNG.")
    ap.add_argument("--nbins", type=int, default=80, help="Histogram bins for log(1+delta).")
    ap.add_argument("--pk-bins", type=int, default=60, help="Number of P(k) bins.")
    ap.add_argument(
        "--kernels",
        type=int,
        nargs="*",
        help="Kernel indices to include (0-based after sorting by KernelRadius). Default: all.",
    )
    args = ap.parse_args()

    box = read_boxsize(args.grid)
    pos, overd, radii = load_gridder_sorted(args.grid)

    kernel_indices = sorted(overd.keys()) if not args.kernels else list(args.kernels)
    for k in kernel_indices:
        if k not in overd:
            raise ValueError(f"Kernel index {k} not available in {args.grid} (have {sorted(overd.keys())}).")

    fig, (ax_hist, ax_pk) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for i, k in enumerate(kernel_indices):
        d = overd[k]
        label = f"{radii[k]:g} Mpc" if k in radii else f"Kernel {k}"
        color = plt.cm.tab10(i % 10)

        # Distribution of log(1+delta)
        if np.any(d <= -1):
            bad = int(np.sum(d <= -1))
            raise RuntimeError(f"Kernel {k}: found {bad} cells with delta <= -1, cannot take log(1+delta).")
        x = np.log10(1.0 + d)
        stats = summarize(x)
        print(f"Kernel {k} ({label}) log10(1+delta) stats: {stats}")
        ax_hist.hist(x, bins=args.nbins, histtype="step", density=True, color=color, lw=1.5, label=label)

        # Power spectrum from gridded delta
        grid, shape = reshape_to_grid(pos, d)
        mean_delta = float(np.mean(grid))
        if abs(mean_delta) > 5e-3:
            print(f"Kernel {k} ({label}) warning: mean(delta)={mean_delta:.3g} (expected ~0).")
        kcen, pk = binned_pk(grid, box_size=box, nbins=args.pk_bins)
        m = np.isfinite(pk)
        ax_pk.loglog(kcen[m], pk[m], color=color, lw=1.5, label=label)

    ax_hist.set_xlabel("log10(1 + overdensity)")
    ax_hist.set_ylabel("PDF")
    ax_hist.legend(title="Kernel radius")

    ax_pk.set_xlabel("k [1/Mpc]")
    ax_pk.set_ylabel("P(k) [(Mpc)^3]")
    ax_pk.grid(True, which="both", alpha=0.2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
