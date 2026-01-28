#!/usr/bin/env python3
"""
Plot a 2D histogram of overdensity measured on two different smoothing scales.

This reads two Kernel_* overdensity grids from a gridder HDF5 output (matching by KernelRadius),
converts to x = log10(1+delta), and plots a 2D histogram in (x_R1, x_R2) space.

Example
-------
python emulator/plot_overdensity_2d.py \\
  --gridder-file /path/to/gridder_output_512.hdf5 \\
  --kernel-radius-1 10 --kernel-radius-2 20 \\
  --out ../plots/overdensity_2d_R10_R20.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_GRIDDER_FILE = Path("/cosma7/data/dp004/dc-love2/data/tessera/parent/gridder/gridder_output_512.hdf5")


def _list_kernel_radii(gridder_file: Path) -> list[float]:
    import h5py

    radii: list[float] = []
    with h5py.File(Path(gridder_file), "r") as f:
        if "Grids" not in f:
            return []
        for name in f["Grids"]:
            if not str(name).startswith("Kernel_"):
                continue
            r = f["Grids"][name].attrs.get("KernelRadius", None)
            if r is None:
                continue
            rr = float(np.asarray(r).reshape(-1)[0])
            if np.isfinite(rr):
                radii.append(rr)
    return sorted(set(radii))


def _load_overdensity_for_radius(gridder_file: Path, *, kernel_radius: float) -> np.ndarray:
    import h5py

    gridder_file = Path(gridder_file)
    r_target = float(kernel_radius)
    with h5py.File(gridder_file, "r") as f:
        if "Grids" not in f:
            raise KeyError(f"{gridder_file}: missing 'Grids' group")
        grids = f["Grids"]
        best = None
        for name in grids:
            if not str(name).startswith("Kernel_"):
                continue
            g = grids[name]
            r = g.attrs.get("KernelRadius", None)
            if r is None:
                continue
            r = float(np.asarray(r).reshape(-1)[0])
            if np.isfinite(r) and np.isclose(r, r_target, rtol=0, atol=1e-8):
                best = g
                break
        if best is None:
            radii = []
            for name in grids:
                if not str(name).startswith("Kernel_"):
                    continue
                r = grids[name].attrs.get("KernelRadius", None)
                if r is None:
                    continue
                rr = float(np.asarray(r).reshape(-1)[0])
                if np.isfinite(rr):
                    radii.append(rr)
            radii = sorted(set(radii))
            raise KeyError(f"{gridder_file}: no Kernel_* group with KernelRadius={r_target:g} (have {radii})")
        return np.asarray(best["GridPointOverDensities"], dtype=np.float64).ravel()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot 2D histogram of log10(1+delta) at two kernel radii from a gridder file."
    )
    ap.add_argument("--gridder-file", type=Path, default=DEFAULT_GRIDDER_FILE, help="Gridder output HDF5 file.")
    ap.add_argument("--kernel-radius-1", type=float, default=None, help="First kernel radius (Mpc).")
    ap.add_argument("--kernel-radius-2", type=float, default=None, help="Second kernel radius (Mpc).")
    ap.add_argument("--nbins", type=int, default=160, help="Number of bins per axis for the 2D histogram.")
    ap.add_argument("--clip-min", type=float, default=-1.0 + 1e-12, help="Clip delta below this before log10(1+delta).")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: ../plots/overdensity_2d_R<R1>_R<R2>.png).",
    )
    args = ap.parse_args()

    if args.kernel_radius_1 is None or args.kernel_radius_2 is None:
        radii = _list_kernel_radii(args.gridder_file)
        if len(radii) < 2:
            raise ValueError(
                "Could not infer two kernel radii from the gridder file. "
                "Provide both --kernel-radius-1 and --kernel-radius-2, and ensure KernelRadius attrs exist."
            )
        r1, r2 = float(radii[0]), float(radii[1])
        print(f"inferred_kernel_radii {r1:g} {r2:g}")
    else:
        r1 = float(args.kernel_radius_1)
        r2 = float(args.kernel_radius_2)
    if r2 < r1:
        r1, r2 = r2, r1

    d1 = _load_overdensity_for_radius(args.gridder_file, kernel_radius=r1)
    d2 = _load_overdensity_for_radius(args.gridder_file, kernel_radius=r2)
    if d1.size != d2.size:
        raise ValueError(f"Kernel grids have different lengths: {d1.size} vs {d2.size}")

    clip_min = float(args.clip_min)
    m = np.isfinite(d1) & np.isfinite(d2) & (d1 > clip_min) & (d2 > clip_min)
    d1 = np.clip(d1[m], clip_min, None)
    d2 = np.clip(d2[m], clip_min, None)
    if d1.size == 0:
        raise RuntimeError("No valid overdensity samples after filtering/clipping.")

    x1 = np.log10(1.0 + d1)
    x2 = np.log10(1.0 + d2)

    nbins = int(args.nbins)
    H, xedges, yedges = np.histogram2d(x1, x2, bins=nbins)

    corr = float(np.corrcoef(x1, x2)[0, 1]) if x1.size >= 2 else float("nan")
    print(f"N={x1.size:,}  corr(log10(1+δ_R{r1:g}), log10(1+δ_R{r2:g})) = {corr:.6g}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    out = args.out
    if out is None:
        outdir = Path(__file__).resolve().parent.parent / "plots"
        outdir.mkdir(parents=True, exist_ok=True)
        out = outdir / f"overdensity_2d_R{r1:g}_R{r2:g}.png"
    else:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    vmax = float(np.max(H)) if H.size else 1.0
    norm = LogNorm(vmin=1.0, vmax=max(vmax, 1.0))
    im = ax.pcolormesh(xedges, yedges, H.T, shading="auto", norm=norm, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Count per bin")

    ax.set_xlabel(rf"$\log_{{10}}(1+\delta_{{R={r1:g}}})$")
    ax.set_ylabel(rf"$\log_{{10}}(1+\delta_{{R={r2:g}}})$")
    ax.set_title(f"Overdensity 2D histogram (N={x1.size:,}, corr={corr:.3f})")
    ax.grid(False)

    fig.savefig(out, dpi=200)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
