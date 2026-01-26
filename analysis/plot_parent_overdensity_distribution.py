#!/usr/bin/env python
"""
Plot the overdensity distribution (from a gridder output) for the parent volume.

This script is intentionally focused on the 1-point distribution used to stratify
kernel selections in log10(1+delta) space.

Input
-----
- A gridder output HDF5 file containing:
  - `Grids/Kernel_*/GridPointOverDensities` (delta = rho/rho_bar - 1)

Output
------
- A PNG with:
  - PDF of log10(1+delta)
  - CDF of log10(1+delta)
  - Optional quantile markers and/or stratification bin edges
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _load_kernels(grid_path: Path) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    with h5py.File(grid_path, "r") as f:
        grids = f["Grids"]
        kernels: list[tuple[float | None, str]] = []
        overd_by_name: dict[str, np.ndarray] = {}
        for key in grids.keys():
            if not str(key).startswith("Kernel_"):
                continue
            grp = grids[key]
            rad = grp.attrs.get("KernelRadius")
            kernels.append((None if rad is None else float(rad), str(key)))
            overd_by_name[str(key)] = np.asarray(grp["GridPointOverDensities"], dtype=np.float64)

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

    return overd, radii


def _choose_kernel_index(*, radii: dict[int, float], kernel_index: int | None, kernel_radius: float | None) -> int:
    if kernel_index is not None and kernel_radius is not None:
        raise ValueError("Pass only one of --kernel-index or --kernel-radius.")
    if kernel_index is not None:
        return int(kernel_index)
    if kernel_radius is not None:
        if not radii:
            raise RuntimeError("Grid file does not contain KernelRadius attributes; cannot use --kernel-radius.")
        items = sorted(radii.items(), key=lambda kv: kv[0])
        r = float(kernel_radius)
        for k, rv in items:
            if np.isclose(float(rv), r, rtol=0, atol=1e-8):
                return int(k)
        raise ValueError(f"No kernel found with KernelRadius={r:g} (available: {sorted(set(radii.values()))})")
    return 0


def _quantile_lines(ax, x: np.ndarray, qs: list[float], *, color: str, label_prefix: str) -> None:
    vals = np.quantile(x, qs)
    for q, v in zip(qs, vals):
        ax.axvline(float(v), color=color, lw=1.0, ls="--", alpha=0.6)
        ax.text(float(v), 0.98, f"{label_prefix}{q:.2f}", rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot overdensity distribution (log10(1+delta)) from a gridder file.")
    ap.add_argument("--grid", type=Path, required=True, help="Gridder output HDF5.")
    ap.add_argument("--out", type=Path, default=Path("../plots/overdensity_distribution.png"), help="Output PNG.")
    ap.add_argument("--kernel-index", type=int, help="Kernel index (0-based after sorting by KernelRadius). Default: 0.")
    ap.add_argument("--kernel-radius", type=float, help="Kernel radius (Mpc) to select by KernelRadius attribute.")
    ap.add_argument("--nbins", type=int, default=120, help="Histogram bins for log10(1+delta).")
    ap.add_argument(
        "--quantiles",
        type=float,
        nargs="*",
        default=[0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99],
        help="Quantiles to mark (in [0,1]).",
    )
    ap.add_argument(
        "--strata",
        type=int,
        default=0,
        help="If >0, also draw quantile bin edges for stratified selection into N equal-count bins.",
    )
    args = ap.parse_args()

    overd, radii = _load_kernels(args.grid)
    k = _choose_kernel_index(radii=radii, kernel_index=args.kernel_index, kernel_radius=args.kernel_radius)
    if k not in overd:
        raise ValueError(f"Kernel index {k} not available (have {sorted(overd.keys())}).")

    d = np.asarray(overd[k], dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise RuntimeError("No finite overdensity values found.")
    if np.any(d <= -1):
        bad = int(np.sum(d <= -1))
        raise RuntimeError(f"Kernel {k}: found {bad} cells with delta <= -1; cannot take log10(1+delta).")

    x = np.log10(1.0 + d)
    label = f"Kernel {k}" if k not in radii else f"Kernel {k} (R={radii[k]:g} Mpc)"

    qs = [float(q) for q in args.quantiles]
    if any((q < 0) or (q > 1) for q in qs):
        raise ValueError("--quantiles must be in [0,1].")

    print(f"{args.grid}  {label}")
    print(f"N={x.size:,}  log10(1+delta) min/median/max = {float(x.min()):.4g} / {float(np.median(x)):.4g} / {float(x.max()):.4g}")
    qvals = np.quantile(x, qs)
    for q, v in zip(qs, qvals):
        print(f"  q={q:.3f}: {float(v):.6g}")
    if args.strata and int(args.strata) > 0:
        edges = np.quantile(x, np.linspace(0, 1, int(args.strata) + 1))
        print(f"Stratification bin edges (N={int(args.strata)}):")
        for i, e in enumerate(edges):
            print(f"  edge[{i:02d}] = {float(e):.6g}")

    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax_pdf.hist(x, bins=int(args.nbins), density=True, histtype="stepfilled", alpha=0.25, color="tab:blue")
    ax_pdf.hist(x, bins=int(args.nbins), density=True, histtype="step", lw=1.5, color="tab:blue", label=label)
    ax_pdf.set_xlabel("log10(1 + overdensity)")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.legend(loc="best")

    xs = np.sort(x)
    ys = (np.arange(xs.size, dtype=np.float64) + 1.0) / float(xs.size)
    ax_cdf.plot(xs, ys, color="tab:blue", lw=1.5)
    ax_cdf.set_xlabel("log10(1 + overdensity)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.grid(True, alpha=0.2)

    _quantile_lines(ax_pdf, x, qs, color="k", label_prefix="q=")
    _quantile_lines(ax_cdf, x, qs, color="k", label_prefix="q=")

    if args.strata and int(args.strata) > 0:
        edges = np.quantile(x, np.linspace(0, 1, int(args.strata) + 1))
        for e in edges[1:-1]:
            ax_pdf.axvline(float(e), color="tab:orange", lw=0.8, ls=":", alpha=0.6)
            ax_cdf.axvline(float(e), color="tab:orange", lw=0.8, ls=":", alpha=0.6)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
