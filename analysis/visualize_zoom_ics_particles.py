#!/usr/bin/env python
"""
Visualise the particle distribution directly in a SWIFT-format zoom IC file.

Plots XY / XZ / YZ projections of particles within a region centered on the zoom
centre, with particles coloured by their mass (coarse vs high-res).

Typical usage (zoom region + 10 Mpc padding):
  python analysis/visualize_zoom_ics_particles.py \\
    --ics /snap7/scratch/dp276/dc-love2/tessera/zooms/0000/zoom_ICS_0000.hdf5 \\
    --music2-conf /snap7/scratch/dp276/dc-love2/tessera/zooms/0000/music2_zoom.conf \\
    --pad 10 \\
    --max-per-mass 50000 \\
    --out zoom_ics_particles_0000.png
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _parse_csv_floats(text: str, n: int) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != n:
        raise ValueError(f"Expected {n} comma-separated floats, got: {text!r}")
    return np.array([float(p) for p in parts], dtype=np.float64)


def read_zoom_region_music2(cfg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(cfg_path)
    center = _parse_csv_floats(cfg["setup"]["ref_center"], 3)
    extent = _parse_csv_floats(cfg["setup"]["ref_extent"], 3)
    return center, extent


def read_boxsize_mpc(f: h5py.File) -> float:
    box_attr = np.array(f["Header"].attrs["BoxSize"])
    if box_attr.ndim == 0:
        return float(box_attr)
    return float(box_attr.reshape(-1)[0])


def read_const_mass_from_masstable(f: h5py.File, part_key: str) -> float | None:
    """
    If a PartType group does not have a Masses dataset, SWIFT/Gadget-style ICs may
    specify a constant particle mass via Header/MassTable.
    """
    try:
        idx = int(str(part_key).replace("PartType", ""))
    except ValueError:
        return None
    header = f.get("Header")
    if header is None or "MassTable" not in header.attrs:
        return None
    mt = np.array(header.attrs["MassTable"]).reshape(-1)
    if idx < 0 or idx >= mt.size:
        return None
    m = float(mt[idx])
    return m if m > 0 else None


def unwrap_relative(pos: np.ndarray, center: np.ndarray, box: float) -> np.ndarray:
    half = 0.5 * float(box)
    delta = pos - center[None, :]
    delta = (delta + half) % float(box) - half
    return delta


def wrap_delta(delta: np.ndarray, box: float) -> np.ndarray:
    half = 0.5 * float(box)
    d = np.asarray(delta, dtype=np.float64)
    return (d + half) % float(box) - half


def fractions_to_mpc(val: np.ndarray, box: float) -> np.ndarray:
    v = np.asarray(val, dtype=np.float64)
    return v * float(box)


def _format_vec(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=np.float64)
    return "[" + ", ".join(f"{x: .6f}" for x in v.tolist()) + "]"


@dataclass
class RunningExtents:
    mins: np.ndarray
    maxs: np.ndarray
    count: int = 0

    @classmethod
    def empty(cls) -> "RunningExtents":
        return cls(
            mins=np.array([np.inf, np.inf, np.inf], dtype=np.float64),
            maxs=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64),
            count=0,
        )

    def update(self, pts: np.ndarray) -> None:
        if pts.size == 0:
            return
        pts = np.asarray(pts, dtype=np.float64)
        self.mins = np.minimum(self.mins, np.min(pts, axis=0))
        self.maxs = np.maximum(self.maxs, np.max(pts, axis=0))
        self.count += int(pts.shape[0])


def compute_parttype1_bbox_center(
    f: h5py.File,
    *,
    box: float,
    center_guess: np.ndarray,
    chunk: int,
) -> tuple[np.ndarray, RunningExtents]:
    """
    Compute an (approx) centre for the PartType1 distribution as the centre of its bbox,
    in a way that is robust to periodic boundaries.

    Works by wrapping relative to `center_guess`, finding the bbox in that wrapped frame,
    then mapping the bbox centre back into absolute coordinates.
    """
    if "PartType1" not in f or "Coordinates" not in f["PartType1"]:
        raise RuntimeError("IC file missing PartType1/Coordinates; cannot compute bbox center.")
    ds = f["PartType1/Coordinates"]
    n = int(ds.shape[0])
    ext = RunningExtents.empty()
    for start in range(0, n, int(chunk)):
        stop = min(start + int(chunk), n)
        coords = np.array(ds[start:stop], dtype=np.float64)
        rel = unwrap_relative(coords, center_guess, box)
        ext.update(rel)
    if ext.count == 0:
        raise RuntimeError("PartType1 appears empty; cannot compute bbox center.")
    center_rel = 0.5 * (ext.mins + ext.maxs)
    center_abs = (np.asarray(center_guess, dtype=np.float64) + center_rel) % float(box)
    return center_abs, ext


@dataclass
class Reservoir:
    k: int
    seen: int = 0
    pts: np.ndarray | None = None  # shape (<=k, 3)

    def update(self, rng: np.random.Generator, new_pts: np.ndarray) -> None:
        if new_pts.size == 0:
            return
        new_pts = np.asarray(new_pts, dtype=np.float64)
        if new_pts.ndim != 2 or new_pts.shape[1] != 3:
            raise ValueError("Expected new_pts with shape (N, 3)")

        if self.pts is None:
            self.pts = np.empty((0, 3), dtype=np.float64)

        r = int(self.pts.shape[0])
        n = int(new_pts.shape[0])

        # Fill reservoir if not yet full.
        if r < self.k:
            take = min(self.k - r, n)
            if take:
                self.pts = np.vstack([self.pts, new_pts[:take]])
                r += take
                self.seen += take
                new_pts = new_pts[take:]
                n -= take

        if n <= 0:
            return

        # Vectorized reservoir updates for the remaining points (sequentially equivalent).
        # For each new item t=1..n, draw j ~ Uniform[0, seen+t-1].
        t = np.arange(1, n + 1, dtype=np.int64)
        j = (rng.random(n) * (self.seen + t)).astype(np.int64)
        keep = j < self.k
        if np.any(keep):
            self.pts[j[keep]] = new_pts[keep]
        self.seen += n


def plot_samples(
    out: Path,
    samples: dict[float, Reservoir],
    halfwidth: np.ndarray,
    *,
    title: str,
    total_counts: dict[float, int] | None = None,
    s: float = 0.4,
    alpha: float = 0.4,
) -> None:
    keys = sorted(samples.keys())
    plotted = [(k, samples[k].pts) for k in keys if samples[k].pts is not None and samples[k].pts.size]
    if not plotted:
        raise RuntimeError("No particles were sampled in the requested region.")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=False)
    projs = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]

    n_bins = len(plotted)
    cmap = plt.get_cmap("tab20" if n_bins <= 20 else "turbo")
    colors = [cmap(i / max(1, n_bins - 1)) for i in range(n_bins)]

    handles: list[object] = []
    labels: list[str] = []

    for ax_idx, (ax, (i, j, xi, yi)) in enumerate(zip(axes, projs)):
        for (idx, (mass, pts)) in enumerate(plotted):
            total = None if total_counts is None else total_counts.get(mass)
            total_txt = "" if total is None else f"total={total:,}, "
            label = f"m={mass:.6g} ({total_txt}inwin={samples[mass].seen:,}, plot={pts.shape[0]:,})"

            sc = ax.scatter(
                pts[:, i],
                pts[:, j],
                s=s,
                alpha=alpha,
                color=colors[idx],
                rasterized=True,
                label=label,
            )
            if ax_idx == 0:
                handles.append(sc)
                labels.append(label)
        ax.set_xlim(-halfwidth[i], halfwidth[i])
        ax.set_ylim(-halfwidth[j], halfwidth[j])
        ax.set_xlabel(f"{xi} - center [Mpc]")
        ax.set_ylabel(f"{yi} - center [Mpc]")
        ax.set_aspect("equal")

    fig.suptitle(title, fontsize=11, y=0.98)
    bottom = 0.08
    if handles:
        ncol = 1 if len(handles) <= 6 else 2
        bottom = 0.12 if ncol == 1 else 0.18
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            fontsize=8,
            frameon=False,
            ncol=ncol,
        )
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.92))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise particles in a zoom IC file, coloured by mass.")
    ap.add_argument("--ics", type=Path, required=True, help="SWIFT-format IC HDF5 (e.g. zoom_ICS_0000.hdf5).")
    ap.add_argument(
        "--music2-conf",
        type=Path,
        help="MUSIC2 zoom config to define the zoom region (uses ref_center/ref_extent).",
    )
    ap.add_argument(
        "--center-mode",
        choices=("pt1bbox", "boxmid", "config"),
        default="pt1bbox",
        help=(
            "When using --music2-conf, choose how to center the plot window. "
            "`config` uses ref_center*BoxSize, `boxmid` uses BoxSize/2, and `pt1bbox` centers on the PartType1 bbox."
        ),
    )
    ap.add_argument(
        "--print-extents",
        action="store_true",
        help="Print bounding box statistics for PartType1 (and counts in/out of the zoom region).",
    )
    ap.add_argument(
        "--center",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Region centre in Mpc (used if --music2-conf is not provided).",
    )
    ap.add_argument(
        "--extent",
        type=float,
        nargs=3,
        metavar=("DX", "DY", "DZ"),
        help="Full region extent in Mpc (used if --music2-conf is not provided).",
    )
    ap.add_argument("--pad", type=float, default=10.0, help="Padding [Mpc] added on each side of the zoom extent.")
    ap.add_argument("--max-per-mass", type=int, default=50000, help="Max points to plot per distinct mass.")
    ap.add_argument("--chunk", type=int, default=1_000_000, help="Chunk size when scanning particles.")
    ap.add_argument(
        "--scan-fraction",
        type=float,
        default=1.0,
        help="Random fraction of chunks to scan (use <1 for faster approximate plots).",
    )
    ap.add_argument(
        "--types",
        type=int,
        nargs="*",
        default=None,
        help="Particle types to include (e.g. 1 2). Default: all PartType* groups present.",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for subsampling.")
    ap.add_argument("--out", type=Path, default=Path("zoom_ics_particles.png"), help="Output PNG.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.ics, "r") as f:
        box = read_boxsize_mpc(f)

        if args.music2_conf:
            ref_center, ref_extent = read_zoom_region_music2(args.music2_conf)
            # In MUSIC2 configs, these are fractions of the coordinate box.
            center_conf_mpc = fractions_to_mpc(ref_center, box)
            extent_mpc = fractions_to_mpc(ref_extent, box)
            center_boxmid_mpc = np.array([0.5 * box, 0.5 * box, 0.5 * box], dtype=np.float64)
            if args.center_mode == "config":
                center_mpc = center_conf_mpc
                pt1_bbox_ext_rel = None
            elif args.center_mode == "boxmid":
                center_mpc = center_boxmid_mpc
                pt1_bbox_ext_rel = None
            else:
                center_mpc, pt1_bbox_ext_rel = compute_parttype1_bbox_center(
                    f,
                    box=box,
                    center_guess=center_boxmid_mpc,
                    chunk=int(args.chunk),
                )
        else:
            if args.center is None:
                raise ValueError("Provide --music2-conf or --center.")
            center_mpc = np.array(args.center, dtype=np.float64)
            if args.extent is None:
                raise ValueError("When using --center, also provide --extent (full width in Mpc).")
            extent_mpc = np.array(args.extent, dtype=np.float64)

        zoom_halfwidth = 0.5 * extent_mpc
        halfwidth = 0.5 * extent_mpc + float(args.pad)
        halfwidth = np.asarray(halfwidth, dtype=np.float64)

        print(f"ICs: {args.ics}")
        print(f"BoxSize: {box:.6f} Mpc")
        if args.music2_conf:
            print(f"MUSIC2 conf: {args.music2_conf}")
            print(f"ref_center (frac): {_format_vec(ref_center)}")
            print(f"ref_extent (frac): {_format_vec(ref_extent)}")
            print(f"Config center (Mpc): {_format_vec(center_conf_mpc)}")
            print(f"Center mode: {args.center_mode} -> using center (Mpc): {_format_vec(center_mpc)}")
            if args.center_mode == "boxmid":
                shift = wrap_delta(center_boxmid_mpc - center_conf_mpc, box)
                print(f"Implied MUSIC2 recenter shift to box-mid (Mpc): {_format_vec(shift)}")
            elif args.center_mode == "pt1bbox":
                shift = wrap_delta(center_mpc - center_conf_mpc, box)
                print(f"Implied MUSIC2 recenter/align shift to PartType1 bbox (Mpc): {_format_vec(shift)}")
        else:
            print(f"Center (Mpc):  {_format_vec(center_mpc)}")
        print(f"Zoom extent (Mpc):  {_format_vec(extent_mpc)}")
        print(f"Zoom halfwidth:     {_format_vec(zoom_halfwidth)}")
        print(f"Plot halfwidth(+pad): {_format_vec(halfwidth)} (pad={args.pad:g} Mpc)")
        if args.scan_fraction < 1.0:
            print(f"NOTE: --scan-fraction={args.scan_fraction:g} => counts/extents are approximate.")

        # Collect reservoirs per (rounded) mass.
        reservoirs: dict[float, Reservoir] = {}
        mass_totals_all: dict[float, int] = {}
        mass_totals_by_type: dict[str, dict[float, int]] = {}

        part_keys = [k for k in f.keys() if k.startswith("PartType")]
        if args.types is not None:
            want = {f"PartType{t}" for t in args.types}
            part_keys = [k for k in part_keys if k in want]
        part_keys = sorted(part_keys)
        if not part_keys:
            raise RuntimeError("No PartType* groups found to plot.")

        # Stats for PartType1 (typically the zoom DM particles in SWIFT-format MUSIC2 outputs).
        pt1_rel_ext = RunningExtents.empty()
        pt1_abs_ext = RunningExtents.empty()
        pt1_in_zoom = 0
        pt1_in_pad = 0
        pt1_total = 0

        for pkey in part_keys:
            g = f[pkey]
            if "Coordinates" not in g:
                continue
            coords_ds = g["Coordinates"]
            masses_ds = g["Masses"] if "Masses" in g else None
            const_mass = read_const_mass_from_masstable(f, pkey)
            has_variable_masses = masses_ds is not None and const_mass is None
            n = int(coords_ds.shape[0])
            if n == 0:
                continue

            print(f"{pkey}: n={n:,}")
            if pkey == "PartType1":
                pt1_total = n

            if const_mass is not None:
                mkey = float(np.round(const_mass, decimals=12))
                per_type = mass_totals_by_type.setdefault(pkey, {})
                per_type[mkey] = per_type.get(mkey, 0) + n
                mass_totals_all[mkey] = mass_totals_all.get(mkey, 0) + n

            for start in range(0, n, int(args.chunk)):
                stop = min(start + int(args.chunk), n)
                masses_chunk = None
                if has_variable_masses:
                    masses_chunk = np.array(masses_ds[start:stop], dtype=np.float64)
                    keys_all = np.round(masses_chunk, decimals=12)
                    uniq_all, counts_all = np.unique(keys_all, return_counts=True)
                    per_type = mass_totals_by_type.setdefault(pkey, {})
                    for u, c in zip(uniq_all, counts_all):
                        mk = float(u)
                        cnt = int(c)
                        per_type[mk] = per_type.get(mk, 0) + cnt
                        mass_totals_all[mk] = mass_totals_all.get(mk, 0) + cnt

                if args.scan_fraction < 1.0 and rng.random() > float(args.scan_fraction):
                    continue
                coords = np.array(coords_ds[start:stop], dtype=np.float64)
                rel = unwrap_relative(coords, center_mpc, box)
                if pkey == "PartType1" and args.print_extents:
                    pt1_abs_ext.update(coords)
                    pt1_rel_ext.update(rel)
                    in_zoom = (
                        (np.abs(rel[:, 0]) <= zoom_halfwidth[0])
                        & (np.abs(rel[:, 1]) <= zoom_halfwidth[1])
                        & (np.abs(rel[:, 2]) <= zoom_halfwidth[2])
                    )
                    in_pad = (
                        (np.abs(rel[:, 0]) <= halfwidth[0])
                        & (np.abs(rel[:, 1]) <= halfwidth[1])
                        & (np.abs(rel[:, 2]) <= halfwidth[2])
                    )
                    pt1_in_zoom += int(np.sum(in_zoom))
                    pt1_in_pad += int(np.sum(in_pad))

                mask = (
                    (np.abs(rel[:, 0]) <= halfwidth[0])
                    & (np.abs(rel[:, 1]) <= halfwidth[1])
                    & (np.abs(rel[:, 2]) <= halfwidth[2])
                )
                if not np.any(mask):
                    continue
                rel_in = rel[mask]

                if const_mass is not None:
                    fill = float(const_mass)
                    masses = np.full(rel_in.shape[0], fill, dtype=np.float64)
                elif masses_ds is None:
                    masses = np.full(rel_in.shape[0], np.nan, dtype=np.float64)
                else:
                    assert masses_chunk is not None
                    masses = masses_chunk[mask]

                # Group by (rounded) mass for stable binning.
                keys = np.round(masses, decimals=12)
                uniq, inv = np.unique(keys, return_inverse=True)
                for ui, mkey in enumerate(uniq):
                    pts_m = rel_in[inv == ui]
                    mass_key = float(mkey)
                    res = reservoirs.get(mass_key)
                    if res is None:
                        res = Reservoir(k=int(args.max_per_mass))
                        reservoirs[mass_key] = res
                    res.update(rng, pts_m)

        if args.print_extents and pt1_total:
            print("\nPartType1 extents (absolute, Mpc):")
            print(f"  mins: {_format_vec(pt1_abs_ext.mins)}")
            print(f"  maxs: {_format_vec(pt1_abs_ext.maxs)}")
            print(f"  span: {_format_vec(pt1_abs_ext.maxs - pt1_abs_ext.mins)}")
            print("PartType1 extents (relative to zoom center, wrapped into [-box/2, box/2], Mpc):")
            print(f"  mins: {_format_vec(pt1_rel_ext.mins)}")
            print(f"  maxs: {_format_vec(pt1_rel_ext.maxs)}")
            print(f"  span: {_format_vec(pt1_rel_ext.maxs - pt1_rel_ext.mins)}")
            print("\nPartType1 containment counts:")
            if args.scan_fraction >= 1.0:
                print(f"  total: {pt1_total:,}")
                print(f"  in zoom box (no pad): {pt1_in_zoom:,} ({pt1_in_zoom / pt1_total:.3%})")
                print(f"  in zoom+pad box:      {pt1_in_pad:,} ({pt1_in_pad / pt1_total:.3%})")
                print(f"  outside zoom+pad:     {pt1_total - pt1_in_pad:,} ({(pt1_total - pt1_in_pad) / pt1_total:.3%})")
            else:
                print(f"  scanned PartType1 particles: {pt1_rel_ext.count:,}")
                print(f"  scanned in zoom box (no pad): {pt1_in_zoom:,}")
                print(f"  scanned in zoom+pad box:      {pt1_in_pad:,}")

        if "PartType2" in mass_totals_by_type:
            print("\nPartType2 mass totals (entire file):")
            for m, c in sorted(mass_totals_by_type["PartType2"].items(), key=lambda kv: kv[0], reverse=True):
                print(f"  m={m:.6g}: total={c:,}")

    title = f"{Path(args.ics).name} (centered)\nbox={box:.3f} Mpc, pad={args.pad:g} Mpc"
    plot_samples(args.out, reservoirs, halfwidth, title=title, total_counts=mass_totals_all)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
