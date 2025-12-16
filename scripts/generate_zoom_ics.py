#!/usr/bin/env python
"""
Generate a MUSIC2 zoom configuration for the Tessera parent box from a kernel selection.

This is a Tessera-focused copy of `scripts/estimate_zoom_mass.py`:
  - Defaults to the Tessera parent config (`tessera/music2_parent_tessera.conf`)
  - Writes a MUSIC2 zoom config (ref_center/ref_extent + levels + seeds)

Inputs:
  - A gridder output to choose a center based on a kernel overdensity.
  - A z=0 snapshot with ParticleIDs to select particles within the chosen kernel.
  - The parent IC file (ParticleIDs + Masses) to trace those particles back to ICs.
"""

import argparse
import configparser
import sys
from pathlib import Path
import shutil
import atexit

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# Prefer the shared utilities in pmwd_zoom_selection.
PMWD_SCRIPTS_DIR = Path("/cosma7/data/dp004/dc-love2/codes/pmwd_zoom_selection/scripts")
if not PMWD_SCRIPTS_DIR.exists():
    PMWD_SCRIPTS_DIR = (ROOT.parent / "pmwd_zoom_selection" / "scripts").resolve()
if not PMWD_SCRIPTS_DIR.exists():
    raise FileNotFoundError(
        "Could not find pmwd_zoom_selection scripts directory; tried "
        f"{PMWD_SCRIPTS_DIR!s} and /cosma7/data/dp004/dc-love2/codes/pmwd_zoom_selection/scripts"
    )
if str(PMWD_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PMWD_SCRIPTS_DIR))

from utilities import (  # noqa: E402
    bounding_box,
    choose_center,
    kernel_radius_from_gridder,
    load_gridder,
    load_snap,
    match_into_ic,
    select_sphere,
    unwrap_relative,
)

DEFAULT_ZOOM_BASE = Path("/snap7/scratch/dp276/dc-love2/tessera/zooms")

from methods import (  # noqa: E402
    _Tee,
    _parse_float,
    _parse_int,
    fof_haloes_relative,
    infer_contiguous_id_offset,
    infer_id_offset,
    load_fof_catalogue,
    next_zoom_index,
    positions_for_ids_by_offset,
    positions_for_ids_by_scan,
    sample_box_particles,
    write_music_zoom_config,
    write_swift_zoom_yaml,
)


def plot_selection(
    out_png: Path,
    sel_ic: np.ndarray,
    box_ic: np.ndarray,
    env_ic: np.ndarray,
    sel_z0: np.ndarray,
    box_z0: np.ndarray,
    env_z0: np.ndarray,
    halo_ic: np.ndarray | None,
    mins_raw: np.ndarray | None,
    maxs_raw: np.ndarray | None,
    mins: np.ndarray,
    maxs: np.ndarray,
    slab_half: float,
    env_half: float,
    kernel_radius: float | None = None,
    kernel_boundary_L: float | None = None,
    halo_rel: np.ndarray | None = None,
    halo_masses: np.ndarray | None = None,
    halo_failed_rel: np.ndarray | None = None,
    halo_failed_masses: np.ndarray | None = None,
    halo_mmin: float = 1e14,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    planes = [(0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z")]
    slab_axes = [2, 1, 0]  # depth axis for XY, XZ, YZ
    plot_fov = 75.0  # Mpc, symmetric limits around zoom center

    def draw_bbox(ax, i, j):
        rect = np.array(
            [
                [mins[i], mins[j]],
                [maxs[i], mins[j]],
                [maxs[i], maxs[j]],
                [mins[i], maxs[j]],
                [mins[i], mins[j]],
            ]
        )
        ax.plot(rect[:, 0], rect[:, 1], color="tab:red", lw=1)
        if mins_raw is not None and maxs_raw is not None:
            rect_raw = np.array(
                [
                    [mins_raw[i], mins_raw[j]],
                    [maxs_raw[i], mins_raw[j]],
                    [maxs_raw[i], maxs_raw[j]],
                    [mins_raw[i], maxs_raw[j]],
                    [mins_raw[i], mins_raw[j]],
                ]
            )
            ax.plot(rect_raw[:, 0], rect_raw[:, 1], color="k", lw=0.8, ls="--", alpha=0.8)

    def draw_row(
        row_axes,
        title,
        sel_pts,
        box_pts,
        env_pts,
        *,
        halo_ic_pts: np.ndarray | None,
        draw_box: bool,
        draw_haloes: bool = True,
    ):
        # Keep the view focused around the zoom region (requested), while ensuring the IC bbox fits.
        bbox_fov = float(np.max(np.abs(np.vstack([mins, maxs])))) * 1.05
        fov = max(plot_fov, bbox_fov)
        for proj, (ax, (i, j, xi, yi)) in enumerate(zip(row_axes, planes)):
            k = slab_axes[proj]
            env_mask = np.abs(env_pts[:, k]) <= slab_half
            ax.scatter(env_pts[env_mask, i], env_pts[env_mask, j], s=0.3, alpha=0.1, color="gray", rasterized=True)
            ax.scatter(box_pts[:, i], box_pts[:, j], s=0.5, alpha=0.15, color="tab:blue", rasterized=True)
            ax.scatter(sel_pts[:, i], sel_pts[:, j], s=2.0, alpha=0.7, color="tab:orange", rasterized=True)
            if halo_ic_pts is not None and halo_ic_pts.size:
                ax.scatter(
                    halo_ic_pts[:, i],
                    halo_ic_pts[:, j],
                    s=0.8,
                    alpha=0.5,
                    color="tab:green",
                    rasterized=True,
                )
            if draw_haloes and halo_rel is not None and halo_masses is not None:
                hm = np.asarray(halo_masses, dtype=np.float64)
                hr = np.asarray(halo_rel, dtype=np.float64)
                mcut = hm >= halo_mmin
                if np.any(mcut):
                    # slab cut for haloes too (by the same depth axis)
                    hslab = mcut & (np.abs(hr[:, k]) <= slab_half)
                    if np.any(hslab):
                        logm = np.log10(hm[hslab])
                        lo, hi = np.percentile(logm, [5, 99.5]) if logm.size > 5 else (logm.min(), logm.max())
                        lo = float(lo)
                        hi = float(hi if hi > lo else lo + 1e-6)
                        sizes = 10.0 + 150.0 * (np.clip(logm, lo, hi) - lo) / (hi - lo)
                        ax.scatter(
                            hr[hslab, i],
                            hr[hslab, j],
                            s=sizes,
                            alpha=0.35,
                            color="tab:purple",
                            rasterized=True,
                        )
            # Overlay “failed” haloes (subset) on top for debugging.
            if draw_haloes and halo_failed_rel is not None and halo_failed_masses is not None:
                fh = np.asarray(halo_failed_masses, dtype=np.float64)
                fr = np.asarray(halo_failed_rel, dtype=np.float64)
                # Do NOT slab-cut failed haloes: the failure is a 3D containment issue and the
                # halo centre can project misleadingly; always show them in every projection.
                if fr.size:
                    flogm = np.log10(np.maximum(fh, 1.0))
                    flo, fhi = np.percentile(flogm, [5, 99.5]) if flogm.size > 5 else (flogm.min(), flogm.max())
                    flo = float(flo)
                    fhi = float(fhi if fhi > flo else flo + 1e-6)
                    fsizes = 30.0 + 220.0 * (np.clip(flogm, flo, fhi) - flo) / (fhi - flo)
                    ax.scatter(
                        fr[:, i],
                        fr[:, j],
                        s=fsizes,
                        alpha=0.9,
                        color="tab:green",
                        marker="x",
                        linewidths=1.2,
                        rasterized=True,
                    )
            # Draw the kernel radius (and boundary band) in z=0 panels for visual debugging.
            if draw_haloes and kernel_radius is not None:
                import matplotlib.patches as mpatches

                circ = mpatches.Circle((0.0, 0.0), float(kernel_radius), fill=False, lw=1.0, ls="--", color="k", alpha=0.7)
                ax.add_patch(circ)
                if kernel_boundary_L is not None and float(kernel_boundary_L) > 0:
                    r_in = max(0.0, float(kernel_radius) - float(kernel_boundary_L))
                    r_out = float(kernel_radius) + float(kernel_boundary_L)
                    ax.add_patch(mpatches.Circle((0.0, 0.0), r_in, fill=False, lw=0.8, ls=":", color="k", alpha=0.5))
                    ax.add_patch(mpatches.Circle((0.0, 0.0), r_out, fill=False, lw=0.8, ls=":", color="k", alpha=0.5))
            if draw_box:
                draw_bbox(ax, i, j)
            ax.set_xlim(-fov, fov)
            ax.set_ylim(-fov, fov)
            ax.set_xlabel(f"{xi} - center")
            ax.set_ylabel(f"{yi} - center")
            ax.set_aspect("equal")
        row_axes[0].set_title(title)

    draw_row(
        axes[0, :],
        "ICs (bbox in red, environment slab in gray)",
        sel_ic,
        box_ic,
        env_ic,
        halo_ic_pts=halo_ic,
        draw_box=True,
        draw_haloes=False,
    )
    draw_row(
        axes[1, :],
        "z=0 (bbox in red, environment slab in gray)",
        sel_z0,
        box_z0,
        env_z0,
        halo_ic_pts=None,
        draw_box=False,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def halo_fully_selected_in_bbox(
    *,
    ids_sim: np.ndarray,
    pos_sim: np.ndarray,
    box_sim: float,
    halo_center: np.ndarray,
    halo_radius: float,
    ic_pos: np.ndarray,
    ic_id_offset: int,
    box_ic: float,
    zoom_center: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
) -> bool:
    """
    Check that *all* particles within a halo radius at z=0 are inside the IC-space bbox.

    This is a pragmatic proxy for “halo fully selected”: if any halo particle maps to an IC
    coordinate outside the refinement bounding box, then the zoom will split that halo.
    """
    # NOTE: `pos_sim` can be 100M+ particles; doing `pos_sim - halo_center` in one shot
    # allocates multi-GB temporaries and can appear to “hang”. Chunk and early-exit instead.
    radius2 = float(halo_radius) ** 2
    half = 0.5 * float(box_sim)

    n = int(pos_sim.shape[0])
    chunk = 2_000_000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        pos = pos_sim[start:stop]
        ids = ids_sim[start:stop].astype(np.int64, copy=False)

        delta = pos - halo_center
        delta = (delta + half) % float(box_sim) - half
        dist2 = np.einsum("ij,ij->i", delta, delta)
        in_halo = dist2 <= radius2
        if not np.any(in_halo):
            continue

        halo_ids = ids[in_halo]
        idx = halo_ids - int(ic_id_offset)
        if np.any((idx < 0) | (idx >= ic_pos.shape[0])):
            return False

        ic_halo_pos = ic_pos[idx]
        ic_rel = unwrap_relative(ic_halo_pos, zoom_center, box_ic)
        inside = (
            (ic_rel[:, 0] >= mins[0]) & (ic_rel[:, 0] <= maxs[0]) &
            (ic_rel[:, 1] >= mins[1]) & (ic_rel[:, 1] <= maxs[1]) &
            (ic_rel[:, 2] >= mins[2]) & (ic_rel[:, 2] <= maxs[2])
        )
        if not bool(np.all(inside)):
            return False

    return True


def halo_ic_positions_for_plot(
    *,
    ids_sim: np.ndarray,
    pos_sim: np.ndarray,
    box_sim: float,
    halo_center: np.ndarray,
    halo_radius: float,
    ic_pos: np.ndarray,
    ic_id_offset: int,
    box_ic: float,
    zoom_center: np.ndarray,
    max_samples: int = 50_000,
    rng_seed: int = 0,
) -> np.ndarray:
    """
    Sample IC-relative positions of particles within a halo radius at z=0 (for plotting).

    Uses reservoir sampling over halo particles; avoids allocating full-size temporaries.
    """
    radius2 = float(halo_radius) ** 2
    half = 0.5 * float(box_sim)
    rng = np.random.default_rng(rng_seed)

    n = int(pos_sim.shape[0])
    chunk = 2_000_000
    samples: list[np.ndarray] = []
    seen = 0

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        pos = pos_sim[start:stop]
        ids = ids_sim[start:stop].astype(np.int64, copy=False)

        delta = pos - halo_center
        delta = (delta + half) % float(box_sim) - half
        dist2 = np.einsum("ij,ij->i", delta, delta)
        in_halo = dist2 <= radius2
        if not np.any(in_halo):
            continue

        halo_ids = ids[in_halo]
        idx = halo_ids - int(ic_id_offset)
        ok = (idx >= 0) & (idx < ic_pos.shape[0])
        if not np.any(ok):
            continue
        idx = idx[ok]

        ic_halo_pos = ic_pos[idx]
        ic_rel = unwrap_relative(ic_halo_pos, zoom_center, box_ic)

        for p in ic_rel:
            seen += 1
            if len(samples) < max_samples:
                samples.append(p)
            else:
                j = int(rng.integers(0, seen))
                if j < max_samples:
                    samples[j] = p

    return np.array(samples, dtype=np.float64)


def make_debug_plot(
    plot_path: Path,
    *,
    center: np.ndarray,
    box_sim: float,
    box_ic: float,
    mins_raw: np.ndarray | None,
    maxs_raw: np.ndarray | None,
    mins: np.ndarray,
    maxs: np.ndarray,
    sel_pos: np.ndarray,
    pos_ic_rel: np.ndarray,
    ic_path: Path,
    snap_path: Path,
    fof_path: Path | None,
    halo_mmin: float,
    kernel_radius: float | None = None,
    kernel_boundary_L: float | None = None,
    failed_halo_centres: np.ndarray | None = None,
    failed_halo_masses: np.ndarray | None = None,
    halo_ic: np.ndarray | None = None,
) -> None:
    rng = np.random.default_rng(0)
    max_sel_plot = 50000
    max_box_plot = 200000
    max_env_plot = 300000

    sel_ic_plot = pos_ic_rel
    sel_z0_plot = unwrap_relative(sel_pos, center, box_sim)
    if sel_ic_plot.shape[0] > max_sel_plot:
        keep = rng.choice(sel_ic_plot.shape[0], size=max_sel_plot, replace=False)
        sel_ic_plot = sel_ic_plot[keep]
        sel_z0_plot = sel_z0_plot[keep]

    box_ids, box_ic_plot, _box_ic_size, _n_box, _m_box = sample_box_particles(
        ic_path=ic_path,
        center=center,
        mins=mins,
        maxs=maxs,
        max_samples=max_box_plot,
        seed=0,
    )
    if box_ids.size == 0:
        raise RuntimeError("No particles found inside the IC-space bounding box to plot.")

    extents = maxs - mins
    zoom_diameter = float(np.max(extents))
    slab_half = 0.5 * zoom_diameter
    env_half = 2.0 * zoom_diameter

    def sample_environment(path: Path, box_size: float) -> np.ndarray:
        import h5py

        env: list[np.ndarray] = []
        seen = 0
        chunk = 2_000_000
        with h5py.File(path, "r") as f:
            pos_ds = f["PartType1/Coordinates"]
            n = int(pos_ds.shape[0])
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                pos = np.array(pos_ds[start:stop], dtype=np.float64)
                rel = unwrap_relative(pos, center, box_size)
                mask = np.all(np.abs(rel) <= env_half, axis=1)
                if not np.any(mask):
                    continue
                rel = rel[mask]
                for rpos in rel:
                    seen += 1
                    if len(env) < max_env_plot:
                        env.append(rpos)
                    else:
                        j = rng.integers(0, seen)
                        if j < max_env_plot:
                            env[j] = rpos
        return np.array(env, dtype=np.float64)

    env_ic_plot = sample_environment(ic_path, box_ic)
    env_z0_plot = sample_environment(snap_path, box_sim)

    try:
        off_z0, _ = infer_id_offset(snap_path)
        box_z0_plot = positions_for_ids_by_offset(
            snap_path=snap_path,
            ids=box_ids,
            offset=off_z0,
            center=center,
            box_size=box_sim,
        )
    except Exception as exc:
        print(f"Falling back to ParticleID scan for z=0 snapshot mapping: {exc}")
        box_z0_plot = positions_for_ids_by_scan(
            snap_path=snap_path,
            ids=box_ids,
            center=center,
            box_size=box_sim,
        )

    halo_rel = halo_masses = None
    failed_rel = failed_m = None
    if fof_path:
        halo_rel, halo_masses = fof_haloes_relative(fof_path, center=center)
    if failed_halo_centres is not None and failed_halo_masses is not None and failed_halo_centres.size:
        # Use the same box size as the z=0 snapshot for relative positioning.
        failed_rel = unwrap_relative(np.asarray(failed_halo_centres, dtype=np.float64), center, box_sim)
        failed_m = np.asarray(failed_halo_masses, dtype=np.float64)

    plot_selection(
        out_png=plot_path,
        sel_ic=sel_ic_plot,
        box_ic=box_ic_plot,
        env_ic=env_ic_plot,
        sel_z0=sel_z0_plot,
        box_z0=box_z0_plot,
        env_z0=env_z0_plot,
        halo_ic=halo_ic,
        mins_raw=mins_raw,
        maxs_raw=maxs_raw,
        mins=mins,
        maxs=maxs,
        slab_half=slab_half,
        env_half=env_half,
        kernel_radius=kernel_radius,
        kernel_boundary_L=kernel_boundary_L,
        halo_rel=halo_rel,
        halo_masses=halo_masses,
        halo_failed_rel=failed_rel,
        halo_failed_masses=failed_m,
        halo_mmin=halo_mmin,
    )


def main():
    ap = argparse.ArgumentParser(description="Generate MUSIC2 zoom config for Tessera from a kernel selection.")
    ap.add_argument("--grid", type=Path, required=True, help="Gridder output HDF5 (positions + overdensities).")
    ap.add_argument("--snap", type=Path, required=True, help="z=0 snapshot with ParticleIDs (pmwd or SWIFT).")
    ap.add_argument("--ic-snap", type=Path, required=True, help="Parent IC snapshot with ParticleIDs and Masses.")
    ap.add_argument("--kernel-index", type=int, default=0, help="Kernel index (0-based).")
    ap.add_argument("--kernel-radius", type=float, help="Kernel radius (Mpc) override.")
    ap.add_argument(
        "--target-logdelta",
        type=float,
        help="Target log10(1+overdensity); pick closest grid point in that space (default: max).",
    )
    ap.add_argument("--target-rank", type=int, default=0, help="Nth closest match to target overdensity (0=closest).")
    ap.add_argument("--max-tries", type=int, default=25, help="Max rank increments to try when rejecting regions.")

    ap.add_argument(
        "--out-base",
        type=Path,
        default=DEFAULT_ZOOM_BASE,
        help="Base output directory; creates per-index subfolders here (0000, 0001, ...).",
    )
    ap.add_argument("--index", type=int, help="Zoom index (default: next available under --out-base).")

    ap.add_argument(
        "--parent-config",
        type=Path,
        default=ROOT / "tessera" / "music2_parent_tessera.conf",
        help="Tessera parent MUSIC2 config.",
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=ROOT / "configs" / "music2_zoom.conf",
        help="MUSIC2 zoom template.",
    )
    ap.add_argument("--out-config", type=Path, help="Output MUSIC2 zoom config path (default: <out-base>/<idx>/music2_zoom.conf).")
    ap.add_argument("--levelmin", type=int, help="Zoom levelmin (defaults to parent levelmin).")
    ap.add_argument("--levelmax", type=int, required=True, help="Zoom levelmax (finest resolution).")
    ap.add_argument("--out-ics", type=Path, help="Output ICs path in MUSIC2 config (default: <out-base>/<idx>/zoom_ICS_<idx>.hdf5).")
    ap.add_argument("--plot-out", type=Path, help="PNG plot path (default: <out-base>/<idx>/zoom_ICs_<idx>.png).")
    ap.add_argument("--fof", type=Path, help="Optional SWIFT-style FOF catalogue (to overlay halo centres).")
    ap.add_argument("--halo-mmin", type=float, default=1e4, help="Minimum halo mass to plot (1e10 Msun).")
    ap.add_argument("--halo-boundary", type=float, default=2.5, help="Boundary buffer L (Mpc).")
    ap.add_argument(
        "--swift-template",
        type=Path,
        default=ROOT / "tessera" / "swift_zoom_params.yaml",
        help="SWIFT zoom YAML template to copy into the zoom directory.",
    )
    args = ap.parse_args()

    zoom_index = int(args.index) if args.index is not None else next_zoom_index(args.out_base)
    run_dir = args.out_base / f"{zoom_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"generate_zoom_ics_{zoom_index:04d}.txt"
    log_fh = log_path.open("w")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_fh)
    print(f"Logging to {log_path}")
    atexit.register(lambda: setattr(sys, "stdout", old_stdout))
    atexit.register(log_fh.close)

    out_config = args.out_config or (run_dir / "music2_zoom.conf")
    out_ics = args.out_ics or (run_dir / f"zoom_ICS_{zoom_index:04d}.hdf5")
    plot_out = args.plot_out or (run_dir / f"zoom_ICs_{zoom_index:04d}.png")

    parent_cfg = configparser.ConfigParser()
    parent_cfg.optionxform = str
    with args.parent_config.open() as fh:
        parent_cfg.read_file(fh)

    parent_levelmin = _parse_int(parent_cfg["setup"]["levelmin"])
    parent_box = _parse_float(parent_cfg["setup"]["boxlength"])
    omega_m = _parse_float(parent_cfg["cosmology"]["Omega_m"])
    omega_b = _parse_float(parent_cfg["cosmology"].get("Omega_b", "0.0"))
    h = _parse_float(parent_cfg["cosmology"]["H0"]) / 100.0
    base_levelmin = args.levelmin if args.levelmin is not None else parent_levelmin
    base_levelmax = args.levelmax
    if base_levelmax < base_levelmin:
        raise ValueError("levelmax cannot be smaller than levelmin")

    pos_grid, overd = load_gridder(args.grid)
    radius = kernel_radius_from_gridder(args.grid, args.kernel_index, args.kernel_radius)

    # Choose centres in log10(1+delta) space (or max delta if no target provided).
    overd_for_choice = {}
    for k, d in overd.items():
        if np.any(d <= -1):
            bad = int(np.sum(d <= -1))
            raise RuntimeError(
                f"{args.grid}: kernel {k} has {bad} grid points with overdensity <= -1; cannot use log10(1+delta)."
            )
        overd_for_choice[k] = np.log10(1.0 + d)
    target_for_choice = None if args.target_logdelta is None else float(args.target_logdelta)

    # Iterate candidate centres by rank until halo-boundary check passes.
    chosen = None
    for attempt in range(args.max_tries):
        rank = args.target_rank + attempt
        center, val_choice, grid_idx = choose_center(
            pos_grid, overd_for_choice, args.kernel_index, target_for_choice, rank=rank
        )
        val_delta = float(overd[args.kernel_index][grid_idx])

        ids_sim, pos_sim, _, box_sim = load_snap(args.snap, include_masses=False)
        sel_ids, sel_pos = select_sphere(ids_sim, pos_sim, center, radius, box_sim)
        if sel_ids.size == 0:
            continue

        ic_ids, ic_pos, ic_masses, box_ic = load_snap(args.ic_snap, include_masses=True)
        pos_ic_rel, mass_ic_sel = match_into_ic(sel_ids, center, ic_ids, ic_pos, ic_masses, box_ic)
        mins_raw, maxs_raw, volume_raw = bounding_box(pos_ic_rel)

        # Fixed comoving padding (Mpc) applied to the IC-space refinement box to reduce the
        # chance that low-res boundary particles contaminate the analysis region.
        pad_mpc = 2.0
        mins = mins_raw - pad_mpc
        maxs = maxs_raw + pad_mpc
        volume = float(np.prod(maxs - mins))

        # Halo boundary check: reject if there is a massive halo close to a bbox face and that
        # halo is not fully contained within the IC-space bounding box (based on particle IDs).
        #
        # Updated logic (requested): first identify massive haloes that lie near the *edge of the
        # selected kernel sphere* (within +/- L of the kernel radius). Only those “boundary haloes”
        # are then tested for full containment inside the IC-space bounding box.
        bad_halo_count = 0
        failed_halo_centres: list[np.ndarray] = []
        failed_halo_masses: list[float] = []
        halo_ic_plot: np.ndarray | None = None
        if args.fof:
            centres_h, masses_h, radii_h, box_fof = load_fof_catalogue(args.fof)
            masses_h = np.asarray(masses_h, dtype=np.float64)
            radii_h = np.asarray(radii_h, dtype=np.float64)
            keep_h = masses_h >= float(args.halo_mmin)
            centres_h = centres_h[keep_h]
            radii_h = radii_h[keep_h]
            masses_h = masses_h[keep_h]
            L = float(args.halo_boundary)
            if centres_h.size:
                # Identify haloes near the *kernel sphere boundary* at z=0:
                # abs(|x_h - center| - R_kernel) <= L
                rel_h = unwrap_relative(centres_h, center, box_fof)
                dist = np.linalg.norm(rel_h, axis=1)
                on_boundary = np.abs(dist - float(radius)) <= L

                boundary_centres = centres_h[on_boundary]
                boundary_radii = radii_h[on_boundary]
                boundary_masses = masses_h[on_boundary]
                if boundary_centres.shape[0]:
                    print(
                        f"Checking {boundary_centres.shape[0]} haloes near kernel boundary "
                        f"(m>={args.halo_mmin:g}, |r-R|<={L:g} Mpc)..."
                    )

                    ic_offset = infer_contiguous_id_offset(ic_ids)
                    if ic_offset is None:
                        raise RuntimeError(
                            f"{args.ic_snap}: ParticleIDs are not contiguous in-file; cannot efficiently "
                            "check halo containment. Regenerate ICs with contiguous ParticleIDs."
                        )

                    for hcen, hr, hm in zip(boundary_centres, boundary_radii, boundary_masses):
                        ok = halo_fully_selected_in_bbox(
                            ids_sim=ids_sim,
                            pos_sim=pos_sim,
                            box_sim=box_sim,
                            halo_center=hcen,
                            halo_radius=float(hr),
                            ic_pos=ic_pos,
                            ic_id_offset=ic_offset,
                            box_ic=box_ic,
                            zoom_center=center,
                            mins=mins,
                            maxs=maxs,
                        )
                        if not ok:
                            bad_halo_count += 1
                            failed_halo_centres.append(np.asarray(hcen, dtype=np.float64))
                            failed_halo_masses.append(float(hm))
                            if halo_ic_plot is None:
                                halo_ic_plot = halo_ic_positions_for_plot(
                                    ids_sim=ids_sim,
                                    pos_sim=pos_sim,
                                    box_sim=box_sim,
                                    halo_center=hcen,
                                    halo_radius=float(hr),
                                    ic_pos=ic_pos,
                                    ic_id_offset=ic_offset,
                                    box_ic=box_ic,
                                    zoom_center=center,
                                    max_samples=80_000,
                                    rng_seed=grid_idx,
                                )
                            if bad_halo_count <= 3:
                                relc = unwrap_relative(np.asarray(hcen, dtype=np.float64)[None, :], center, box_sim)[0]
                                d = float(np.linalg.norm(relc))
                                print(
                                    f"  Failed halo: M={hm:.3g}, Rfof={float(hr):.3g} Mpc, "
                                    f"|r-center|={d:.3g} Mpc, |(|r|-Rkernel)|={abs(d-float(radius)):.3g} Mpc"
                                )

        if bad_halo_count:
            print(
                f"Rejected center rank={rank} (grid index={grid_idx}) due to {bad_halo_count} massive haloes "
                "near the kernel boundary that are not fully contained within the IC bounding box."
            )
            # Always write reject plots into the per-zoom output directory (unless the user
            # explicitly overrides --plot-out to somewhere else).
            if plot_out:
                reject_png = plot_out.with_name(f"{plot_out.stem}_reject_{grid_idx}.png")
                make_debug_plot(
                    reject_png,
                    center=center,
                    box_sim=box_sim,
                    box_ic=box_ic,
                    mins_raw=mins_raw,
                    maxs_raw=maxs_raw,
                    mins=mins,
                    maxs=maxs,
                    sel_pos=sel_pos,
                    pos_ic_rel=pos_ic_rel,
                    ic_path=args.ic_snap,
                    snap_path=args.snap,
                    fof_path=args.fof,
                    halo_mmin=float(args.halo_mmin),
                    kernel_radius=float(radius),
                    kernel_boundary_L=float(args.halo_boundary),
                    failed_halo_centres=np.array(failed_halo_centres, dtype=np.float64) if failed_halo_centres else None,
                    failed_halo_masses=np.array(failed_halo_masses, dtype=np.float64) if failed_halo_masses else None,
                    halo_ic=halo_ic_plot,
                )
                print(f"Wrote rejected selection plot to {reject_png} (grid index={grid_idx}).")
            continue

        chosen = (
            center,
            val_choice,
            val_delta,
            grid_idx,
            sel_ids,
            sel_pos,
            pos_ic_rel,
            mass_ic_sel,
            mins_raw,
            maxs_raw,
            volume_raw,
            mins,
            maxs,
            volume,
            box_sim,
            box_ic,
            ic_masses,
        )
        break

    if chosen is None:
        raise RuntimeError("Failed to find an acceptable centre after max tries.")

    (
        center,
        val_choice,
        val_delta,
        grid_idx,
        sel_ids,
        sel_pos,
        pos_ic_rel,
        mass_ic_sel,
        mins_raw,
        maxs_raw,
        volume_raw,
        mins,
        maxs,
        volume,
        box_sim,
        box_ic,
        ic_masses,
    ) = chosen
    print(
        f"Selected {sel_ids.size} particles within r={radius} Mpc "
        f"(log10(1+delta)={val_choice:.3g}, delta={val_delta:.3g}, grid index={grid_idx})."
    )
    total_mass = float(mass_ic_sel.sum())
    mean_density_raw = total_mass / volume_raw
    mean_density = total_mass / volume

    # MUSIC2 refinement levels define an effective full-box resolution of 2^level.
    # The high-res particle mass corresponds to that full-box spacing, even though
    # high-res particles are only realized within the refined region.
    #
    # Use the parent IC particle mass (assumed uniform) and scale by the refinement factor.
    m_parent = float(np.median(ic_masses))  # (10^10 Msun/h)
    refine_levels = base_levelmax - parent_levelmin
    if refine_levels < 0:
        raise ValueError("levelmax cannot be smaller than parent levelmin")
    m_high = m_parent / (8.0**refine_levels)  # each +1 level => /8 in mass
    n_high_for_mass = total_mass / m_high
    # Compute mass actually contained in the IC-space bounding box (can differ from cosmic mean).
    # This better reflects how many high-res particles would be needed if the entire box region
    # is refined at levelmax.
    _ids_box_sample, _pos_box_sample, _boxsize_ic, n_box_particles, mass_box = sample_box_particles(
        ic_path=args.ic_snap,
        center=center,
        mins=mins,
        maxs=maxs,
        max_samples=1,  # minimal sampling; this call is primarily for mass/count
        seed=0,
    )
    n_high_in_box = mass_box / m_high

    print("\nIC-space bounding box (relative to kernel center, Mpc):")
    print(f"  mins (raw): {mins_raw}")
    print(f"  maxs (raw): {maxs_raw}")
    print(f"  extents (raw): {maxs_raw - mins_raw}")
    print(f"  volume (raw): {volume_raw:.6e} Mpc^3")
    print(f"  padding: {pad_mpc:.3g} Mpc")
    print(f"  mins (padded): {mins}")
    print(f"  maxs (padded): {maxs}")
    print(f"  extents (padded): {maxs - mins}")
    print(f"  volume (padded): {volume:.6e} Mpc^3")
    print("\nMass summary (from IC particle masses):")
    print(f"  Total mass in selection: {total_mass:.6e} (10^10 Msun)")
    print(f"  Mean density in raw box: {mean_density_raw:.6e} (10^10 Msun) / Mpc^3")
    print(f"  Mean density in padded:  {mean_density:.6e} (10^10 Msun) / Mpc^3")
    print("\nZoom mass / resolution estimate:")
    print(f"  Parent levelmin: {parent_levelmin} (m_parent={m_parent:.6e} in 10^10 Msun)")
    print(f"  Zoom levelmax:   {base_levelmax} (Δlevels={refine_levels}, m_high={m_high:.6e} in 10^10 Msun)")
    print(f"  High-res particles to represent selected mass: ~{n_high_for_mass:,.0f} ({n_high_for_mass**(1/3):.1f}^3)")
    print(f"  High-res particles in bounding-box volume (if fully filled): ~{n_high_in_box:,.0f} ({n_high_in_box**(1/3):.1f}^3)")
    print(f"  Bounding-box particle count (parent IC): {n_box_particles:,}")
    print(f"  Bounding-box mass (parent IC): {mass_box:.6e} (10^10 Msun)")
    abs_min = (center + mins) % parent_box
    abs_max = (center + maxs) % parent_box
    extent_abs = (abs_max - abs_min) % parent_box
    center_abs = (abs_min + 0.5 * extent_abs) % parent_box

    write_music_zoom_config(
        template_path=args.template,
        out_path=out_config,
        parent_cfg=parent_cfg,
        center_abs=center_abs,
        extent_abs=extent_abs,
        box_size=parent_box,
        base_levelmin=base_levelmin,
        base_levelmax=base_levelmax,
        out_ics=out_ics,
    )
    print(f"\nWrote MUSIC2 zoom config to {out_config}")

    # Copy/write SWIFT zoom YAML into the run directory, updating cosmology and paths.
    swift_out = run_dir / "swift_zoom_params.yaml"
    write_swift_zoom_yaml(
        template=args.swift_template,
        out_path=swift_out,
        run_dir=run_dir,
        ic_path=out_ics,
        omega_m=omega_m,
        omega_b=omega_b,
        h=h,
        boxlength_mpc_h=parent_box,
        levelmax=base_levelmax,
    )
    print(f"Wrote SWIFT zoom params to {swift_out}")

    # Copy the SWIFT zoom submission script into the run directory for convenience.
    submit_tpl = ROOT / "tessera" / "submit_swift_zoom.slurm"
    if submit_tpl.exists():
        shutil.copy2(submit_tpl, run_dir / submit_tpl.name)
        print(f"Copied SWIFT submit script to {run_dir / submit_tpl.name}")

    # Copy MUSIC2 helper script for convenience.
    music2_sh = ROOT / "tessera" / "generate_music2_ics.sh"
    if music2_sh.exists():
        shutil.copy2(music2_sh, run_dir / music2_sh.name)
        print(f"Copied MUSIC2 helper script to {run_dir / music2_sh.name}")

    # Copy SWIFT output list to the run directory if present (template refers to output_times.txt).
    out_times = ROOT / "tessera" / "output_times.txt"
    if out_times.exists():
        shutil.copy2(out_times, run_dir / out_times.name)
        print(f"Copied output times to {run_dir / out_times.name}")

    if plot_out:
        make_debug_plot(
            plot_out,
            center=center,
            box_sim=box_sim,
            box_ic=box_ic,
            mins_raw=mins_raw,
            maxs_raw=maxs_raw,
            mins=mins,
            maxs=maxs,
            sel_pos=sel_pos,
            pos_ic_rel=pos_ic_rel,
            ic_path=args.ic_snap,
            snap_path=args.snap,
            fof_path=args.fof,
            halo_mmin=float(args.halo_mmin),
            kernel_radius=float(radius),
            kernel_boundary_L=float(args.halo_boundary),
            halo_ic=None,
        )
        print(f"Wrote selection plot to {plot_out}")
if __name__ == "__main__":
    main()
