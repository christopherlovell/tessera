#!/usr/bin/env python
"""
Generate a MUSIC2 zoom configuration for the Tessera parent box from a selection sphere.

Inputs:
  - A gridder output to choose a center based on a kernel overdensity (kernel mode), or a SWIFT FOF catalogue (halo mode).
  - A z=0 snapshot with ParticleIDs to select particles within the chosen sphere.
  - The parent IC file (ParticleIDs + Masses) to trace those particles back to ICs.
"""

import argparse
import configparser
import os
import sys
from pathlib import Path
import shutil
import atexit

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

def _resolve_in_run_dir(p: Path | None, run_dir: Path) -> Path | None:
    if p is None:
        return None
    if p.is_absolute():
        return p
    run_dir = run_dir.resolve()
    candidate = (run_dir / p).resolve()
    try:
        candidate.relative_to(run_dir)
    except ValueError:
        # Prevent `../` from escaping the run directory.
        candidate = run_dir / p.name
    return candidate


DEFAULT_PARENT_CONFIG = ROOT / "configs" / "music2_parent_tessera.conf"
DEFAULT_SUBMIT_SCRIPT_COSMA7 = ROOT / "scripts" / "submit_zoom_cosma7.slurm"
DEFAULT_SUBMIT_SCRIPT_COSMA8 = ROOT / "scripts" / "submit_zoom_cosma8.slurm"
DEFAULT_MUSIC2_HELPER = ROOT / "scripts" / "generate_music2_ics.sh"
DEFAULT_OUTPUT_TIMES = ROOT / "configs" / "output_times.txt"
DEFAULT_ZOOM_BASE = Path("/snap8/scratch/dp004/dc-love2/tessera/")

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

from utilities import (
    bounding_box,
    choose_center,
    kernel_radius_from_gridder,
    load_gridder,
    load_snap,
    match_into_ic,
    select_sphere,
    unwrap_relative,
)

from methods import (
    _Tee,
    _parse_float,
    _parse_int,
    iter_existing_zoom_selections,
    periodic_distance,
    selection_center_and_radius,
    infer_contiguous_id_offset,
    halo_ic_positions_for_plot,
    load_fof_catalogue,
    make_debug_plot,
    next_zoom_index,
    sample_box_particles,
    utc_now_iso,
    write_music_zoom_config,
    write_swift_zoom_yaml,
    write_zoom_selection_metadata,
)


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


def main():
    ap = argparse.ArgumentParser(description="Generate MUSIC2 zoom config for Tessera.")
    ap.add_argument(
        "--select-mode",
        choices=("kernel", "halo"),
        default="kernel",
        help="Selection mode: overdensity kernel centre (kernel) or random FOF halo (halo).",
    )
    ap.add_argument("--label", type=str, help="Optional label stored in zoom_selection.json (useful for suite runs).")
    ap.add_argument("--grid", type=Path, help="Gridder output HDF5 (positions + overdensities). Required for kernel mode.")
    ap.add_argument("--snap", type=Path, required=True, help="z=0 snapshot with ParticleIDs (pmwd or SWIFT).")
    ap.add_argument("--ic-snap", type=Path, required=True, help="Parent IC snapshot with ParticleIDs and Masses.")
    ap.add_argument(
        "--machine",
        choices=("cosma7", "cosma8"),
        default=os.environ.get("MACHINE", "cosma7"),
        help="Select machine for copying the appropriate submit script into the run directory.",
    )
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
        "--selection-radius",
        type=float,
        help="Selection sphere radius (Mpc). Required for halo mode; ignored for kernel mode unless --kernel-radius is used.",
    )

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
        default=DEFAULT_PARENT_CONFIG,
        help="Tessera parent MUSIC2 config.",
    )
    ap.add_argument(
        "--template",
        type=Path,
        default=ROOT / "configs" / "music2_zoom.conf",
        help="MUSIC2 zoom template.",
    )
    ap.add_argument(
        "--out-config",
        type=Path,
        help=(
            "Output MUSIC2 zoom config path. If relative, it is interpreted inside the run directory. "
            "Default: <out-base>/<idx>/music2_zoom.conf."
        ),
    )
    ap.add_argument(
        "--levelmin",
        type=int,
        help="Zoom levelmin (default: setup.levelmin from --template; fallback: parent levelmin).",
    )
    ap.add_argument(
        "--levelmax",
        type=int,
        help="Zoom levelmax (finest resolution; default: setup.levelmax from --template).",
    )
    ap.add_argument(
        "--out-ics",
        type=Path,
        help=(
            "Output ICs path in MUSIC2 config. If relative, it is interpreted inside the run directory. "
            "Default: <out-base>/<idx>/zoom_ICS_<idx>.hdf5."
        ),
    )
    ap.add_argument(
        "--plot-out",
        type=Path,
        help=(
            "PNG plot path. If relative, it is interpreted inside the run directory. "
            "Default: <out-base>/<idx>/zoom_ICs_<idx>.png."
        ),
    )
    ap.add_argument(
        "--fof",
        type=Path,
        help="SWIFT-style FOF catalogue (required for halo mode; optional for kernel mode plotting/checks).",
    )
    ap.add_argument("--halo-mmin", type=float, default=1e4, help="Minimum halo mass (1e10 Msun).")
    ap.add_argument("--halo-mmax", type=float, help="Maximum halo mass (1e10 Msun).")
    ap.add_argument("--halo-seed", type=int, default=0, help="RNG seed for random halo choice in halo mode.")
    ap.add_argument(
        "--halo-rank",
        type=int,
        help="Select the Nth most massive halo within the mass bounds (0=most massive). If set, overrides --halo-seed.",
    )
    ap.add_argument(
        "--halo-boundary",
        type=float,
        default=2.5,
        help="Boundary buffer L (Mpc) for optional 'boundary halo' rejection check (kernel mode by default).",
    )
    ap.add_argument(
        "--check-boundary-haloes",
        action="store_true",
        help="Also run the boundary-halo rejection check in halo mode.",
    )
    ap.add_argument(
        "--target-nhigh",
        type=float,
        help="Target estimated high-res particle count in the padded IC-space bbox (geometric estimate).",
    )
    ap.add_argument("--target-nhigh-rtol", type=float, default=0.10, help="Relative tolerance for --target-nhigh.")
    ap.add_argument("--pad-mpc", type=float, default=2.0, help="IC-space bbox padding (Mpc) when not tuning.")
    ap.add_argument("--pad-min-mpc", type=float, default=0.0, help="Minimum padding (Mpc) during tuning.")
    ap.add_argument("--pad-max-mpc", type=float, default=20.0, help="Maximum padding (Mpc) during tuning.")
    ap.add_argument("--pad-tune-iters", type=int, default=30, help="Max iterations for pad tuning (binary search).")
    ap.add_argument("--allow-overlap", action="store_true", help="Allow overlap with previously created selection spheres.")
    ap.add_argument("--overlap-buffer-mpc", type=float, default=0.0, help="Extra buffer added when rejecting overlaps (Mpc).")
    ap.add_argument(
        "--swift-template",
        type=Path,
        default=ROOT / "configs" / "swift_zoom_params.yaml",
        help="SWIFT zoom YAML template to copy into the zoom directory.",
    )
    ap.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Selection metadata JSON filename (basename only; default: zoom_selection.json).",
    )
    args = ap.parse_args()

    def _resolve_existing_path(p: Path | None) -> Path | None:
        if p is None:
            return None
        p = Path(p)
        if p.is_absolute():
            return p
        if p.exists():
            return p
        alt = (ROOT / p)
        if alt.exists():
            return alt
        return p

    # Allow running from any working directory while passing repo-relative paths like `configs/...`.
    args.parent_config = _resolve_existing_path(args.parent_config)
    args.template = _resolve_existing_path(args.template)
    args.swift_template = _resolve_existing_path(args.swift_template)
    args.grid = _resolve_existing_path(args.grid)
    args.fof = _resolve_existing_path(args.fof)
    args.snap = _resolve_existing_path(args.snap)
    args.ic_snap = _resolve_existing_path(args.ic_snap)

    zoom_index = int(args.index) if args.index is not None else next_zoom_index(args.out_base)
    run_dir = (args.out_base / f"{zoom_index:04d}")
    preexisting = run_dir.exists() and any(run_dir.iterdir())
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()
    log_path = run_dir / f"generate_zoom_ics_{zoom_index:04d}.txt"
    log_fh = log_path.open("w")
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, log_fh)
    # print(f"Logging to {log_path}")
    if preexisting:
        print(f"WARNING: {run_dir} already existed and was non-empty; outputs may be overwritten.")
    atexit.register(lambda: setattr(sys, "stdout", old_stdout))
    atexit.register(log_fh.close)

    out_config = _resolve_in_run_dir(args.out_config, run_dir) or (run_dir / "music2_zoom.conf")
    plot_out = _resolve_in_run_dir(args.plot_out, run_dir) or (run_dir / f"zoom_ICs_{zoom_index:04d}.png")

    # Always place the MUSIC2 IC output inside the run directory (only allow overriding the basename).
    out_ics_name = args.out_ics.name if args.out_ics is not None else f"zoom_ICS_{zoom_index:04d}.hdf5"
    out_ics = run_dir / out_ics_name
    metadata_name = args.metadata.name if args.metadata is not None else "zoom_selection.json"
    metadata_path = run_dir / metadata_name

    if args.parent_config is None:
        raise ValueError("--parent-config cannot be None.")
    if not args.parent_config.exists():
        raise FileNotFoundError(f"Parent MUSIC2 config not found: {args.parent_config} (pass --parent-config).")
    if args.template is None or not args.template.exists():
        raise FileNotFoundError(
            "Could not find MUSIC2 zoom template config; pass --template. "
            f"Tried: {args.template!s}"
        )

    parent_cfg = configparser.ConfigParser()
    parent_cfg.optionxform = str
    with args.parent_config.open() as fh:
        parent_cfg.read_file(fh)

    template_cfg = configparser.ConfigParser()
    template_cfg.optionxform = str
    with args.template.open() as fh:
        template_cfg.read_file(fh)

    parent_levelmin = _parse_int(parent_cfg["setup"]["levelmin"])
    parent_box = _parse_float(parent_cfg["setup"]["boxlength"])
    omega_m = _parse_float(parent_cfg["cosmology"]["Omega_m"])
    omega_b = _parse_float(parent_cfg["cosmology"].get("Omega_b", "0.0"))
    h = _parse_float(parent_cfg["cosmology"]["H0"]) / 100.0

    template_levelmin = None
    template_levelmax = None
    if "setup" in template_cfg:
        if "levelmin" in template_cfg["setup"]:
            template_levelmin = _parse_int(template_cfg["setup"]["levelmin"])
        if "levelmax" in template_cfg["setup"]:
            template_levelmax = _parse_int(template_cfg["setup"]["levelmax"])

    if args.levelmin is not None:
        base_levelmin = int(args.levelmin)
    elif template_levelmin is not None:
        base_levelmin = int(template_levelmin)
    else:
        base_levelmin = int(parent_levelmin)
        print(f"WARNING: {args.template} missing [setup]/levelmin; defaulting to parent levelmin={parent_levelmin}.")

    if args.levelmax is not None:
        base_levelmax = int(args.levelmax)
    elif template_levelmax is not None:
        base_levelmax = int(template_levelmax)
    else:
        raise ValueError(f"{args.template} missing [setup]/levelmax and --levelmax was not provided.")

    if base_levelmax < base_levelmin:
        raise ValueError("levelmax cannot be smaller than levelmin")

    if args.select_mode == "kernel":
        if args.grid is None:
            raise ValueError("--grid is required for --select-mode kernel.")
    elif args.select_mode == "halo":
        if args.fof is None:
            raise ValueError("--fof is required for --select-mode halo.")
        if args.selection_radius is None:
            raise ValueError("--selection-radius is required for --select-mode halo.")
    else:
        raise RuntimeError(f"Unexpected --select-mode={args.select_mode!r}")

    if args.target_nhigh is not None and float(args.target_nhigh) <= 0:
        raise ValueError("--target-nhigh must be > 0.")
    if float(args.target_nhigh_rtol) < 0:
        raise ValueError("--target-nhigh-rtol must be >= 0.")
    if float(args.pad_min_mpc) < 0 or float(args.pad_max_mpc) < 0:
        raise ValueError("--pad-min-mpc/--pad-max-mpc must be >= 0.")
    if int(args.pad_tune_iters) < 1:
        raise ValueError("--pad-tune-iters must be >= 1.")
    if float(args.overlap_buffer_mpc) < 0:
        raise ValueError("--overlap-buffer-mpc must be >= 0.")
    if args.halo_rank is not None and int(args.halo_rank) < 0:
        raise ValueError("--halo-rank must be >= 0.")

    ids_sim, pos_sim, _, box_sim = load_snap(args.snap, include_masses=False)
    ic_ids, ic_pos, ic_masses, box_ic = load_snap(args.ic_snap, include_masses=True)

    fof_centres = None
    fof_masses = None
    fof_radii = None
    box_fof = None
    if args.fof is not None:
        fof_centres, fof_masses, fof_radii, box_fof = load_fof_catalogue(args.fof)
        if box_fof is not None and abs(float(box_fof) - float(box_sim)) > 1e-5:
            print(f"WARNING: FOF BoxSize={float(box_fof):g} differs from snapshot BoxSize={float(box_sim):g} (units mismatch?).")

    existing = []
    if not args.allow_overlap:
        for meta in iter_existing_zoom_selections(args.out_base):
            run_dir_prev = meta.get("_run_dir", None)
            if isinstance(run_dir_prev, str) and Path(run_dir_prev).resolve() == run_dir:
                continue
            cr = selection_center_and_radius(meta)
            if cr is None:
                continue
            cprev, rprev = cr
            existing.append((cprev, float(rprev), run_dir_prev))

    selection_radius = None
    overd = None
    overd_for_choice = None
    target_for_choice = None
    candidate_halo_indices = None
    chosen_halo_index = None
    chosen_halo_rank = None
    chosen_halo_mass = None

    if args.select_mode == "kernel":
        pos_grid, overd = load_gridder(args.grid)
        selection_radius = kernel_radius_from_gridder(args.grid, args.kernel_index, args.kernel_radius)

        overd_for_choice = {}
        for k, d in overd.items():
            if np.any(d <= -1):
                bad = int(np.sum(d <= -1))
                raise RuntimeError(
                    f"{args.grid}: kernel {k} has {bad} grid points with overdensity <= -1; cannot use log10(1+delta)."
                )
            overd_for_choice[k] = np.log10(1.0 + d)
        target_for_choice = None if args.target_logdelta is None else float(args.target_logdelta)
    else:
        selection_radius = float(args.selection_radius)
        masses = np.asarray(fof_masses, dtype=np.float64)
        mmin = float(args.halo_mmin)
        mmax = float(args.halo_mmax) if args.halo_mmax is not None else float("inf")
        eligible = np.where((masses >= mmin) & (masses <= mmax))[0]
        if eligible.size == 0:
            raise RuntimeError(f"No haloes found in {args.fof} with {mmin:g} <= M <= {mmax:g} (1e10 Msun).")
        order = eligible[np.argsort(masses[eligible])[::-1]]  # most massive first
        if args.halo_rank is not None:
            if int(args.halo_rank) >= int(order.size):
                raise RuntimeError(
                    f"--halo-rank {int(args.halo_rank)} out of range for {order.size} eligible haloes "
                    f"({mmin:g} <= M <= {mmax:g} in 1e10 Msun)."
                )
            candidate_halo_indices = order
        else:
            rng = np.random.default_rng(int(args.halo_seed))
            candidate_halo_indices = rng.permutation(eligible)

    def overlaps_previous(center: np.ndarray, rad: float) -> tuple[bool, str | None]:
        if not existing:
            return False, None
        for cprev, rprev, run_dir_prev in existing:
            d = periodic_distance(np.asarray(center, dtype=np.float64), cprev, float(box_sim))
            if d < float(rad) + float(rprev) + float(args.overlap_buffer_mpc):
                return True, (None if run_dir_prev is None else str(run_dir_prev))
        return False, None

    def nhigh_estimate_for_bbox(mins_raw: np.ndarray, maxs_raw: np.ndarray, pad: float) -> float:
        ext = (np.asarray(maxs_raw, dtype=np.float64) - np.asarray(mins_raw, dtype=np.float64)) + (2.0 * float(pad))
        volume = float(np.prod(ext))
        spacing_mpc = (float(parent_box) / (2 ** int(base_levelmax))) / float(h)
        return volume / (spacing_mpc ** 3)

    def tune_pad(mins_raw: np.ndarray, maxs_raw: np.ndarray) -> float:
        if args.target_nhigh is None:
            return float(args.pad_mpc)
        target = float(args.target_nhigh)
        lo = float(args.pad_min_mpc)
        hi = float(args.pad_max_mpc)
        if lo < 0:
            lo = 0.0
        if hi < lo:
            hi = lo
        n_lo = nhigh_estimate_for_bbox(mins_raw, maxs_raw, lo)
        n_hi = nhigh_estimate_for_bbox(mins_raw, maxs_raw, hi)
        if target <= n_lo:
            return lo
        if target >= n_hi:
            return hi
        for _ in range(int(args.pad_tune_iters)):
            mid = 0.5 * (lo + hi)
            n_mid = nhigh_estimate_for_bbox(mins_raw, maxs_raw, mid)
            if n_mid < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def pad_hits_target(mins_raw: np.ndarray, maxs_raw: np.ndarray, pad: float) -> bool:
        if args.target_nhigh is None:
            return True
        target = float(args.target_nhigh)
        est = nhigh_estimate_for_bbox(mins_raw, maxs_raw, pad)
        rtol = float(args.target_nhigh_rtol)
        return abs(est - target) <= rtol * target

    # Iterate candidate centres until overlap / target-nhigh / optional boundary-halo checks pass.
    chosen = None
    for attempt in range(args.max_tries):
        if args.select_mode == "kernel":
            rank = args.target_rank + attempt
            center, val_choice, grid_idx = choose_center(
                pos_grid, overd_for_choice, args.kernel_index, target_for_choice, rank=rank
            )
            val_delta = float(overd[args.kernel_index][grid_idx])
        else:
            start = int(args.halo_rank) if args.halo_rank is not None else 0
            idx_in_list = start + attempt
            if idx_in_list >= int(candidate_halo_indices.size):
                break
            chosen_halo_rank = idx_in_list if args.halo_rank is not None else None
            chosen_halo_index = int(candidate_halo_indices[idx_in_list])
            center = np.asarray(fof_centres[chosen_halo_index], dtype=np.float64)
            chosen_halo_mass = float(fof_masses[chosen_halo_index])
            val_choice = None
            val_delta = None
            grid_idx = None
            rank = None

        ov, ov_dir = overlaps_previous(center, float(selection_radius))
        if ov:
            print(
                f"Rejected candidate center due to overlap with prior selection sphere (dir={ov_dir}). "
                f"R={float(selection_radius):g} Mpc."
            )
            continue

        sel_ids, sel_pos = select_sphere(ids_sim, pos_sim, center, float(selection_radius), box_sim)
        if sel_ids.size == 0:
            continue

        pos_ic_rel, mass_ic_sel = match_into_ic(sel_ids, center, ic_ids, ic_pos, ic_masses, box_ic)
        mins_raw, maxs_raw, volume_raw = bounding_box(pos_ic_rel)

        # Fixed comoving padding (Mpc) applied to the IC-space refinement box to reduce the
        # chance that low-res boundary particles contaminate the analysis region.
        pad_mpc = tune_pad(mins_raw, maxs_raw)
        mins = mins_raw - pad_mpc
        maxs = maxs_raw + pad_mpc
        volume = float(np.prod(maxs - mins))

        if not pad_hits_target(mins_raw, maxs_raw, pad_mpc):
            if args.target_nhigh is not None:
                est = nhigh_estimate_for_bbox(mins_raw, maxs_raw, pad_mpc)
                print(
                    f"Rejected candidate due to target-nhigh miss: est={est:,.3g}, target={float(args.target_nhigh):,.3g}, "
                    f"rtol={float(args.target_nhigh_rtol):g}."
                )
                continue

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
        do_boundary_check = (
            (args.select_mode == "kernel") or (args.select_mode == "halo" and bool(args.check_boundary_haloes))
        )
        if do_boundary_check and args.fof and float(args.halo_boundary) > 0 and fof_centres is not None:
            masses_h = np.asarray(fof_masses, dtype=np.float64)
            radii_h = np.asarray(fof_radii, dtype=np.float64)
            keep_h = masses_h >= float(args.halo_mmin)
            centres_h = np.asarray(fof_centres, dtype=np.float64)[keep_h]
            radii_h = radii_h[keep_h]
            masses_h = masses_h[keep_h]
            L = float(args.halo_boundary)
            if centres_h.size:
                rel_h = unwrap_relative(centres_h, center, float(box_fof))
                dist = np.linalg.norm(rel_h, axis=1)
                on_boundary = np.abs(dist - float(selection_radius)) <= L

                boundary_centres = centres_h[on_boundary]
                boundary_radii = radii_h[on_boundary]
                boundary_masses = masses_h[on_boundary]
                if boundary_centres.shape[0]:
                    print(
                        f"Checking {boundary_centres.shape[0]} haloes near selection boundary "
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
                                    rng_seed=(0 if grid_idx is None else int(grid_idx)),
                                )

        if bad_halo_count:
            print(
                f"Rejected center due to {bad_halo_count} massive haloes "
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
                    kernel_radius=float(selection_radius),
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
            rank,
            attempt,
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
            float(pad_mpc),
        )
        break

    if chosen is None:
        raise RuntimeError("Failed to find an acceptable centre after max tries.")

    (
        center,
        val_choice,
        val_delta,
        grid_idx,
        chosen_rank,
        chosen_attempt,
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
        pad_mpc,
    ) = chosen
    print(
        f"Selected {sel_ids.size} particles within r={float(selection_radius)} Mpc "
        + (
            f"(log10(1+delta)={val_choice:.3g}, delta={val_delta:.3g}, grid index={grid_idx})."
            if args.select_mode == "kernel"
            else (
                f"(halo rank={chosen_halo_rank}, index={chosen_halo_index}, M={chosen_halo_mass:.3g} in 1e10 Msun)."
                if chosen_halo_rank is not None
                else f"(halo index={chosen_halo_index}, M={chosen_halo_mass:.3g} in 1e10 Msun)."
            )
        )
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

    print("\nIC-space bounding box (relative to selection center, Mpc):")
    print(f"  mins (raw): {mins_raw}")
    print(f"  maxs (raw): {maxs_raw}")
    print(f"  extents (raw): {maxs_raw - mins_raw}")
    print(f"  volume (raw): {volume_raw:.6e} Mpc^3")
    print(f"  padding: {pad_mpc:.3g} Mpc")
    print(f"  mins (padded): {mins}")
    print(f"  maxs (padded): {maxs}")
    print(f"  extents (padded): {maxs - mins}")
    print(f"  volume (padded): {volume:.6e} Mpc^3")
    # print("\nMass summary (from IC particle masses):")
    # print(f"  Total mass in selection: {total_mass:.6e} (10^10 Msun)")
    # print(f"  Mean density in raw box: {mean_density_raw:.6e} (10^10 Msun) / Mpc^3")
    # print(f"  Mean density in padded:  {mean_density:.6e} (10^10 Msun) / Mpc^3")
    print("\nZoom mass / resolution estimate:")
    print(f"  Parent levelmin: {parent_levelmin} (m_parent={m_parent:.6e} in 10^10 Msun)")
    print(f"  Zoom levelmax:   {base_levelmax} (Δlevels={refine_levels}, m_high={m_high:.6e} in 10^10 Msun)")
    print(f"  High-res particles to represent selected mass: ~{n_high_for_mass:,.0f} ({n_high_for_mass**(1/3):.1f}^3)")
    print(f"  High-res particles in bounding-box volume (if fully filled): ~{n_high_in_box:,.0f} ({n_high_in_box**(1/3):.1f}^3)")
    print(f"  Bounding-box particle count (parent IC): {n_box_particles:,}")
    # print(f"  Bounding-box mass (parent IC): {mass_box:.6e} (10^10 Msun)")
    spacing_mpc = (float(parent_box) / (2 ** int(base_levelmax))) / float(h)
    n_high_geom_raw = float(volume_raw) / (spacing_mpc ** 3)
    n_high_geom_padded = float(volume) / (spacing_mpc ** 3)
    print("\nGeometric high-res particle estimate:")
    print(f"  spacing (Mpc): {spacing_mpc:.6e}")
    print(f"  N_high (raw bbox):    ~{n_high_geom_raw:,.0f} ({n_high_geom_raw**(1/3):.1f}^3)")
    print(f"  N_high (padded bbox): ~{n_high_geom_padded:,.0f} ({n_high_geom_padded**(1/3):.1f}^3)")
    if args.target_nhigh is not None:
        print(f"  target_nhigh: {float(args.target_nhigh):,.3g} (rtol={float(args.target_nhigh_rtol):g})")
    # Coordinates coming from SWIFT-format HDF5 snapshots are in Mpc (Header.BoxSize units),
    # while MUSIC2 config `boxlength` is in Mpc/h. Use the parent IC Header.BoxSize for
    # periodic wrapping and compute dimensionless fractions accordingly in `write_music_zoom_config`.
    abs_min = (center + mins) % box_ic
    abs_max = (center + maxs) % box_ic
    extent_abs = (abs_max - abs_min) % box_ic
    center_abs = (abs_min + 0.5 * extent_abs) % box_ic

    selection = {
        "mode": str(args.select_mode),
        "selection_radius_mpc": float(selection_radius),
        "center_mpc": np.asarray(center, dtype=np.float64),
    }
    if args.select_mode == "kernel":
        selection.update(
            {
                "kernel_index": int(args.kernel_index),
                "kernel_radius_mpc": float(selection_radius),
                "kernel_radius_override_mpc": (None if args.kernel_radius is None else float(args.kernel_radius)),
                "target_logdelta": (None if args.target_logdelta is None else float(args.target_logdelta)),
                "target_rank": int(args.target_rank),
                "chosen_rank": int(chosen_rank),
                "chosen_attempt": int(chosen_attempt),
                "grid_index": int(grid_idx),
                "log10_1p_delta": float(val_choice),
                "delta": float(val_delta),
            }
        )
    else:
        selection.update(
            {
                "halo_index": (None if chosen_halo_index is None else int(chosen_halo_index)),
                "halo_mass_1e10msun": (None if chosen_halo_mass is None else float(chosen_halo_mass)),
                "halo_seed": int(args.halo_seed),
                "halo_rank": (None if args.halo_rank is None else int(args.halo_rank)),
                "halo_rank_chosen": (None if chosen_halo_rank is None else int(chosen_halo_rank)),
                "halo_mmin_1e10msun": float(args.halo_mmin),
                "halo_mmax_1e10msun": (None if args.halo_mmax is None else float(args.halo_mmax)),
                "chosen_attempt": int(chosen_attempt),
            }
        )
    bbox = {
        "pad_mpc": float(pad_mpc),
        "mins_raw_mpc": np.asarray(mins_raw, dtype=np.float64),
        "maxs_raw_mpc": np.asarray(maxs_raw, dtype=np.float64),
        "mins_padded_mpc": np.asarray(mins, dtype=np.float64),
        "maxs_padded_mpc": np.asarray(maxs, dtype=np.float64),
        "volume_raw_mpc3": float(volume_raw),
        "volume_padded_mpc3": float(volume),
        "center_abs_mpc": np.asarray(center_abs, dtype=np.float64),
        "extent_abs_mpc": np.asarray(extent_abs, dtype=np.float64),
        "box_ic_mpc": float(box_ic),
        "box_sim_mpc": float(box_sim),
    }
    levels = {
        "parent_levelmin": int(parent_levelmin),
        "zoom_levelmin": int(base_levelmin),
        "zoom_levelmax": int(base_levelmax),
    }
    cosmology = {
        "Omega_m": float(omega_m),
        "Omega_b": float(omega_b),
        "h": float(h),
        "boxlength_mpc_h": float(parent_box),
    }
    mass_summary = {
        "total_mass_selection_1e10msun": float(total_mass),
        "mean_density_raw_1e10msun_mpc3": float(mean_density_raw),
        "mean_density_padded_1e10msun_mpc3": float(mean_density),
        "parent_particle_mass_1e10msun_h": float(m_parent),
        "highres_particle_mass_1e10msun_h": float(m_high),
        "n_high_for_selection_mass": float(n_high_for_mass),
        "bbox_parent_particle_count": int(n_box_particles),
        "bbox_parent_mass_1e10msun_h": float(mass_box),
        "n_high_in_bbox_if_filled": float(n_high_in_box),
        "n_high_geom_raw_bbox": float(n_high_geom_raw),
        "n_high_geom_padded_bbox": float(n_high_geom_padded),
    }
    payload = {
        "schema": "tessera.zoom_selection.v2",
        "created_utc": utc_now_iso(),
        "label": (None if args.label is None else str(args.label)),
        "zoom_index": int(zoom_index),
        "run_dir": run_dir,
        "inputs": {
            "gridder": args.grid,
            "snap_z0": args.snap,
            "ic_snap": args.ic_snap,
            "fof": args.fof,
            "parent_config": args.parent_config,
            "music_template": args.template,
            "swift_template": args.swift_template,
        },
        "outputs": {
            "metadata": metadata_path,
            "log": log_path,
            "music2_zoom_conf": out_config,
            "zoom_ics_hdf5": out_ics,
            "plot_png": plot_out,
            "swift_params_yaml": (run_dir / "swift_zoom_params.yaml"),
        },
        "selection": selection,
        "bbox": bbox,
        "levels": levels,
        "cosmology": cosmology,
        "mass_summary": mass_summary,
        "counts": {
            "n_selected_particles": int(sel_ids.size),
        },
        "command": {
            "argv": list(sys.argv),
            "cwd": str(Path.cwd()),
        },
    }
    write_zoom_selection_metadata(metadata_path, payload)
    # print(f"Wrote selection metadata to {metadata_path}")

    write_music_zoom_config(
        template_path=args.template,
        out_path=out_config,
        parent_cfg=parent_cfg,
        center_abs=center_abs,
        extent_abs=extent_abs,
        boxlength_mpc_h=parent_box,
        coord_box_size=box_ic,
        base_levelmin=base_levelmin,
        base_levelmax=base_levelmax,
        out_ics=out_ics,
    )
    # print(f"\nWrote MUSIC2 zoom config to {out_config}")

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
    # print(f"Wrote SWIFT zoom params to {swift_out}")

    # Copy the SWIFT zoom submission script into the run directory for convenience.
    submit_script = DEFAULT_SUBMIT_SCRIPT_COSMA7 if args.machine == "cosma7" else DEFAULT_SUBMIT_SCRIPT_COSMA8
    if not submit_script.exists():
        print(f"WARNING: submit script not found at {submit_script} for machine={args.machine}; not copying to run directory.")
    else:
        dest = run_dir / "submit_zoom.slurm"
        shutil.copy2(submit_script, dest)
        # print(f"Copied submit script to {dest} (source: {submit_script})")

    # Copy MUSIC2 helper script for convenience.
    if not DEFAULT_MUSIC2_HELPER.exists():
        print(f"WARNING: MUSIC2 helper script not found at {DEFAULT_MUSIC2_HELPER}; not copying to run directory.")
    else:
        shutil.copy2(DEFAULT_MUSIC2_HELPER, run_dir / DEFAULT_MUSIC2_HELPER.name)
        # print(f"Copied MUSIC2 helper script to {run_dir / DEFAULT_MUSIC2_HELPER.name}")

    # Copy SWIFT output list to the run directory if present (template refers to output_times.txt).
    if not DEFAULT_OUTPUT_TIMES.exists():
        print(f"WARNING: output times not found at {DEFAULT_OUTPUT_TIMES}; SWIFT may fail if template uses output_list.")
    else:
        shutil.copy2(DEFAULT_OUTPUT_TIMES, run_dir / DEFAULT_OUTPUT_TIMES.name)
        # print(f"Copied output times to {run_dir / DEFAULT_OUTPUT_TIMES.name}")

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
            kernel_radius=float(selection_radius),
            kernel_boundary_L=float(args.halo_boundary),
            halo_ic=None,
        )
        # print(f"Wrote selection plot to {plot_out}")
if __name__ == "__main__":
    main()
