#!/usr/bin/env python
"""
Shared helper methods for Tessera scripts.

This file collects reusable utilities extracted from `generate_zoom_ics.py` to keep
the main script focused on orchestration/CLI logic.
"""

import configparser
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import yaml


def unwrap_relative(pos, center, box):
    """Center positions on center with periodic wrap to [-box/2, box/2]."""
    delta = pos - center
    delta = (delta + box / 2) % box - box / 2
    return delta

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _parse_int(val: str) -> int:
    return int(str(val).split()[0])


def _parse_float(val: str) -> float:
    return float(str(val).split()[0])


def next_zoom_index(base: Path) -> int:
    if not base.exists():
        return 0
    idxs: list[int] = []
    for p in base.iterdir():
        if p.is_dir() and p.name.isdigit() and len(p.name) == 4:
            idxs.append(int(p.name))
    return (max(idxs) + 1) if idxs else 0


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_zoom_selection_metadata(path: Path, payload: dict) -> None:
    """
    Write a small, structured metadata file describing a zoom selection/run.

    The file is written atomically (write temp + replace).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    tmp.write_text(text)
    tmp.replace(path)


def read_zoom_selection_metadata(path: Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text())


def selection_center_and_radius(meta: dict) -> tuple[np.ndarray, float] | None:
    sel = meta.get("selection", {})
    if not isinstance(sel, dict):
        return None
    center = sel.get("center_mpc", None)
    if center is None:
        return None
    radius = sel.get("selection_radius_mpc", sel.get("kernel_radius_mpc", None))
    if radius is None:
        return None
    return np.asarray(center, dtype=np.float64), float(radius)


def iter_existing_zoom_selections(out_base: Path) -> list[dict]:
    out_base = Path(out_base)
    if not out_base.exists():
        return []
    metas: list[dict] = []
    for p in sorted(out_base.iterdir()):
        if not (p.is_dir() and p.name.isdigit() and len(p.name) == 4):
            continue
        meta_path = p / "zoom_selection.json"
        if not meta_path.exists():
            continue
        try:
            meta = read_zoom_selection_metadata(meta_path)
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        meta["_meta_path"] = str(meta_path)
        meta["_run_dir"] = str(p)
        metas.append(meta)
    return metas


def periodic_distance(a: np.ndarray, b: np.ndarray, box: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = unwrap_relative(b[None, :], a, float(box))[0]
    return float(np.linalg.norm(delta))


def existing_labels(out_base: Path) -> set[str]:
    labels: set[str] = set()
    for meta in iter_existing_zoom_selections(out_base):
        lab = meta.get("label", None)
        if isinstance(lab, str) and lab:
            labels.add(lab)
    return labels


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    plot_fov = 75.0  # Mpc, fallback if no points plotted

    # Use a single, global marker-size scaling so halo marker sizes are consistent across
    # all projections (and between the two rows when applicable).
    halo_logm_lo: float | None = None
    halo_logm_hi: float | None = None
    logm_ref_parts: list[np.ndarray] = []
    if halo_rel is not None and halo_masses is not None:
        hm0 = np.asarray(halo_masses, dtype=np.float64)
        hr0 = np.asarray(halo_rel, dtype=np.float64)
        env0 = np.all(np.abs(hr0) <= float(env_half), axis=1)
        m0 = env0 & (hm0 >= float(halo_mmin))
        if np.any(m0):
            logm_ref_parts.append(np.log10(np.maximum(hm0[m0], 1.0)))
    if halo_failed_masses is not None and halo_failed_rel is not None:
        fh0 = np.asarray(halo_failed_masses, dtype=np.float64)
        fr0 = np.asarray(halo_failed_rel, dtype=np.float64)
        envf = np.all(np.abs(fr0) <= float(env_half), axis=1)
        mf = envf & (fh0 >= float(halo_mmin))
        if np.any(mf):
            logm_ref_parts.append(np.log10(np.maximum(fh0[mf], 1.0)))
    if logm_ref_parts:
        logm_ref = np.concatenate(logm_ref_parts, axis=0)
        if logm_ref.size:
            lo, hi = np.percentile(logm_ref, [5, 99.5]) if logm_ref.size > 5 else (logm_ref.min(), logm_ref.max())
            halo_logm_lo = float(lo)
            halo_logm_hi = float(hi if hi > lo else lo + 1e-6)

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

    def set_limits_from_points(ax, pts2d: np.ndarray, *, pad_frac: float = 0.03, min_pad: float = 1.0) -> None:
        if pts2d.size == 0:
            ax.set_xlim(-plot_fov, plot_fov)
            ax.set_ylim(-plot_fov, plot_fov)
            return
        xs = pts2d[:, 0]
        ys = pts2d[:, 1]
        xmin = float(np.min(xs))
        xmax = float(np.max(xs))
        ymin = float(np.min(ys))
        ymax = float(np.max(ys))
        rx = xmax - xmin
        ry = ymax - ymin
        pad = max(min_pad, pad_frac * max(rx, ry, 1e-12))
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

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
        focus_half: float | None = None,
    ):
        for proj, (ax, (i, j, xi, yi)) in enumerate(zip(row_axes, planes)):
            k = slab_axes[proj]
            env_mask = np.abs(env_pts[:, k]) <= slab_half
            env_xy = env_pts[env_mask][:, [i, j]]
            box_xy = box_pts[:, [i, j]]
            sel_xy = sel_pts[:, [i, j]]
            ax.scatter(env_xy[:, 0], env_xy[:, 1], s=0.3, alpha=0.1, color="gray", rasterized=True)
            ax.scatter(box_xy[:, 0], box_xy[:, 1], s=0.5, alpha=0.15, color="tab:blue", rasterized=True)
            ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=2.0, alpha=0.35, color="tab:orange", rasterized=True)

            lim_pts = [env_xy, box_xy, sel_xy]
            if halo_ic_pts is not None and halo_ic_pts.size:
                halo_ic_xy = halo_ic_pts[:, [i, j]]
                ax.scatter(
                    halo_ic_xy[:, 0],
                    halo_ic_xy[:, 1],
                    s=0.8,
                    alpha=0.5,
                    color="tab:green",
                    rasterized=True,
                )
                lim_pts.append(halo_ic_xy)
            if draw_haloes and halo_rel is not None and halo_masses is not None:
                hm = np.asarray(halo_masses, dtype=np.float64)
                hr = np.asarray(halo_rel, dtype=np.float64)
                mcut = hm >= halo_mmin
                if np.any(mcut):
                    env_cut = np.all(np.abs(hr) <= env_half, axis=1)
                    hslab = mcut & env_cut & (np.abs(hr[:, k]) <= slab_half)
                    if np.any(hslab):
                        hr_xy = hr[hslab][:, [i, j]]
                        logm = np.log10(hm[hslab])
                        if halo_logm_lo is None or halo_logm_hi is None:
                            lo, hi = np.percentile(logm, [5, 99.5]) if logm.size > 5 else (logm.min(), logm.max())
                            lo = float(lo)
                            hi = float(hi if hi > lo else lo + 1e-6)
                        else:
                            lo = float(halo_logm_lo)
                            hi = float(halo_logm_hi)
                        sizes = 10.0 + 150.0 * (np.clip(logm, lo, hi) - lo) / (hi - lo)
                        ax.scatter(
                            hr_xy[:, 0],
                            hr_xy[:, 1],
                            s=sizes,
                            alpha=0.35,
                            color="tab:purple",
                            rasterized=True,
                        )
                        lim_pts.append(hr_xy)
            if draw_haloes and halo_failed_rel is not None and halo_failed_masses is not None:
                fh = np.asarray(halo_failed_masses, dtype=np.float64)
                fr = np.asarray(halo_failed_rel, dtype=np.float64)
                if fr.size:
                    env_cut = np.all(np.abs(fr) <= env_half, axis=1)
                    fr = fr[env_cut]
                    fh = fh[env_cut]
                if fr.size:
                    fr_xy = fr[:, [i, j]]
                    flogm = np.log10(np.maximum(fh, 1.0))
                    if halo_logm_lo is None or halo_logm_hi is None:
                        flo, fhi = np.percentile(flogm, [5, 99.5]) if flogm.size > 5 else (flogm.min(), flogm.max())
                        flo = float(flo)
                        fhi = float(fhi if fhi > flo else flo + 1e-6)
                    else:
                        flo = float(halo_logm_lo)
                        fhi = float(halo_logm_hi)
                    fsizes = 30.0 + 220.0 * (np.clip(flogm, flo, fhi) - flo) / (fhi - flo)
                    ax.scatter(
                        fr_xy[:, 0],
                        fr_xy[:, 1],
                        s=fsizes,
                        alpha=0.9,
                        color="tab:green",
                        marker="x",
                        linewidths=1.2,
                        rasterized=True,
                    )
                    lim_pts.append(fr_xy)
            if draw_haloes and kernel_radius is not None:
                import matplotlib.patches as mpatches

                circ = mpatches.Circle((0.0, 0.0), float(kernel_radius), fill=False, lw=1.0, ls="--", color="k", alpha=0.7)
                ax.add_patch(circ)
                if kernel_boundary_L is not None and float(kernel_boundary_L) > 0:
                    r_in = max(0.0, float(kernel_radius) - float(kernel_boundary_L))
                    r_out = float(kernel_radius) + float(kernel_boundary_L)
                    ax.add_patch(mpatches.Circle((0.0, 0.0), r_in, fill=False, lw=0.8, ls=":", color="k", alpha=0.5))
                    ax.add_patch(mpatches.Circle((0.0, 0.0), r_out, fill=False, lw=0.8, ls=":", color="k", alpha=0.5))
                    lim_pts.append(np.array([[-r_out, -r_out], [r_out, r_out]], dtype=np.float64))
            if draw_box:
                draw_bbox(ax, i, j)
                lim_pts.append(np.array([[mins[i], mins[j]], [maxs[i], maxs[j]]], dtype=np.float64))
                if mins_raw is not None and maxs_raw is not None:
                    lim_pts.append(np.array([[mins_raw[i], mins_raw[j]], [maxs_raw[i], maxs_raw[j]]], dtype=np.float64))

            if focus_half is not None:
                fh = float(focus_half)
                ax.set_xlim(-fh, fh)
                ax.set_ylim(-fh, fh)
            else:
                set_limits_from_points(ax, np.vstack([p for p in lim_pts if p.size]))
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
        "z=0 (selection sphere dashed, environment slab in gray)",
        sel_z0,
        box_z0,
        env_z0,
        halo_ic_pts=None,
        draw_box=False,
        focus_half=(
            None
            if kernel_radius is None
            else 3 * (float(kernel_radius) + (0.0 if kernel_boundary_L is None else float(kernel_boundary_L)))
        ),
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


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


def write_swift_zoom_yaml(
    *,
    template: Path,
    out_path: Path,
    run_dir: Path,
    ic_path: Path,
    omega_m: float,
    omega_b: float,
    h: float,
    boxlength_mpc_h: float,
    levelmax: int,
) -> None:
    # Some SWIFT templates in this repo contain literal tab characters (e.g. in comments),
    # which PyYAML rejects. Normalize tabs to spaces before parsing.
    text = template.read_text()
    if "\t" in text:
        text = text.expandtabs(2)
    params = yaml.safe_load(text)
    if not isinstance(params, dict):
        raise RuntimeError(f"{template}: expected YAML mapping at root")

    omega_lambda = 1.0 - omega_m
    omega_cdm = omega_m - omega_b

    params.setdefault("Cosmology", {})
    params["Cosmology"]["Omega_cdm"] = float(omega_cdm)
    params["Cosmology"]["Omega_lambda"] = float(omega_lambda)
    params["Cosmology"]["Omega_b"] = float(omega_b)
    params["Cosmology"]["h"] = float(h)

    params.setdefault("InitialConditions", {})
    params["InitialConditions"]["file_name"] = str(ic_path)

    # Mean inter-particle separation for a (2^levelmax)^3 particle load across the full box.
    # MUSIC2 uses boxlength in Mpc/h; SWIFT internal units here are Mpc (no h).
    spacing_mpc = (float(boxlength_mpc_h) / (2**int(levelmax))) / float(h)
    soft_mpc = spacing_mpc / 25.0
    params.setdefault("Gravity", {})
    params["Gravity"]["comoving_DM_softening"] = float(soft_mpc)
    params["Gravity"]["max_physical_DM_softening"] = float(soft_mpc)
    # Keep ratio consistent if present in template.
    if "softening_ratio_background" in params["Gravity"]:
        params["Gravity"]["softening_ratio_background"] = float(1.0 / 25.0)

    # Ensure outputs go into the run directory by using path-valued basenames.
    params.setdefault("Snapshots", {})
    params["Snapshots"]["basename"] = "snap"
    if "FOF" in params and isinstance(params["FOF"], dict):
        params["FOF"]["basename"] = "fof"
    if "CSDS" in params and isinstance(params["CSDS"], dict):
        params["CSDS"]["basename"] = "csds_index"

    # SWIFT's parameter reader expects some sequences to be written in flow-style (e.g. `[0, 1, 0]`)
    # rather than block-style dashes. Force flow-style for all YAML sequences.
    class _SwiftDumper(yaml.SafeDumper):
        pass

    def _repr_seq(dumper: yaml.SafeDumper, data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    _SwiftDumper.add_representer(list, _repr_seq)
    _SwiftDumper.add_representer(tuple, _repr_seq)

    # NOTE: use `yaml.dump` (not `safe_dump`) so we can pass a custom SafeDumper subclass.
    out_path.write_text(
        yaml.dump(
            params,
            sort_keys=False,
            Dumper=_SwiftDumper,
            default_flow_style=False,
            width=120,
        )
    )


def write_music_zoom_config(
    template_path: Path,
    out_path: Path,
    parent_cfg: configparser.ConfigParser,
    center_abs: np.ndarray,
    extent_abs: np.ndarray,
    boxlength_mpc_h: float,
    coord_box_size: float,
    base_levelmin: int,
    base_levelmax: int,
    out_ics: Path | None,
) -> None:
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    with template_path.open() as fh:
        cfg.read_file(fh)

    parent_levelmin = _parse_int(parent_cfg["setup"]["levelmin"])
    parent_levelmax = _parse_int(parent_cfg["setup"].get("levelmax", parent_levelmin))

    parent_seeds = {k: v for k, v in parent_cfg["random"].items() if k.startswith("seed[")}
    seeds = dict(parent_seeds)

    lvl_key_parent = f"seed[{parent_levelmin}]"
    if lvl_key_parent in parent_seeds:
        base_seed = int(parent_seeds[lvl_key_parent])
    elif parent_seeds:
        base_seed = int(next(iter(parent_seeds.values())))
        seeds.setdefault(lvl_key_parent, str(base_seed))
    else:
        base_seed = 424242
        seeds[lvl_key_parent] = str(base_seed)

    rng = random.Random(base_seed)
    for lvl in range(parent_levelmax + 1, base_levelmax + 1):
        seeds[f"seed[{lvl}]"] = str(rng.randint(1, 2**31 - 1))

    # MUSIC2 expects `boxlength` in Mpc/h, while SWIFT-format coordinates/BoxSize are in Mpc.
    # We therefore express the zoom region in *dimensionless fractions* of the coordinate box
    # (usually the parent IC file Header.BoxSize), which is consistent regardless of h.
    center_frac = center_abs / float(coord_box_size)
    extent_frac = extent_abs / float(coord_box_size)

    cfg["setup"]["levelmin"] = str(base_levelmin)
    cfg["setup"]["levelmin_TF"] = str(parent_levelmin)
    cfg["setup"]["levelmax"] = str(base_levelmax)
    cfg["setup"]["boxlength"] = str(boxlength_mpc_h)
    cfg["setup"]["ref_center"] = ", ".join(f"{c:.6f}" for c in center_frac)
    cfg["setup"]["ref_extent"] = ", ".join(f"{e:.6f}" for e in extent_frac)

    # Keep cosmology consistent with the parent ICs. This affects the SWIFT Header.BoxSize
    # (via boxlength/h) and particle masses.
    if "cosmology" not in cfg:
        cfg.add_section("cosmology")
    if "cosmology" in parent_cfg:
        for key in ("Omega_m", "Omega_b", "Omega_L", "H0", "n_s", "sigma_8", "transfer"):
            if key in parent_cfg["cosmology"]:
                cfg["cosmology"][key] = str(parent_cfg["cosmology"][key]).split()[0]

    if out_ics is not None:
        cfg["output"]["filename"] = str(out_ics)

    if "random" not in cfg:
        cfg.add_section("random")
    for k, v in seeds.items():
        cfg["random"][k] = str(v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        cfg.write(fh)


def infer_id_offset(path: Path) -> tuple[int, int]:
    """
    Infer whether ParticleIDs are contiguous and map to the particle index with a constant offset:
        ParticleID == offset + index
    Returns (offset, n). Raises if this is not true.
    """
    import h5py

    with h5py.File(path, "r") as f:
        ids = f["PartType1/ParticleIDs"]
        n = int(ids.shape[0])
        head = np.array(ids[:1024], dtype=np.int64)
        tail = np.array(ids[max(0, n - 1024):], dtype=np.int64)

    if not (np.all(np.diff(head) == 1) and np.all(np.diff(tail) == 1)):
        raise RuntimeError(f"{path}: ParticleIDs not contiguous in sampled ranges; cannot map ID->index directly.")
    offset = int(head[0])
    if int(tail[-1]) != offset + (n - 1):
        raise RuntimeError(f"{path}: ParticleIDs do not match offset+index convention (offset={offset}, n={n}).")
    return offset, n


def sample_box_particles(
    ic_path: Path,
    center: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """
    Reservoir-sample particles inside the IC-space bounding box.
    Also accumulates the total particle count and total mass inside the box.

    Returns (ids_sample, rel_positions_sample, box_size, n_in_box, mass_in_box).
    """
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(ic_path, "r") as f:
        ids_ds = f["PartType1/ParticleIDs"]
        pos_ds = f["PartType1/Coordinates"]
        masses_ds = f["PartType1/Masses"] if "Masses" in f["PartType1"] else None
        box = np.array(f["Header"].attrs["BoxSize"], dtype=np.float64)
        box_size = float(box.item()) if box.ndim == 0 else float(box[0])
        n = int(pos_ds.shape[0])

        chunk = 2_000_000
        sample_ids: list[int] = []
        sample_pos: list[np.ndarray] = []
        seen = 0
        n_in_box = 0
        mass_in_box = 0.0

        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            pos = np.array(pos_ds[start:stop], dtype=np.float64)
            ids = np.array(ids_ds[start:stop], dtype=np.uint64)
            masses = None if masses_ds is None else np.array(masses_ds[start:stop], dtype=np.float64)
            rel = unwrap_relative(pos, center, box_size)
            mask = (
                (rel[:, 0] >= mins[0]) & (rel[:, 0] <= maxs[0]) &
                (rel[:, 1] >= mins[1]) & (rel[:, 1] <= maxs[1]) &
                (rel[:, 2] >= mins[2]) & (rel[:, 2] <= maxs[2])
            )
            if not np.any(mask):
                continue
            rel = rel[mask]
            ids = ids[mask]
            if masses is None:
                n_in_box += int(ids.size)
            else:
                masses = masses[mask]
                n_in_box += int(ids.size)
                mass_in_box += float(masses.sum())

            for pid, rpos in zip(ids, rel):
                seen += 1
                if len(sample_ids) < max_samples:
                    sample_ids.append(int(pid))
                    sample_pos.append(rpos)
                else:
                    j = rng.integers(0, seen)
                    if j < max_samples:
                        sample_ids[j] = int(pid)
                        sample_pos[j] = rpos

    if masses_ds is None:
        raise RuntimeError(f"{ic_path}: IC file missing PartType1/Masses; cannot compute bounding-box mass.")

    return (
        np.array(sample_ids, dtype=np.uint64),
        np.array(sample_pos, dtype=np.float64),
        box_size,
        n_in_box,
        mass_in_box,
    )


def positions_for_ids_by_offset(
    snap_path: Path,
    ids: np.ndarray,
    offset: int,
    center: np.ndarray,
    box_size: float,
) -> np.ndarray:
    """
    Fetch coordinates for a list of ParticleIDs using the offset+index convention.
    Returns centered positions (wrapped into [-box/2, box/2]).
    """
    import h5py

    idx = ids.astype(np.int64) - int(offset)
    if np.any(idx < 0):
        raise RuntimeError("Negative indices after applying ID offset; check ParticleID convention.")
    order = np.argsort(idx)
    idx_sorted = idx[order]
    with h5py.File(snap_path, "r") as f:
        pos_sorted = np.array(f["PartType1/Coordinates"][idx_sorted, :], dtype=np.float64)
    pos = np.empty_like(pos_sorted)
    pos[order] = pos_sorted
    return unwrap_relative(pos, center, box_size)


def positions_for_ids_by_scan(
    snap_path: Path,
    ids: np.ndarray,
    center: np.ndarray,
    box_size: float,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """
    Fetch coordinates for a list of ParticleIDs by scanning the snapshot once.

    This is slower than the offset+index convention but works when ParticleIDs are
    not contiguous / not ordered in the file.
    """
    import h5py

    ids = np.asarray(ids, dtype=np.uint64)
    order_in = np.argsort(ids)
    ids_sorted = ids[order_in]
    ids_unique = np.unique(ids_sorted)

    found: dict[int, np.ndarray] = {}
    with h5py.File(snap_path, "r") as f:
        ids_ds = f["PartType1/ParticleIDs"]
        pos_ds = f["PartType1/Coordinates"]
        n = int(ids_ds.shape[0])

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            ids_chunk = np.array(ids_ds[start:stop], dtype=np.uint64)
            # vectorized membership via searchsorted on sorted unique IDs
            idx = np.searchsorted(ids_unique, ids_chunk)
            inb = idx < ids_unique.size
            mask = np.zeros_like(inb, dtype=bool)
            if np.any(inb):
                mask[inb] = ids_unique[idx[inb]] == ids_chunk[inb]
            if not np.any(mask):
                continue
            pos_chunk = np.array(pos_ds[start:stop], dtype=np.float64)
            hit_ids = ids_chunk[mask]
            hit_pos = pos_chunk[mask]
            for pid, p in zip(hit_ids, hit_pos):
                found.setdefault(int(pid), p)
            if len(found) == ids_unique.size:
                break

    if len(found) != ids_unique.size:
        missing = ids_unique[[int(x) not in found for x in ids_unique]]
        raise RuntimeError(
            f"Did not find all ParticleIDs in {snap_path}: missing {missing.size} / {ids_unique.size} "
            f"(first few missing: {missing[:10]})"
        )

    pos_sorted = np.stack([found[int(pid)] for pid in ids_sorted], axis=0)
    pos = np.empty_like(pos_sorted)
    pos[order_in] = pos_sorted
    return unwrap_relative(pos, center, box_size)


def load_fof_catalogue(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load a SWIFT-style FOF catalogue.

    Returns
    -------
    centres : (N, 3) float64
    masses : (N,) float64
    radii : (N,) float64
    """
    import h5py

    with h5py.File(path, "r") as f:
        centres = np.array(f["Groups/Centres"], dtype=np.float64)
        masses = np.array(f["Groups/Masses"], dtype=np.float64)
        radii = np.array(f["Groups/Radii"], dtype=np.float64) if "Groups/Radii" in f else np.zeros(len(masses))
        box = np.array(f["Header"].attrs["BoxSize"], dtype=np.float64)
        box_size = float(box.item()) if box.ndim == 0 else float(box[0])
    return centres, masses, radii, box_size


def fof_haloes_relative(path: Path, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centres, masses, _radii, box_size = load_fof_catalogue(path)
    rel = unwrap_relative(centres, center, box_size)
    return rel, masses


def infer_contiguous_id_offset(ids: np.ndarray) -> int | None:
    """
    Return offset if `ids[i] == offset + i` for file ordering, else None.

    We rely on this fast mapping for IC files to avoid sorting 100M+ ParticleIDs.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size < 2:
        return None
    n = ids.size
    head = ids[: min(4096, n)]
    tail = ids[max(0, n - 4096):]
    if not (np.all(np.diff(head) == 1) and np.all(np.diff(tail) == 1)):
        return None
    offset = int(head[0])
    if int(tail[-1]) != offset + (n - 1):
        return None
    return offset
