#!/usr/bin/env python
"""
Shared helper methods for Tessera scripts.

This file collects reusable utilities extracted from `generate_zoom_ics.py` to keep
the main script focused on orchestration/CLI logic.
"""

import configparser
import random
import sys
from pathlib import Path

import numpy as np
import yaml


def _ensure_pmwd_utilities_on_path() -> None:
    pmwd_scripts = Path("/cosma7/data/dp004/dc-love2/codes/pmwd_zoom_selection/scripts")
    if not pmwd_scripts.exists():
        root = Path(__file__).resolve().parent.parent
        pmwd_scripts = (root.parent / "pmwd_zoom_selection" / "scripts").resolve()
    if pmwd_scripts.exists() and str(pmwd_scripts) not in sys.path:
        sys.path.insert(0, str(pmwd_scripts))


_ensure_pmwd_utilities_on_path()

try:
    from utilities import unwrap_relative  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Could not import `utilities` from pmwd_zoom_selection; expected it at "
        "`/cosma7/data/dp004/dc-love2/codes/pmwd_zoom_selection/scripts/utilities.py`."
    ) from exc


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
    params["Snapshots"]["basename"] = str(run_dir / "snap")
    if "FOF" in params and isinstance(params["FOF"], dict):
        params["FOF"]["basename"] = str(run_dir / "fof")
    if "CSDS" in params and isinstance(params["CSDS"], dict):
        params["CSDS"]["basename"] = str(run_dir / "csds_index")

    out_path.write_text(yaml.safe_dump(params, sort_keys=False))


def write_music_zoom_config(
    template_path: Path,
    out_path: Path,
    parent_cfg: configparser.ConfigParser,
    center_abs: np.ndarray,
    extent_abs: np.ndarray,
    box_size: float,
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

    center_frac = center_abs / box_size
    extent_frac = extent_abs / box_size

    cfg["setup"]["levelmin"] = str(base_levelmin)
    cfg["setup"]["levelmin_TF"] = str(parent_levelmin)
    cfg["setup"]["levelmax"] = str(base_levelmax)
    cfg["setup"]["boxlength"] = str(box_size)
    cfg["setup"]["ref_center"] = ", ".join(f"{c:.6f}" for c in center_frac)
    cfg["setup"]["ref_extent"] = ", ".join(f"{e:.6f}" for e in extent_frac)

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

