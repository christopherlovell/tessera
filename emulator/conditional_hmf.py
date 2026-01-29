#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp

MSUN_CGS = 1.98841e33
MASS_PIVOT_MSUN = 1e10  # Plot/fit masses in units of 1e10 Msun.


def read_swift_fof(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    import h5py

    with h5py.File(path, "r") as f:
        centres = np.asarray(f["Groups/Centres"], dtype=np.float64)
        masses = np.asarray(f["Groups/Masses"], dtype=np.float64)
        box = np.asarray(f["Header"].attrs["BoxSize"], dtype=np.float64)
        boxsize = float(box.item()) if box.ndim == 0 else float(box[0])
        redshift = float(np.asarray(f["Header"].attrs["Redshift"])[0])
        unit_mass_cgs = float(np.asarray(f["Units"].attrs["Unit mass in cgs (U_M)"])[0])
    return centres, masses * (unit_mass_cgs / MSUN_CGS), boxsize, redshift


def load_gridder_overdensity(path: Path, *, kernel_radius: float) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        pos = np.asarray(f["Grids/GridPointPositions"], dtype=np.float64)
        kernel_group = None
        for name in f["Grids"]:
            if not str(name).startswith("Kernel_"):
                continue
            g = f["Grids"][name]
            r = float(np.asarray(g.attrs.get("KernelRadius", np.nan)))
            if np.isfinite(r) and np.isclose(r, kernel_radius, rtol=0, atol=1e-8):
                kernel_group = g
                break
        if kernel_group is None:
            raise KeyError(f"{path}: no kernel with KernelRadius={kernel_radius}")
        delta = np.asarray(kernel_group["GridPointOverDensities"], dtype=np.float64)
    return pos, delta


def log10_mass_bins(
    masses_msun: np.ndarray,
    *,
    n_bins: int,
    log10M_min: float | None = None,
    log10M_max: float | None = None,
    last_bin_ratio: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses_msun = np.asarray(masses_msun, dtype=np.float64)
    log10M = np.log10(np.clip(masses_msun, 1e-300, None)) - np.log10(MASS_PIVOT_MSUN)
    if log10M_min is None:
        log10M_min = float(np.min(log10M))
    if log10M_max is None:
        log10M_max = float(np.max(log10M))
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    last_bin_ratio = float(last_bin_ratio)
    if not np.isfinite(last_bin_ratio) or last_bin_ratio <= 0.0:
        raise ValueError(f"last_bin_ratio must be finite and > 0 (got {last_bin_ratio})")

    # Use equal-width bins except allow the final bin to be wider by `last_bin_ratio`.
    # The total range [min,max] is preserved by shrinking the earlier bins accordingly.
    if np.isclose(last_bin_ratio, 1.0, rtol=0, atol=1e-12):
        edges = np.linspace(log10M_min, log10M_max, n_bins + 1)
    else:
        total = float(log10M_max - log10M_min)
        base = total / float((n_bins - 1) + last_bin_ratio)
        widths = np.full(n_bins, base, dtype=np.float64)
        widths[-1] = base * last_bin_ratio
        edges = log10M_min + np.concatenate([[0.0], np.cumsum(widths)])
        edges[-1] = float(log10M_max)
    dlog10M = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, dlog10M


def hmf_dn_dlog10M(masses_msun: np.ndarray, volume_mpc3: float, log10M_edges: np.ndarray) -> np.ndarray:
    log10M = np.log10(np.clip(np.asarray(masses_msun, dtype=np.float64), 1e-300, None)) - np.log10(MASS_PIVOT_MSUN)
    counts, _ = np.histogram(log10M, bins=np.asarray(log10M_edges, dtype=np.float64))
    dlog10M = np.diff(log10M_edges)
    return counts / (float(volume_mpc3) * dlog10M)

def baseline_from_spheres(
    N: np.ndarray,
    V: np.ndarray,
    dlog10M: np.ndarray,
    *,
    pseudocount: float = 0.5,
    sphere_weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate an unconditional/baseline HMF from the (sphere, mass-bin) counts.

    This is the Poisson MLE for the per-bin mean number density, aggregating all spheres:
        n_j = (sum_s N_{s,j} + pseudocount) / (sum_s V_s * dlog10M_j)

    If `sphere_weights` is provided (shape (S,)), we compute the weighted analogue:
        n_j = (sum_s w_s N_{s,j} + pseudocount) / (sum_s w_s V_s * dlog10M_j).

    Note: this matches the sampling distribution of the spheres. If spheres are selected
    stratified in overdensity rather than drawn from the true p(delta), this baseline is
    not an unbiased estimate of the global-box HMF.
    """
    N = np.asarray(N, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dlog10M = np.asarray(dlog10M, dtype=np.float64)
    if sphere_weights is None:
        tot_counts = np.sum(N, axis=0) + float(pseudocount)
        tot_vol = float(np.sum(V))
    else:
        w = np.asarray(sphere_weights, dtype=np.float64).reshape(-1)
        if w.size != N.shape[0]:
            raise ValueError(f"sphere_weights must have shape (S,) with S={N.shape[0]} (got {w.shape})")
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
        if not np.any(w > 0.0):
            w = np.ones_like(w)
        w = w / float(np.mean(w))
        tot_counts = np.sum(w[:, None] * N, axis=0) + float(pseudocount)
        tot_vol = float(np.sum(w * V))
    return tot_counts / (tot_vol * dlog10M)

def overdensity_pdf_weights(
    *,
    delta_spheres: np.ndarray,
    delta_parent: np.ndarray,
    nbins: int = 120,
    clip_min: float = -1.0 + 1e-12,
) -> np.ndarray:
    """
    Return per-sphere weights proportional to the *global* overdensity PDF evaluated at each sphere's δ.

    We estimate p(δ) from the parent grid's overdensity values using a histogram in
    x = log10(1+δ) (as used elsewhere for plotting/binning), then convert to p(δ):
        p(δ) = p(x) / (ln 10 * (1+δ)).

    The returned weights are proportional to p(δ) (up to histogram discretization). Any overall
    normalization cancels in downstream weighted aggregation.
    """
    delta_parent = np.asarray(delta_parent, dtype=np.float64).ravel()
    if delta_parent.size == 0:
        return np.ones_like(np.asarray(delta_spheres, dtype=np.float64).ravel())

    nbins = int(nbins)
    if nbins <= 1:
        raise ValueError("nbins must be > 1")

    clip_min = float(clip_min)
    chunk = 2_000_000

    # Pass 1: determine x-range without allocating x for the full grid.
    x_min = np.inf
    x_max = -np.inf
    n_finite = 0
    for start in range(0, int(delta_parent.size), int(chunk)):
        d = delta_parent[start : start + int(chunk)]
        m = np.isfinite(d)
        if not np.any(m):
            continue
        d = np.clip(d[m], clip_min, None)
        x = np.log10(1.0 + d)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        n_finite += int(x.size)
        x_min = min(float(x_min), float(np.min(x)))
        x_max = max(float(x_max), float(np.max(x)))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or n_finite == 0 or not (x_max > x_min):
        return np.ones_like(np.asarray(delta_spheres, dtype=np.float64).ravel())

    edges = np.linspace(float(x_min), float(x_max), nbins + 1, dtype=np.float64)
    counts = np.zeros(nbins, dtype=np.float64)

    # Pass 2: histogram counts.
    for start in range(0, int(delta_parent.size), int(chunk)):
        d = delta_parent[start : start + int(chunk)]
        m = np.isfinite(d)
        if not np.any(m):
            continue
        d = np.clip(d[m], clip_min, None)
        x = np.log10(1.0 + d)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        c, _ = np.histogram(x, bins=edges, density=False)
        counts += np.asarray(c, dtype=np.float64)

    sw = float(np.sum(counts))
    if not np.isfinite(sw) or sw <= 0.0:
        return np.ones_like(np.asarray(delta_spheres, dtype=np.float64).ravel())

    widths = np.diff(edges)
    hist = counts / (sw * widths)

    ds = np.asarray(delta_spheres, dtype=np.float64).ravel()
    ds_clip = np.clip(ds, float(clip_min), None)
    x_s = np.log10(1.0 + ds_clip)
    bin_idx = np.searchsorted(edges, x_s, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, hist.size - 1)
    pdf_x = hist[bin_idx]
    pdf_delta = pdf_x / (np.log(10.0) * (1.0 + ds_clip))
    pdf_delta = np.where(np.isfinite(pdf_delta) & (pdf_delta > 0.0), pdf_delta, 0.0)

    pos = pdf_delta[pdf_delta > 0.0]
    floor = float(np.min(pos)) if pos.size else 1.0
    pdf_delta = np.where(pdf_delta > 0.0, pdf_delta, floor)
    return pdf_delta


def _choose_centers_stratified(delta: np.ndarray, n: int, seed: int, n_bins: int = 10) -> np.ndarray:
    """
    Choose `n` indices stratified in log-overdensity space.

    Stratification is performed in x = log10(1 + delta) using equal-width bins in x.
    We then draw (approximately) equal numbers of samples from each bin.

    This intentionally over-represents the tails of the parent p(delta) compared to
    simple random sampling, and is useful when training conditional models that need
    coverage across the full overdensity range.
    """
    delta = np.asarray(delta, dtype=np.float64).ravel()
    n = int(n)
    if n <= 0 or n > delta.size:
        raise ValueError(f"n={n} invalid for delta.size={delta.size}")

    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0 (got {n_bins})")
    n_bins = min(n_bins, n)

    clip_min = -1.0 + 1e-12
    finite = np.isfinite(delta)
    valid = finite & (delta > float(clip_min))
    if not np.any(valid):
        raise ValueError("No valid delta samples for stratified selection.")

    x = np.full(delta.shape, np.nan, dtype=np.float64)
    x[valid] = np.log10(1.0 + np.clip(delta[valid], float(clip_min), None))
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or not (xmax > xmin):
        rng = np.random.default_rng(int(seed))
        return rng.choice(np.arange(delta.size, dtype=np.int64), size=n, replace=False)

    edges = np.linspace(xmin, xmax, n_bins + 1, dtype=np.float64)

    # Candidate indices per bin.
    cand_bins: list[np.ndarray] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            m = valid & (x >= lo) & (x <= hi)
        else:
            m = valid & (x >= lo) & (x < hi)
        cand_bins.append(np.nonzero(m)[0].astype(np.int64))

    rng = np.random.default_rng(int(seed))

    # Target counts per bin (as equal as possible).
    base = n // n_bins
    rem = n - base * n_bins
    targets = np.full(n_bins, base, dtype=np.int64)
    if rem > 0:
        # Distribute remainder across bins uniformly.
        extra_bins = rng.choice(np.arange(n_bins, dtype=np.int64), size=rem, replace=False)
        targets[extra_bins] += 1

    chosen: list[int] = []
    used = np.zeros(delta.size, dtype=bool)
    deficits = 0
    for i in range(n_bins):
        need = int(targets[i])
        if need <= 0:
            continue
        cand = cand_bins[i]
        if cand.size == 0:
            deficits += need
            continue
        k = min(need, int(cand.size))
        pick = rng.choice(cand, size=k, replace=False)
        for ii in np.asarray(pick, dtype=np.int64).tolist():
            used[int(ii)] = True
            chosen.append(int(ii))
        deficits += max(0, need - k)

    # Fill any deficits from the remaining pool (still uniform in x-bins as much as possible).
    if deficits > 0:
        remaining = np.nonzero(valid & (~used))[0].astype(np.int64)
        if remaining.size < deficits:
            raise RuntimeError(f"Stratified selection underfilled: need {deficits}, have {remaining.size} remaining.")
        fill = rng.choice(remaining, size=int(deficits), replace=False)
        chosen.extend([int(ii) for ii in np.asarray(fill, dtype=np.int64).tolist()])

    chosen = np.asarray(chosen, dtype=np.int64)
    if chosen.size != n:
        # As a last resort, enforce exact size via random trimming/fill (should be rare).
        if chosen.size > n:
            chosen = rng.choice(chosen, size=n, replace=False)
        else:
            remaining = np.nonzero(valid & (~used))[0].astype(np.int64)
            fill = rng.choice(remaining, size=(n - chosen.size), replace=False)
            chosen = np.concatenate([chosen, fill], axis=0)

    return np.asarray(chosen, dtype=np.int64)

def _choose_centers_stratified_excluding(
    delta: np.ndarray, n: int, seed: int, exclude_idx: np.ndarray, n_bins: int = 10
) -> np.ndarray:
    """
    Choose `n` indices stratified in `delta`, excluding any indices in `exclude_idx`.

    Returns indices into the original `delta` array.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = int(n)
    exclude_idx = np.asarray(exclude_idx, dtype=np.int64).ravel()
    if exclude_idx.size == 0:
        return _choose_centers_stratified(delta, n, seed, n_bins=n_bins)
    exclude_idx = np.unique(exclude_idx)
    if np.any(exclude_idx < 0) or np.any(exclude_idx >= delta.size):
        raise ValueError("exclude_idx contains out-of-range indices")
    mask = np.ones(delta.size, dtype=bool)
    mask[exclude_idx] = False
    avail = np.nonzero(mask)[0].astype(np.int64)
    if n <= 0 or n > avail.size:
        raise ValueError(f"n={n} invalid for available.size={avail.size}")
    sub = _choose_centers_stratified(delta[avail], n, seed, n_bins=n_bins)
    return avail[np.asarray(sub, dtype=np.int64)]


def _periodic_ok(p: np.ndarray, keep_pos: list[np.ndarray], boxsize: float, min_sep2: float) -> bool:
    if not keep_pos:
        return True
    L = float(boxsize)
    arr = np.stack(keep_pos, axis=0)
    d = np.abs(arr - p[None, :])
    d = np.minimum(d, L - d)
    return bool(np.all(np.sum(d * d, axis=1) >= float(min_sep2)))


def _unique_preserve_order(idx: np.ndarray) -> np.ndarray:
    seen: set[int] = set()
    out: list[int] = []
    for ii in np.asarray(idx, dtype=np.int64).tolist():
        if ii in seen:
            continue
        seen.add(int(ii))
        out.append(int(ii))
    return np.asarray(out, dtype=np.int64)


def _select_nonoverlapping_centers_stratified(
    *,
    grid_pos: np.ndarray,
    delta: np.ndarray,
    boxsize: float,
    kernel_radius: float,
    n_select: int,
    n_bins: int = 10,
    seed: int,
    max_attempts: int = 32,
    avoid_pos: list[np.ndarray] | None = None,
    exclude_idx: np.ndarray | None = None,
) -> np.ndarray:
    """
    Choose sphere centers such that spheres of radius `kernel_radius` do not overlap.

    Selection is seeded and enforces stratification in overdensity (one center per stratum)
    with the additional hard constraint that all chosen centers satisfy separation >= 2R
    (periodic distances). Any positions in `avoid_pos` are treated as already occupied.
    """
    grid_pos = np.asarray(grid_pos, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    L = float(boxsize)
    R = float(kernel_radius)
    n_select = int(n_select)
    if n_select <= 0:
        raise ValueError("n_select must be > 0")
    if grid_pos.shape[0] != delta.shape[0]:
        raise ValueError("grid_pos and delta must have the same length")

    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0 (got {n_bins})")
    n_bins = min(n_bins, n_select)

    min_sep2 = (2.0 * R) ** 2
    pos = np.mod(grid_pos, L)

    avoid_pos = [] if avoid_pos is None else list(avoid_pos)
    exclude_set: set[int] = set()
    if exclude_idx is not None:
        for ii in np.asarray(exclude_idx, dtype=np.int64).ravel().tolist():
            exclude_set.add(int(ii))

    clip_min = -1.0 + 1e-12
    finite = np.isfinite(delta)
    valid = finite & (delta > float(clip_min))
    if not np.any(valid):
        raise ValueError("No valid delta samples for stratified non-overlap selection.")

    x = np.full(delta.shape, np.nan, dtype=np.float64)
    x[valid] = np.log10(1.0 + np.clip(delta[valid], float(clip_min), None))
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or not (xmax > xmin):
        raise ValueError("Degenerate log10(1+delta) range; cannot stratify.")

    x_edges = np.linspace(xmin, xmax, n_bins + 1, dtype=np.float64)
    strata: list[np.ndarray] = []
    for i in range(n_bins):
        lo = x_edges[i]
        hi = x_edges[i + 1]
        if i == n_bins - 1:
            m = valid & (x >= lo) & (x <= hi)
        else:
            m = valid & (x >= lo) & (x < hi)
        strata.append(np.nonzero(m)[0].astype(np.int64))
    if not any(s.size > 0 for s in strata):
        raise ValueError("All log-overdensity bins are empty; cannot stratify.")

    for attempt in range(int(max_attempts)):
        rng = np.random.default_rng(int(seed) + attempt)
        keep: list[int] = []
        keep_pos: list[np.ndarray] = list(avoid_pos)
        used: set[int] = set(exclude_set)

        base = n_select // n_bins
        rem = n_select - base * n_bins
        targets = np.full(n_bins, base, dtype=np.int64)
        if rem > 0:
            extra_bins = rng.choice(np.arange(n_bins, dtype=np.int64), size=rem, replace=False)
            targets[extra_bins] += 1

        # Expand into a sequence of bin assignments, then randomize bin order.
        bins: list[int] = []
        for b in range(n_bins):
            bins.extend([int(b)] * int(targets[b]))
        rng.shuffle(bins)
        # We try to place a center for each requested bin draw; if a particular draw
        # can't find a non-overlapping candidate in that bin, treat it as a deficit
        # and fill from the remaining pool later (still enforcing non-overlap).
        deficits = 0
        for b in bins:
            cand = np.asarray(strata[int(b)], dtype=np.int64)
            if cand.size == 0:
                # Bin is empty; we'll fill later from any remaining valid points.
                deficits += 1
                continue
            cand = rng.permutation(cand)

            chosen = None
            for ii in cand.tolist():
                if int(ii) in used:
                    continue
                p = pos[int(ii)]
                if _periodic_ok(p, keep_pos, L, min_sep2):
                    chosen = int(ii)
                    break
            if chosen is None:
                deficits += 1
                continue
            keep.append(int(chosen))
            keep_pos.append(pos[int(chosen)])
            used.add(int(chosen))

        if deficits > 0 or len(keep) < n_select:
            # Fill any missing slots from the remaining valid points, respecting non-overlap.
            remaining = np.nonzero(valid)[0].astype(np.int64)
            if used:
                remaining = remaining[~np.isin(remaining, np.fromiter(used, dtype=np.int64, count=len(used)))]
            remaining = rng.permutation(remaining)
            for ii in remaining.tolist():
                if len(keep) >= n_select:
                    break
                p = pos[int(ii)]
                if _periodic_ok(p, keep_pos, L, min_sep2):
                    keep.append(int(ii))
                    keep_pos.append(p)
                    used.add(int(ii))

        if len(keep) == n_select:
            return np.asarray(keep, dtype=np.int64)

    raise RuntimeError(
        f"Could not select {n_select} non-overlapping spheres (R={R:g}, min center sep={2*R:g}). "
        "Try reducing --n-spheres, reducing --kernel-radius, or enabling overlaps."
    )


@dataclass(frozen=True)
class ParentDataset:
    delta: np.ndarray  # (S,)
    V: np.ndarray  # (S,)
    N: np.ndarray  # (S,J) int
    center_idx: np.ndarray  # (S,) indices into the gridder arrays used for sphere centres
    log10M_edges: np.ndarray  # (J+1,)
    log10M_centers: np.ndarray  # (J,)
    dlog10M: np.ndarray  # (J,)
    log_n_base: np.ndarray  # (J,)
    log_n_box: np.ndarray  # (J,) unconditional HMF from full box using the same mass bins
    delta_grid: np.ndarray  # (Q,) for p(delta)


def build_parent_dataset(
    *,
    parent_fof: Path,
    gridder_file: Path,
    kernel_radius: float,
    n_spheres: int = 512,
    n_delta_bins: int = 10,
    seed: int = 0,
    n_mass_bins: int = 25,
    last_bin_ratio: float = 1.2,
    log10M_min: float | None = None,
    log10M_max: float | None = None,
    n_delta_q: int = 2048,
    baseline: str = "box",
    baseline_weight: str = "uniform",
    baseline_pseudocount: float = 0.5,
    include_top_overdense: int = 0,
    include_top_halos: int = 0,
    allow_overlap: bool = False,
) -> ParentDataset:
    centres_h, masses_h, boxsize, _z = read_swift_fof(parent_fof)
    grid_pos, grid_delta = load_gridder_overdensity(gridder_file, kernel_radius=float(kernel_radius))

    log10M_edges, log10M_centers, dlog10M = log10_mass_bins(
        masses_h,
        n_bins=int(n_mass_bins),
        log10M_min=log10M_min,
        log10M_max=log10M_max,
        last_bin_ratio=float(last_bin_ratio),
    )
    log_n_box = np.log(hmf_dn_dlog10M(masses_h, boxsize**3, log10M_edges) + 1e-300)

    baseline = str(baseline).lower().strip()
    if baseline not in {"box", "spheres", "hybrid"}:
        raise ValueError(f"baseline must be 'box', 'spheres', or 'hybrid' (got {baseline!r})")
    baseline_weight = str(baseline_weight).lower().strip()
    if baseline_weight not in {"uniform", "pdf"}:
        raise ValueError(f"baseline_weight must be 'uniform' or 'pdf' (got {baseline_weight!r})")
    log_n_base: np.ndarray | None = None
    if baseline == "box":
        log_n_base = log_n_box

    n_spheres = int(n_spheres)
    include_top_overdense = int(include_top_overdense)
    include_top_halos = int(include_top_halos)
    if include_top_overdense < 0:
        raise ValueError(f"include_top_overdense must be >= 0 (got {include_top_overdense})")
    if include_top_overdense > n_spheres:
        raise ValueError(
            f"include_top_overdense={include_top_overdense} cannot exceed n_spheres={n_spheres}"
        )
    if include_top_halos < 0:
        raise ValueError(f"include_top_halos must be >= 0 (got {include_top_halos})")
    if include_top_halos > n_spheres:
        raise ValueError(f"include_top_halos={include_top_halos} cannot exceed n_spheres={n_spheres}")

    halo_center_idx = None
    if include_top_halos > 0:
        from scipy.spatial import cKDTree

        # Map the positions of the most massive halos to nearest grid points (periodic).
        L = float(boxsize)
        gp = np.mod(np.asarray(grid_pos, dtype=np.float64), L)
        tree_gp = cKDTree(gp, boxsize=L)
        halo_pos = np.mod(np.asarray(centres_h, dtype=np.float64), L)
        order_h = np.argsort(np.asarray(masses_h, dtype=np.float64))[::-1]
        top_h = order_h[: int(include_top_halos)]
        _, nn = tree_gp.query(halo_pos[top_h], k=1, workers=-1)
        halo_center_idx = np.asarray(nn, dtype=np.int64)

    forced_parts: list[np.ndarray] = []
    if halo_center_idx is not None and halo_center_idx.size:
        forced_parts.append(np.asarray(halo_center_idx, dtype=np.int64))
    if include_top_overdense > 0:
        order_desc = np.argsort(np.asarray(grid_delta, dtype=np.float64))[::-1]
        top_idx = np.asarray(order_desc[:include_top_overdense], dtype=np.int64)
        forced_parts.append(np.asarray(top_idx, dtype=np.int64))
    forced_idx = _unique_preserve_order(np.concatenate(forced_parts, axis=0)) if forced_parts else np.zeros(0, dtype=np.int64)

    if forced_idx.size > n_spheres:
        forced_idx = forced_idx[:n_spheres]

    if bool(allow_overlap):
        n_rem = int(n_spheres) - int(forced_idx.size)
        strat_idx = (
            _choose_centers_stratified_excluding(grid_delta, n_rem, int(seed), forced_idx, n_bins=int(n_delta_bins))
            if n_rem > 0
            else np.zeros(0, dtype=np.int64)
        )
        center_idx = np.concatenate([forced_idx, np.asarray(strat_idx, dtype=np.int64)], axis=0)
        if center_idx.size != n_spheres:
            raise RuntimeError(f"Selection underfilled (got {center_idx.size}, expected {n_spheres}).")
    else:
        # Two-stage selection:
        #  1) force include "halo-selected" and/or top-overdense centres (without overlap),
        #  2) fill remaining slots with a δ-stratified non-overlapping sample excluding forced centres.
        L = float(boxsize)
        R = float(kernel_radius)
        min_sep2 = (2.0 * R) ** 2
        pos = np.mod(np.asarray(grid_pos, dtype=np.float64), L)

        keep_forced: list[int] = []
        keep_forced_pos: list[np.ndarray] = []
        for ii in np.asarray(forced_idx, dtype=np.int64).tolist():
            p = pos[int(ii)]
            if _periodic_ok(p, keep_forced_pos, L, min_sep2):
                keep_forced.append(int(ii))
                keep_forced_pos.append(p)
        forced_keep_idx = np.asarray(keep_forced, dtype=np.int64)

        n_rem = int(n_spheres) - int(forced_keep_idx.size)
        rem_idx = (
            _select_nonoverlapping_centers_stratified(
                grid_pos=grid_pos,
                delta=grid_delta,
                boxsize=float(boxsize),
                kernel_radius=float(kernel_radius),
                n_select=n_rem,
                n_bins=int(n_delta_bins),
                seed=int(seed),
                avoid_pos=keep_forced_pos,
                exclude_idx=forced_keep_idx,
            )
            if n_rem > 0
            else np.zeros(0, dtype=np.int64)
        )
        center_idx = np.concatenate([forced_keep_idx, np.asarray(rem_idx, dtype=np.int64)], axis=0)
        if center_idx.size != n_spheres:
            raise RuntimeError(f"Selection underfilled (got {center_idx.size}, expected {n_spheres}).")
    centers = grid_pos[center_idx]
    delta = grid_delta[center_idx]

    from scipy.spatial import cKDTree

    L = float(boxsize)
    halo_pos = np.mod(np.asarray(centres_h, dtype=np.float64), L)
    tree = cKDTree(halo_pos, boxsize=L)
    hits = tree.query_ball_point(np.mod(centers, L), r=float(kernel_radius), workers=-1)

    J = log10M_centers.size
    N = np.zeros((int(n_spheres), J), dtype=np.int32)
    log10M = np.log10(np.clip(masses_h, 1e-300, None)) - np.log10(MASS_PIVOT_MSUN)
    for s, ids in enumerate(hits):
        if not ids:
            continue
        N[s], _ = np.histogram(log10M[np.asarray(ids, dtype=np.int64)], bins=log10M_edges)

    V = np.full(int(n_spheres), float(4.0 / 3.0 * np.pi * kernel_radius**3), dtype=np.float64)

    if log_n_base is None:
        w_base = None
        if baseline_weight == "pdf":
            w_base = overdensity_pdf_weights(delta_spheres=delta, delta_parent=grid_delta)
        n_base = baseline_from_spheres(
            N, V, dlog10M, pseudocount=float(baseline_pseudocount), sphere_weights=w_base
        )
        log_n_base = np.log(np.asarray(n_base, dtype=np.float64) + 1e-300)
        if baseline == "hybrid":
            # Anchor the highest-mass bin to the unconditional parent-box HMF, but learn delta-dependent
            # deviations from the spheres. This uses full-box information only for the tail normalization.
            if log_n_base.size > 0:
                log_n_base[-1] = log_n_box[-1]

    rng = np.random.default_rng(int(seed) + 1)
    if int(n_delta_q) <= 0:
        delta_grid = np.asarray(grid_delta, dtype=np.float64)
    elif grid_delta.size <= int(n_delta_q):
        delta_grid = np.asarray(grid_delta, dtype=np.float64)
    else:
        delta_grid = rng.choice(np.asarray(grid_delta, dtype=np.float64), size=int(n_delta_q), replace=False)

    return ParentDataset(
        delta=np.asarray(delta, dtype=np.float64),
        V=V,
        N=N,
        center_idx=np.asarray(center_idx, dtype=np.int64),
        log10M_edges=np.asarray(log10M_edges, dtype=np.float64),
        log10M_centers=np.asarray(log10M_centers, dtype=np.float64),
        dlog10M=np.asarray(dlog10M, dtype=np.float64),
        log_n_base=np.asarray(log_n_base, dtype=np.float64),
        log_n_box=np.asarray(log_n_box, dtype=np.float64),
        delta_grid=np.asarray(delta_grid, dtype=np.float64),
    )


def pca_mass_basis(
    N: np.ndarray,
    V: np.ndarray,
    dlog10M: np.ndarray,
    log_n_base: np.ndarray,
    *,
    K: int,
    pseudocount: float = 0.5,
    add_intercept: bool = True,
) -> np.ndarray:
    N = np.asarray(N, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    dlog10M = np.asarray(dlog10M, dtype=np.float64)
    log_n_base = np.asarray(log_n_base, dtype=np.float64)

    y = np.log((N + float(pseudocount)) / (V[:, None] * dlog10M[None, :])) - log_n_base[None, :]
    y = y - y.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(y, full_matrices=False)
    Phi = vh[: int(K)]
    if add_intercept:
        Phi = np.vstack([np.ones_like(Phi[:1]), Phi])
    return np.asarray(Phi, dtype=np.float64)


def _rbf_kernel(jnp, delta: "jnp.ndarray", amp: "jnp.ndarray", ell: "jnp.ndarray", jitter: "jnp.ndarray"):
    d = delta[:, None] - delta[None, :]
    K = (amp**2) * jnp.exp(-0.5 * (d**2) / (ell**2))
    jitter2 = jitter**2 + jnp.asarray(1e-6, dtype=delta.dtype)
    return K + jitter2 * jnp.eye(delta.shape[0], dtype=delta.dtype)


def _poisson_loglik(jnp, jsp, N: "jnp.ndarray", lam: "jnp.ndarray"):
    return jnp.sum(N * jnp.log(lam + 1e-30) - lam - jsp.special.gammaln(N + 1.0))


def make_loss_fn(
    *,
    N: np.ndarray,
    V: np.ndarray,
    dlog10M: np.ndarray,
    log_n_base: np.ndarray,
    Phi: np.ndarray,
    delta: np.ndarray,
    delta_q: np.ndarray,
    alpha_cons: float,
    beta_tail: float = 0.0,
    tail_top_bins: int = 0,
    tail_eps: float = 1.0,
    jit: bool = True,
):
    N = jnp.asarray(N)
    V = jnp.asarray(V)
    dlog10M = jnp.asarray(dlog10M)
    log_n_base = jnp.asarray(log_n_base)
    Phi = jnp.asarray(Phi)
    delta = jnp.asarray(delta)
    delta_q = jnp.asarray(delta_q)
    use_consistency = bool(float(alpha_cons) > 0.0) and (int(delta_q.shape[0]) > 0)
    w_q = None
    if use_consistency:
        w_q = jnp.full((delta_q.shape[0],), 1.0 / float(delta_q.shape[0]), dtype=delta_q.dtype)

    mu_d = jnp.mean(delta)
    sig_d = jnp.std(delta) + 1e-12
    delta_t = (delta - mu_d) / sig_d
    delta_q_t = (delta_q - mu_d) / sig_d

    S, J = N.shape
    K = Phi.shape[0]
    tail_top_bins = int(tail_top_bins)
    use_tail = bool(float(beta_tail) > 0.0)
    if use_tail and not (1 <= tail_top_bins <= int(J)):
        raise ValueError(f"tail_top_bins={tail_top_bins} invalid for J={int(J)}")
    tail_j0 = int(J) - int(tail_top_bins) if use_tail else int(J)
    tail_eps = float(tail_eps)
    if use_tail and not (tail_eps > 0.0):
        raise ValueError(f"tail_eps must be > 0 (got {tail_eps})")

    def loss(params):
        amp = jnp.exp(params["log_amp"])
        ell = jnp.exp(params["log_ell"])
        jit = jnp.exp(params["log_jitter"])
        Z = params["Z"]

        A = []
        Vvec = []
        logdet_term = 0.0
        for k in range(K):
            Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
            Lk = jnp.linalg.cholesky(Kk)
            zk = Z[k]
            A.append(Lk @ zk)
            Vvec.append(jsp.linalg.solve_triangular(Lk.T, zk, lower=False))
            logdet_term = logdet_term + jnp.sum(jnp.log(jnp.clip(jnp.diag(Lk), 1e-30)))
        A = jnp.stack(A, axis=0)  # (K,S)
        Vvec = jnp.stack(Vvec, axis=0)  # (K,S)

        log_n = log_n_base[None, :] + (A.T @ Phi)
        lam = V[:, None] * dlog10M[None, :] * jnp.exp(log_n)
        ll = _poisson_loglik(jnp, jsp, N, lam)
        # Important: include the Jacobian term for MAP over hyperparameters.
        # With a = L z, log p(a|theta) = -0.5||z||^2 - sum(log diag(L)) + const.
        lp = -0.5 * jnp.sum(Z**2) - logdet_term

        penalty = 0.0
        if use_consistency:
            mus = []
            for k in range(K):
                d = delta_t[:, None] - delta_q_t[None, :]
                k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
                mus.append(k_star.T @ Vvec[k])
            mus = jnp.stack(mus, axis=0)  # (K,Q)

            log_n_q = log_n_base[None, :] + (mus.T @ Phi)
            n_bar = (w_q[:, None] * jnp.exp(log_n_q)).sum(axis=0)
            penalty = float(alpha_cons) * jnp.sum((jnp.log(n_bar + 1e-30) - log_n_base) ** 2)

        tail = 0.0
        if use_tail:
            lam_tail = jnp.sum(lam[:, tail_j0:], axis=1)
            N_tail = jnp.sum(N[:, tail_j0:], axis=1)
            tail = jnp.sum((jnp.log(lam_tail + tail_eps) - jnp.log(N_tail + tail_eps)) ** 2)

        return -(ll + lp) + penalty + float(beta_tail) * tail

    aux = {"delta_mu": mu_d, "delta_sig": sig_d, "delta_train_t": delta_t}
    return (jax.jit(loss) if jit else loss), aux


def train_map(
    loss_fn,
    *,
    K: int,
    S: int,
    steps: int = 5000,
    lr: float = 1e-2,
    lr_schedule: str = "constant",
    lr_min: float = 0.0,
    lr_decay: float = 0.0,
    lr_warmup: int = 0,
    seed: int = 0,
    jit: bool = True,
    init_jitter: float = 1e-2,
    clip_grad_norm: float = 10.0,
    print_every: int = 200,
):
    key = jax.random.key(int(seed))
    params = {
        "log_amp": jnp.full((K,), jnp.log(0.3)),
        "log_ell": jnp.full((K,), jnp.log(1.0)),
        "log_jitter": jnp.full((K,), jnp.log(float(init_jitter))),
        "Z": 0.01 * jax.random.normal(key, (K, S)),
    }

    def zlike(x):
        return jnp.zeros_like(x)

    m = jax.tree_util.tree_map(zlike, params)
    v = jax.tree_util.tree_map(zlike, params)
    t = jnp.array(0, dtype=jnp.int32)

    def global_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum([jnp.sum(jnp.square(x)) for x in leaves]))

    lr0 = float(lr)
    lr_floor = float(lr_min)
    warmup_steps = int(lr_warmup)
    total_steps = int(steps)
    decay = float(lr_decay)

    def lr_at(t):
        # t is 1-based (int32) inside the optimizer step.
        t = jnp.asarray(t, dtype=jnp.float32)

        # Linear warmup from 0 -> lr0 over warmup_steps.
        if warmup_steps > 0:
            w = jnp.clip(t / float(warmup_steps), 0.0, 1.0)
            lr_w = lr0 * w
        else:
            lr_w = lr0

        if lr_schedule == "constant":
            return lr_w

        # After warmup, apply a schedule relative to lr0.
        if total_steps <= warmup_steps:
            return lr_w

        t2 = jnp.maximum(0.0, t - float(warmup_steps))
        denom = float(total_steps - warmup_steps)
        prog = jnp.clip(t2 / denom, 0.0, 1.0)

        if lr_schedule == "cosine":
            lr_s = lr_floor + 0.5 * (lr0 - lr_floor) * (1.0 + jnp.cos(jnp.pi * prog))
        elif lr_schedule == "linear":
            lr_s = lr_floor + (lr0 - lr_floor) * (1.0 - prog)
        elif lr_schedule == "exp":
            lr_s = lr0 * jnp.exp(-decay * t2)
            lr_s = jnp.maximum(lr_floor, lr_s)
        else:
            raise ValueError(f"Unsupported lr_schedule: {lr_schedule!r}")

        # During warmup, cap at lr_w (smooth transition); afterward lr_s is <= lr0 typically.
        return jnp.minimum(lr_w, lr_s)

    def step(params, m, v, t):
        val, grads = jax.value_and_grad(loss_fn)(params)
        if float(clip_grad_norm) > 0:
            gnorm = global_norm(grads)
            scale = jnp.minimum(1.0, (float(clip_grad_norm) / (gnorm + 1e-12)))
            grads = jax.tree_util.tree_map(lambda g: g * scale, grads)
        t = t + 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, m, grads)
        v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * (g * g), v, grads)
        mhat = jax.tree_util.tree_map(lambda a: a / (1 - b1**t), m)
        vhat = jax.tree_util.tree_map(lambda a: a / (1 - b2**t), v)
        lr_t = lr_at(t)
        params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr_t * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
        return params, m, v, t, val, lr_t
    step = jax.jit(step) if jit else step

    val0 = float(loss_fn(params))
    if not np.isfinite(val0):
        raise FloatingPointError(
            "Initial loss is non-finite. Try `--x64 --init-jitter 1e-1 --no-jit --debug-nans` "
            "and/or reduce `--alpha-cons`."
        )

    for i in range(int(steps)):
        params, m, v, t, val, lr_t = step(params, m, v, t)
        if not np.isfinite(float(val)):
            log_amp = np.asarray(params["log_amp"])
            log_ell = np.asarray(params["log_ell"])
            log_jit = np.asarray(params["log_jitter"])
            print(
                f"non-finite at step={i} "
                f"log_amp[min,max]=({np.nanmin(log_amp):.3g},{np.nanmax(log_amp):.3g}) "
                f"log_ell[min,max]=({np.nanmin(log_ell):.3g},{np.nanmax(log_ell):.3g}) "
                f"log_jitter[min,max]=({np.nanmin(log_jit):.3g},{np.nanmax(log_jit):.3g})"
            )
            raise FloatingPointError(
                "Non-finite loss encountered. Re-run with `--debug-nans --no-jit --x64` and/or increase `--init-jitter`."
            )
        if int(print_every) > 0 and (i % int(print_every) == 0):
            loss_per_sample = float(val) / float(S)
            print(i, loss_per_sample, float(lr_t))
    return params


def save_emulator(
    out: Path,
    *,
    params: dict,
    Phi: np.ndarray,
    log10M_edges: np.ndarray,
    log_n_base: np.ndarray,
    delta_mu: float,
    delta_sig: float,
    delta_train: np.ndarray,
    train_center_idx: np.ndarray | None,
    train_gridder_file: Path | None,
    baseline_mode: str | None = None,
    beta_tail: float | None = None,
    tail_top_bins: int | None = None,
    tail_eps: float | None = None,
    kernel_radius: float,
):
    out = Path(out)
    if out.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(out, "w") as f:
            f.attrs["format"] = "tessera.conditional_hmf"
            f.attrs["mass_log_base"] = 10
            f.attrs["mass_pivot_msun"] = float(MASS_PIVOT_MSUN)
            f.attrs["delta_mu"] = float(delta_mu)
            f.attrs["delta_sig"] = float(delta_sig)
            f.attrs["kernel_radius"] = float(kernel_radius)
            if train_gridder_file is not None:
                f.attrs["gridder_file"] = str(Path(train_gridder_file))
            if baseline_mode is not None:
                f.attrs["baseline_mode"] = str(baseline_mode)
            if beta_tail is not None:
                f.attrs["beta_tail"] = float(beta_tail)
            if tail_top_bins is not None:
                f.attrs["tail_top_bins"] = int(tail_top_bins)
            if tail_eps is not None:
                f.attrs["tail_eps"] = float(tail_eps)
            f.create_dataset("Phi", data=np.asarray(Phi, dtype=np.float64))
            f.create_dataset("log10M_edges", data=np.asarray(log10M_edges, dtype=np.float64))
            f.create_dataset("log_n_base", data=np.asarray(log_n_base, dtype=np.float64))
            f.create_dataset("delta_train", data=np.asarray(delta_train, dtype=np.float64))
            if train_center_idx is not None:
                f.create_dataset("train_center_idx", data=np.asarray(train_center_idx, dtype=np.int64))
            f.create_dataset("log_amp", data=np.asarray(params["log_amp"]))
            f.create_dataset("log_ell", data=np.asarray(params["log_ell"]))
            f.create_dataset("log_jitter", data=np.asarray(params["log_jitter"]))
            f.create_dataset("Z", data=np.asarray(params["Z"]))
        return

    np.savez(
        out,
        **{
            "Phi": np.asarray(Phi, dtype=np.float64),
            "log10M_edges": np.asarray(log10M_edges, dtype=np.float64),
            "mass_log_base": np.asarray(10, dtype=np.int64),
            "mass_pivot_msun": np.asarray(MASS_PIVOT_MSUN, dtype=np.float64),
            "log_n_base": np.asarray(log_n_base, dtype=np.float64),
            "delta_mu": float(delta_mu),
            "delta_sig": float(delta_sig),
            "delta_train": np.asarray(delta_train, dtype=np.float64),
            "kernel_radius": float(kernel_radius),
            "log_amp": np.asarray(params["log_amp"]),
            "log_ell": np.asarray(params["log_ell"]),
            "log_jitter": np.asarray(params["log_jitter"]),
            "Z": np.asarray(params["Z"]),
            **(
                {"train_center_idx": np.asarray(train_center_idx, dtype=np.int64)}
                if train_center_idx is not None
                else {}
            ),
            **(
                {"gridder_file": np.asarray(str(Path(train_gridder_file)), dtype=np.str_)}
                if train_gridder_file is not None
                else {}
            ),
            **(
                {"baseline_mode": np.asarray(str(baseline_mode), dtype=np.str_)}
                if baseline_mode is not None
                else {}
            ),
            **({"beta_tail": np.asarray(float(beta_tail), dtype=np.float64)} if beta_tail is not None else {}),
            **({"tail_top_bins": np.asarray(int(tail_top_bins), dtype=np.int64)} if tail_top_bins is not None else {}),
            **({"tail_eps": np.asarray(float(tail_eps), dtype=np.float64)} if tail_eps is not None else {}),
        },
    )


def load_emulator(path: Path) -> dict[str, np.ndarray | float]:
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as f:
            out: dict[str, np.ndarray | float] = {k: np.asarray(f[k]) for k in f.keys()}
            for a in ["delta_mu", "delta_sig", "kernel_radius"]:
                if a in f.attrs:
                    out[a] = float(f.attrs[a])
            if "gridder_file" in f.attrs:
                gf = f.attrs["gridder_file"]
                if isinstance(gf, bytes):
                    gf = gf.decode("utf-8")
                out["gridder_file"] = str(gf)
            if "baseline_mode" in f.attrs:
                bm = f.attrs["baseline_mode"]
                if isinstance(bm, bytes):
                    bm = bm.decode("utf-8")
                out["baseline_mode"] = str(bm)
            if "beta_tail" in f.attrs:
                out["beta_tail"] = float(f.attrs["beta_tail"])
            if "tail_top_bins" in f.attrs:
                out["tail_top_bins"] = int(f.attrs["tail_top_bins"])
            if "tail_eps" in f.attrs:
                out["tail_eps"] = float(f.attrs["tail_eps"])
            if "mass_log_base" in f.attrs:
                out["mass_log_base"] = int(f.attrs["mass_log_base"])
            if "mass_pivot_msun" in f.attrs:
                out["mass_pivot_msun"] = float(f.attrs["mass_pivot_msun"])
            if "log10M_edges" not in out and "lnM_edges" in out:
                out["log10M_edges"] = out.pop("lnM_edges")
            if "mass_pivot_msun" not in out and "log10M_edges" in out:
                edges = np.asarray(out["log10M_edges"], dtype=np.float64)
                out["mass_pivot_msun"] = 1.0 if float(np.nanmean(edges)) > 8.0 else float(MASS_PIVOT_MSUN)
        return out

    d = np.load(path)
    out = {k: d[k] for k in d.files}
    if "log10M_edges" not in out and "lnM_edges" in out:
        out["log10M_edges"] = out.pop("lnM_edges")
    if "mass_log_base" in out:
        out["mass_log_base"] = int(np.asarray(out["mass_log_base"]).item())
    if "mass_pivot_msun" in out:
        out["mass_pivot_msun"] = float(np.asarray(out["mass_pivot_msun"]).item())
    elif "log10M_edges" in out:
        edges = np.asarray(out["log10M_edges"], dtype=np.float64)
        out["mass_pivot_msun"] = 1.0 if float(np.nanmean(edges)) > 8.0 else float(MASS_PIVOT_MSUN)
    if "gridder_file" in out:
        out["gridder_file"] = str(np.asarray(out["gridder_file"]).item())
    if "baseline_mode" in out:
        out["baseline_mode"] = str(np.asarray(out["baseline_mode"]).item())
    for k in ["beta_tail", "tail_eps"]:
        if k in out:
            out[k] = float(np.asarray(out[k]).item())
    if "tail_top_bins" in out:
        out["tail_top_bins"] = int(np.asarray(out["tail_top_bins"]).item())
    return out


def predict_log_n(model: dict[str, np.ndarray | float], delta: float) -> tuple[np.ndarray, np.ndarray]:
    Phi = jnp.asarray(model["Phi"])
    log_n_base = jnp.asarray(model["log_n_base"])
    log10M_edges = np.asarray(model["log10M_edges"])
    log10M_centers = 0.5 * (log10M_edges[:-1] + log10M_edges[1:])

    delta_mu = float(model["delta_mu"])
    delta_sig = float(model["delta_sig"]) + 1e-12
    delta_train = jnp.asarray(model["delta_train"])
    delta_t = (delta_train - delta_mu) / delta_sig
    delta_star_t = (float(delta) - delta_mu) / delta_sig

    amp = jnp.exp(jnp.asarray(model["log_amp"]))
    ell = jnp.exp(jnp.asarray(model["log_ell"]))
    jit = jnp.exp(jnp.asarray(model["log_jitter"]))
    Z = jnp.asarray(model["Z"])

    mu = []
    for k in range(Phi.shape[0]):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        d = delta_t - delta_star_t
        k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
        mu.append(jnp.dot(k_star, v))
    mu = jnp.stack(mu, axis=0)

    log_n = log_n_base + (mu @ Phi)
    return np.asarray(log10M_centers, dtype=np.float64), np.asarray(log_n)

def _print_data_summary(ds: ParentDataset, Phi: np.ndarray | None) -> None:
    def rng(x):
        x = np.asarray(x)
        return float(np.nanmin(x)), float(np.nanmax(x))

    print(f"S={ds.delta.size} J={ds.dlog10M.size}")
    print(f"delta[min,max]={rng(ds.delta)} std={float(np.std(ds.delta)):.6g}")
    print(f"counts[min,max]={rng(ds.N)} total={int(ds.N.sum())}")
    print(f"log_n_base[min,max]={rng(ds.log_n_base)}")
    print(f"log_n_box[min,max]={rng(ds.log_n_box)}")
    print(f"delta_grid[min,max]={rng(ds.delta_grid)}")
    if Phi is not None:
        print(f"Phi shape={Phi.shape} Phi[min,max]={rng(Phi)}")


def _plot_training_overabundance(ds: ParentDataset, *, outdir: Path) -> None:
    """
    Plot the relative abundance (dn/dlog10M in selected spheres) / (dn/dlog10M in full box).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = np.asarray(ds.N.sum(axis=0), dtype=np.float64)
    Vtot = float(np.sum(ds.V))
    n_spheres = counts / (Vtot * np.asarray(ds.dlog10M, dtype=np.float64))
    n_box = np.exp(np.asarray(ds.log_n_box, dtype=np.float64))
    ratio = n_spheres / np.clip(n_box, 1e-300, None)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "conditional_hmf_train_overabundance.png"

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.axhline(1.0, color="k", lw=1, alpha=0.6)
    ax.plot(ds.log10M_centers, ratio, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    ax.set_ylabel(r"$(dn/d\log_{10}M)_\mathrm{spheres} \; / \; (dn/d\log_{10}M)_\mathrm{box}$")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _jax_imports():
    return jax, jnp, jsp


def _initial_params(*, K: int, S: int, seed: int, init_jitter: float):
    jax, jnp, _jsp = _jax_imports()
    key = jax.random.key(int(seed))
    return {
        "log_amp": jnp.full((K,), jnp.log(0.3)),
        "log_ell": jnp.full((K,), jnp.log(1.0)),
        "log_jitter": jnp.full((K,), jnp.log(float(init_jitter))),
        "Z": 0.01 * jax.random.normal(key, (K, S)),
    }


def _loss_parts(
    *,
    N: np.ndarray,
    V: np.ndarray,
    dlog10M: np.ndarray,
    log_n_base: np.ndarray,
    Phi: np.ndarray,
    delta: np.ndarray,
    delta_q: np.ndarray,
    alpha_cons: float,
    beta_tail: float,
    tail_top_bins: int,
    tail_eps: float,
    params,
):
    jax, jnp, jsp = _jax_imports()
    N = jnp.asarray(N)
    V = jnp.asarray(V)
    dlog10M = jnp.asarray(dlog10M)
    log_n_base = jnp.asarray(log_n_base)
    Phi = jnp.asarray(Phi)
    delta = jnp.asarray(delta)
    delta_q = jnp.asarray(delta_q)
    use_consistency = bool(float(alpha_cons) > 0.0) and (int(delta_q.shape[0]) > 0)
    use_tail = bool(float(beta_tail) > 0.0)
    tail_top_bins = int(tail_top_bins)
    if use_tail:
        J = int(N.shape[1])
        if not (1 <= tail_top_bins <= J):
            raise ValueError(f"tail_top_bins={tail_top_bins} invalid for J={J}")
        tail_j0 = J - tail_top_bins
    else:
        tail_j0 = int(N.shape[1])
    tail_eps = float(tail_eps)

    mu_d = jnp.mean(delta)
    sig_d = jnp.std(delta) + 1e-12
    delta_t = (delta - mu_d) / sig_d
    delta_q_t = (delta_q - mu_d) / sig_d
    w_q = jnp.full((delta_q.shape[0],), 1.0 / float(delta_q.shape[0]), dtype=delta_q.dtype)

    amp = jnp.exp(params["log_amp"])
    ell = jnp.exp(params["log_ell"])
    jit = jnp.exp(params["log_jitter"])
    Z = params["Z"]

    A = []
    Vvec = []
    Ldiag_min = []
    logdet_term = 0.0
    for k in range(Phi.shape[0]):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        Ldiag_min.append(jnp.min(jnp.diag(Lk)))
        logdet_term = logdet_term + jnp.sum(jnp.log(jnp.clip(jnp.diag(Lk), 1e-30)))
        zk = Z[k]
        A.append(Lk @ zk)
        Vvec.append(jsp.linalg.solve_triangular(Lk.T, zk, lower=False))
    A = jnp.stack(A, axis=0)
    Vvec = jnp.stack(Vvec, axis=0)
    Ldiag_min = jnp.stack(Ldiag_min, axis=0)

    log_n = log_n_base[None, :] + (A.T @ Phi)
    lam = V[:, None] * dlog10M[None, :] * jnp.exp(log_n)
    ll = _poisson_loglik(jnp, jsp, N, lam)
    lp = -0.5 * jnp.sum(Z**2) - logdet_term

    penalty = 0.0
    if use_consistency:
        w_q = jnp.full((delta_q.shape[0],), 1.0 / float(delta_q.shape[0]), dtype=delta_q.dtype)
        mus = []
        for k in range(Phi.shape[0]):
            d = delta_t[:, None] - delta_q_t[None, :]
            k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
            mus.append(k_star.T @ Vvec[k])
        mus = jnp.stack(mus, axis=0)
        log_n_q = log_n_base[None, :] + (mus.T @ Phi)
        n_bar = (w_q[:, None] * jnp.exp(log_n_q)).sum(axis=0)
        penalty = float(alpha_cons) * jnp.sum((jnp.log(n_bar + 1e-30) - log_n_base) ** 2)

    tail = 0.0
    if use_tail:
        lam_tail = jnp.sum(lam[:, tail_j0:], axis=1)
        N_tail = jnp.sum(N[:, tail_j0:], axis=1)
        tail = jnp.sum((jnp.log(lam_tail + tail_eps) - jnp.log(N_tail + tail_eps)) ** 2)

    loss = -(ll + lp) + penalty + float(beta_tail) * tail
    stats = {
        "loss": loss,
        "ll": ll,
        "lp": lp,
        "penalty": penalty,
        "tail": tail,
        "logdet": logdet_term,
        "lam_min": jnp.min(lam),
        "lam_max": jnp.max(lam),
        "log_n_min": jnp.min(log_n),
        "log_n_max": jnp.max(log_n),
        "Ldiag_min": jnp.min(Ldiag_min),
    }
    return {k: float(stats[k]) for k in stats}


def main():
    ap = argparse.ArgumentParser(description="Conditional HMF emulator (MAP; JAX).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train from parent FOF + gridder overdensity grid.")
    tr.add_argument("--parent-fof", type=Path, default=Path("/cosma7/data/dp004/dc-love2/data/tessera/parent/fof_0011.hdf5"))
    tr.add_argument("--gridder-file", type=Path, default=Path("/cosma7/data/dp004/dc-love2/data/tessera/parent/gridder/gridder_output_512.hdf5"))
    tr.add_argument("--kernel-radius", type=float, default=15.0)
    tr.add_argument("--n-spheres", type=int, default=512)
    tr.add_argument(
        "--n-delta-bins",
        type=int,
        default=10,
        help="Number of equal-width bins in log10(1+delta) used for stratified sphere selection.",
    )
    tr.add_argument(
        "--include-top-overdense",
        type=int,
        default=0,
        help="Force-include this many of the most overdense grid points in the training sphere selection.",
    )
    tr.add_argument(
        "--include-top-halos",
        type=int,
        default=0,
        help="Force-include sphere centres at grid points nearest to the top-N most massive halos (by FOF mass).",
    )
    tr.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Allow spheres to overlap (default is to enforce non-overlapping spheres).",
    )
    tr.add_argument("--n-mass-bins", type=int, default=25)
    tr.add_argument(
        "--last-bin-ratio",
        type=float,
        default=1.2,
        help="Make the final mass bin wider by this factor (while keeping the overall [min,max] range fixed). "
        "Set to 1 for equal-width bins.",
    )
    tr.add_argument("--K", type=int, default=6, help="Number of PCA modes (excluding intercept).")
    tr.add_argument("--steps", type=int, default=5000)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear", "exp"],
        help="Learning-rate schedule (applied to Adam step size).",
    )
    tr.add_argument(
        "--lr-min",
        type=float,
        default=0.0,
        help="Minimum learning rate used by --lr-schedule (0 disables floor).",
    )
    tr.add_argument(
        "--lr-decay",
        type=float,
        default=0.0,
        help="Exponential decay rate per step for --lr-schedule=exp (0 disables).",
    )
    tr.add_argument(
        "--lr-warmup",
        type=int,
        default=0,
        help="Warmup steps: linearly ramp lr from 0 to --lr over this many steps.",
    )
    tr.add_argument("--alpha-cons", type=float, default=0)
    tr.add_argument(
        "--beta-tail",
        type=float,
        default=0.0,
        help="Weight for an additional tail objective: per-sphere squared error between log predicted and log observed "
        "cumulative counts above a mass threshold. 0 disables.",
    )
    tr.add_argument(
        "--tail-top-bins",
        type=int,
        default=1,
        help="Define the tail as the top-K mass bins.",
    )
    tr.add_argument(
        "--tail-eps",
        type=float,
        default=1.0,
        help="Epsilon added inside logs for the tail objective: log(sum lambda + eps) - log(sum N + eps).",
    )
    tr.add_argument(
        "--baseline",
        type=str,
        default="spheres",
        choices=["box", "spheres", "hybrid"],
        help="How to set log_n_base. 'box' uses the full-box HMF; 'spheres' estimates it from the training spheres; "
        "'hybrid' uses the spheres baseline for all bins except the highest-mass bin, which is anchored to the full box.",
    )
    tr.add_argument(
        "--baseline-weight",
        type=str,
        default="uniform",
        choices=["uniform", "pdf"],
        help="When --baseline=spheres, how to weight spheres when estimating the baseline. "
        "'uniform' gives each sphere equal weight; 'pdf' weights each sphere by the global overdensity PDF p(delta).",
    )
    tr.add_argument(
        "--baseline-pseudocount",
        type=float,
        default=0.5,
        help="Pseudocount used when estimating the baseline from spheres (helps avoid -inf when a bin is empty).",
    )
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--out", type=Path, default=Path("conditional_hmf_emulator.h5"))
    tr.add_argument("--x64", action="store_true", help="Enable JAX float64 (often fixes GP Cholesky NaNs).")
    tr.add_argument("--debug-nans", action="store_true", help="Enable JAX NaN/Inf debugging.")
    tr.add_argument("--no-jit", action="store_true", help="Disable JIT (slower, better stack traces).")
    tr.add_argument("--init-jitter", type=float, default=1e-2, help="Initial GP jitter (stddev, not variance).")
    tr.add_argument("--clip-grad-norm", type=float, default=10.0, help="Global grad-norm clip (0 disables).")
    tr.add_argument("--print-every", type=int, default=200)
    tr.add_argument("--check-data", action="store_true", help="Print dataset/Phi stats then exit.")
    tr.add_argument("--diagnose-initial", action="store_true", help="Print initial loss components then exit.")

    pr = sub.add_parser("predict", help="Predict log n(M|delta) from a saved model.")
    pr.add_argument("--model", type=Path, required=True)
    pr.add_argument("--delta", type=float, required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        if args.x64:
            jax.config.update("jax_enable_x64", True)
        if args.debug_nans:
            jax.config.update("jax_debug_nans", True)
            jax.config.update("jax_debug_infs", True)
        ds = build_parent_dataset(
            parent_fof=args.parent_fof,
            gridder_file=args.gridder_file,
            kernel_radius=float(args.kernel_radius),
            n_spheres=int(args.n_spheres),
            n_delta_bins=int(args.n_delta_bins),
            include_top_overdense=int(args.include_top_overdense),
            include_top_halos=int(args.include_top_halos),
            allow_overlap=bool(args.allow_overlap),
            seed=int(args.seed),
            n_mass_bins=int(args.n_mass_bins),
            last_bin_ratio=float(args.last_bin_ratio),
            baseline=str(args.baseline),
            baseline_weight=str(args.baseline_weight),
            baseline_pseudocount=float(args.baseline_pseudocount),
        )
        if str(args.baseline).lower().strip() == "hybrid" and ds.log10M_edges.size >= 2:
            lo, hi = float(ds.log10M_edges[-2]), float(ds.log10M_edges[-1])
            print(f"baseline_hybrid anchored_last_bin log10M∈[{lo:.6g},{hi:.6g}] to parent box")
        # Diagnostics for training-set mass coverage.
        print("log10M_edges " + " ".join(f"{float(x):.8g}" for x in np.asarray(ds.log10M_edges, dtype=np.float64).tolist()))
        print("training_counts_per_bin " + " ".join(str(int(x)) for x in np.asarray(ds.N.sum(axis=0)).tolist()))
        _centres_box, masses_box, _boxsize, _z = read_swift_fof(args.parent_fof)
        log10M_all = np.log10(np.clip(np.asarray(masses_box, dtype=np.float64), 1e-300, None)) - np.log10(MASS_PIVOT_MSUN)
        box_counts, _ = np.histogram(log10M_all, bins=np.asarray(ds.log10M_edges, dtype=np.float64))
        print("parent_box_counts_per_bin " + " ".join(str(int(x)) for x in np.asarray(box_counts).tolist()))
        try:
            outdir = Path(__file__).resolve().parent.parent / "plots"
            _plot_training_overabundance(ds, outdir=outdir)
            print(f"Wrote {outdir / 'conditional_hmf_train_overabundance.png'}")
        except Exception as e:
            print(f"Warning: failed to write training overabundance plot: {e}")
        Phi = pca_mass_basis(ds.N, ds.V, ds.dlog10M, ds.log_n_base, K=int(args.K), add_intercept=True)
        if args.check_data:
            _print_data_summary(ds, Phi)
            return
        J = int(ds.dlog10M.size)
        tail_top_bins = int(args.tail_top_bins)
        if float(args.beta_tail) > 0.0:
            if not (1 <= tail_top_bins <= J):
                raise ValueError(f"--tail-top-bins={tail_top_bins} invalid for J={J}")
        tail_j0 = max(0, J - tail_top_bins)
        if float(args.beta_tail) > 0.0:
            tail_mass = float(ds.log10M_centers[min(tail_j0, J - 1)]) if J > 0 else float("nan")
            print(
                f"tail_objective beta={float(args.beta_tail):g} tail_top_bins={tail_top_bins} "
                f"tail_log10M_min={tail_mass:.6g} tail_eps={float(args.tail_eps):g}"
            )
        if args.diagnose_initial:
            params0 = _initial_params(K=Phi.shape[0], S=ds.delta.size, seed=int(args.seed), init_jitter=float(args.init_jitter))
            parts = _loss_parts(
                N=ds.N,
                V=ds.V,
                dlog10M=ds.dlog10M,
                log_n_base=ds.log_n_base,
                Phi=Phi,
                delta=ds.delta,
                delta_q=ds.delta_grid,
                alpha_cons=float(args.alpha_cons),
                beta_tail=float(args.beta_tail),
                tail_top_bins=tail_top_bins,
                tail_eps=float(args.tail_eps),
                params=params0,
            )
            for k in ["loss", "ll", "lp", "penalty", "tail", "logdet", "lam_min", "lam_max", "log_n_min", "log_n_max", "Ldiag_min"]:
                print(f"{k} {parts[k]:.8e}")
            return

        loss_fn, aux = make_loss_fn(
            N=ds.N,
            V=ds.V,
            dlog10M=ds.dlog10M,
            log_n_base=ds.log_n_base,
            Phi=Phi,
            delta=ds.delta,
            delta_q=ds.delta_grid,
            alpha_cons=float(args.alpha_cons),
            beta_tail=float(args.beta_tail),
            tail_top_bins=tail_top_bins,
            tail_eps=float(args.tail_eps),
            jit=not bool(args.no_jit),
        )
        print(
            "lr_schedule "
            f"{str(args.lr_schedule)} lr={float(args.lr):g} lr_min={float(args.lr_min):g} "
            f"lr_decay={float(args.lr_decay):g} lr_warmup={int(args.lr_warmup)}"
        )
        params = train_map(
            loss_fn,
            K=Phi.shape[0],
            S=ds.delta.size,
            steps=int(args.steps),
            lr=float(args.lr),
            lr_schedule=str(args.lr_schedule),
            lr_min=float(args.lr_min),
            lr_decay=float(args.lr_decay),
            lr_warmup=int(args.lr_warmup),
            seed=int(args.seed),
            jit=not bool(args.no_jit),
            init_jitter=float(args.init_jitter),
            clip_grad_norm=float(args.clip_grad_norm),
            print_every=int(args.print_every),
        )
        save_emulator(
            args.out,
            params=params,
            Phi=Phi,
            log10M_edges=ds.log10M_edges,
            log_n_base=ds.log_n_base,
            delta_mu=float(aux["delta_mu"]),
            delta_sig=float(aux["delta_sig"]),
            delta_train=np.asarray(ds.delta, dtype=np.float64),
            train_center_idx=np.asarray(ds.center_idx, dtype=np.int64),
            train_gridder_file=args.gridder_file,
            baseline_mode=str(args.baseline),
            beta_tail=(float(args.beta_tail) if float(args.beta_tail) > 0.0 else None),
            tail_top_bins=(tail_top_bins if float(args.beta_tail) > 0.0 else None),
            tail_eps=(float(args.tail_eps) if float(args.beta_tail) > 0.0 else None),
            kernel_radius=float(args.kernel_radius),
        )
        print(f"Wrote {args.out}")
    elif args.cmd == "predict":
        model = load_emulator(args.model)
        log10M, log_n = predict_log_n(model, float(args.delta))
        for x, y in zip(log10M, log_n):
            print(f"{x:.8e} {y:.8e}")


if __name__ == "__main__":
    main()
