#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def _add_repo_root_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_root_to_path()

from tessera.emulator.conditional_hmf import (
    _choose_centers_stratified,
    _choose_centers_stratified_2d,
    _choose_centers_stratified_2d_excluding,
    _rbf_kernel,
    infer_two_smallest_kernel_radii,
    load_emulator,
    load_gridder_overdensity,
    load_gridder_overdensity_pair,
    read_swift_fof,
)

def _choose_centers_stratified_excluding(delta: np.ndarray, n: int, seed: int, exclude_idx: np.ndarray) -> np.ndarray:
    """
    Choose `n` center indices stratified in delta, excluding any indices in `exclude_idx`.

    Returns indices into the *original* `delta` array.
    """
    delta = np.asarray(delta, dtype=np.float64)
    n = int(n)
    exclude_idx = np.asarray(exclude_idx, dtype=np.int64).ravel()

    if n <= 0:
        raise ValueError(f"n={n} must be > 0")
    if exclude_idx.size == 0:
        return _choose_centers_stratified(delta, n, seed)

    exclude_idx = np.unique(exclude_idx)
    if np.any(exclude_idx < 0) or np.any(exclude_idx >= delta.size):
        raise ValueError("exclude_idx contains out-of-range indices")

    mask = np.ones(delta.size, dtype=bool)
    mask[exclude_idx] = False
    avail_idx = np.nonzero(mask)[0].astype(np.int64)
    if n > avail_idx.size:
        raise ValueError(f"n={n} exceeds available grid points after exclusion ({avail_idx.size})")

    sub = _choose_centers_stratified(delta[avail_idx], n, seed)
    return avail_idx[np.asarray(sub, dtype=np.int64)]


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    return arr

def _infer_unique_kernel_radius_from_gridder(path: Path) -> float | None:
    """
    Best-effort inference of KernelRadius when a model doesn't store it.

    Returns a radius if the gridder file contains exactly one unique finite KernelRadius
    across its Kernel_* groups; otherwise returns None.
    """
    try:
        import h5py
    except Exception:
        return None

    path = Path(path)
    radii: list[float] = []
    with h5py.File(path, "r") as f:
        if "Grids" not in f:
            return None
        for name in f["Grids"]:
            if not str(name).startswith("Kernel_"):
                continue
            grp = f["Grids"][name]
            r = grp.attrs.get("KernelRadius", None)
            if r is None:
                continue
            rr = float(np.asarray(r).ravel()[0])
            if np.isfinite(rr):
                radii.append(rr)

    if not radii:
        return None
    uniq = sorted(set(float(r) for r in radii))
    return uniq[0] if len(uniq) == 1 else None


def predict_log_n_batch(model: dict, delta_eval: np.ndarray) -> np.ndarray:
    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    Phi = jnp.asarray(model["Phi"], dtype=dtype)
    log_n_base = jnp.asarray(model["log_n_base"], dtype=dtype)
    delta_mu = np.asarray(model["delta_mu"], dtype=np.float64).reshape(-1)
    delta_sig = np.asarray(model["delta_sig"], dtype=np.float64).reshape(-1) + 1e-12
    delta_train = _as_2d(np.asarray(model["delta_train"], dtype=np.float64))
    delta_eval = _as_2d(np.asarray(delta_eval, dtype=np.float64))
    if delta_train.shape[1] != delta_eval.shape[1] or int(delta_mu.size) != int(delta_train.shape[1]):
        raise ValueError("delta dimension mismatch between model and delta_eval")

    amp = jnp.exp(jnp.asarray(model["log_amp"], dtype=dtype))
    ell = jnp.exp(jnp.asarray(model["log_ell"], dtype=dtype))
    jit = jnp.exp(jnp.asarray(model["log_jitter"], dtype=dtype))
    Z = jnp.asarray(model["Z"], dtype=dtype)

    delta_t = (jnp.asarray(delta_train, dtype=dtype) - jnp.asarray(delta_mu, dtype=dtype)[None, :]) / jnp.asarray(
        delta_sig, dtype=dtype
    )[None, :]
    delta_eval_t = (jnp.asarray(delta_eval, dtype=dtype) - jnp.asarray(delta_mu, dtype=dtype)[None, :]) / jnp.asarray(
        delta_sig, dtype=dtype
    )[None, :]

    mus = []
    for k in range(Phi.shape[0]):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        ellk = ell[k]
        if ellk.ndim == 0:
            ellk = jnp.full((delta_t.shape[1],), ellk, dtype=delta_t.dtype)
        diff = (delta_t[:, None, :] - delta_eval_t[None, :, :]) / ellk[None, None, :]
        sq = jnp.sum(diff * diff, axis=-1)
        k_star = (amp[k] ** 2) * jnp.exp(-0.5 * sq)
        mus.append(k_star.T @ v)
    mus = jnp.stack(mus, axis=0)  # (K,S_eval)

    log_n = log_n_base[None, :] + (mus.T @ Phi)
    return np.asarray(log_n)


def predict_log_n_batch_with_var(model: dict, delta_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (mean_log_n, var_log_n) across a batch of overdensities.

    The variance is the GP conditional variance of log n induced by independent
    per-mode GPs for the PCA coefficients, propagated through the basis as:
        var_log_n_j = sum_k var(a_k) * Phi_{k,j}^2
    """
    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    Phi = jnp.asarray(model["Phi"], dtype=dtype)
    Phi2 = Phi * Phi
    log_n_base = jnp.asarray(model["log_n_base"], dtype=dtype)
    delta_mu = np.asarray(model["delta_mu"], dtype=np.float64).reshape(-1)
    delta_sig = np.asarray(model["delta_sig"], dtype=np.float64).reshape(-1) + 1e-12
    delta_train = _as_2d(np.asarray(model["delta_train"], dtype=np.float64))
    delta_eval = _as_2d(np.asarray(delta_eval, dtype=np.float64))
    if delta_train.shape[1] != delta_eval.shape[1] or int(delta_mu.size) != int(delta_train.shape[1]):
        raise ValueError("delta dimension mismatch between model and delta_eval")

    amp = jnp.exp(jnp.asarray(model["log_amp"], dtype=dtype))
    ell = jnp.exp(jnp.asarray(model["log_ell"], dtype=dtype))
    jit = jnp.exp(jnp.asarray(model["log_jitter"], dtype=dtype))
    Z = jnp.asarray(model["Z"], dtype=dtype)

    delta_t = (jnp.asarray(delta_train, dtype=dtype) - jnp.asarray(delta_mu, dtype=dtype)[None, :]) / jnp.asarray(
        delta_sig, dtype=dtype
    )[None, :]
    delta_eval_t = (jnp.asarray(delta_eval, dtype=dtype) - jnp.asarray(delta_mu, dtype=dtype)[None, :]) / jnp.asarray(
        delta_sig, dtype=dtype
    )[None, :]

    mus = []
    vars_a = []
    for k in range(Phi.shape[0]):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        ellk = ell[k]
        if ellk.ndim == 0:
            ellk = jnp.full((delta_t.shape[1],), ellk, dtype=delta_t.dtype)
        diff = (delta_t[:, None, :] - delta_eval_t[None, :, :]) / ellk[None, None, :]
        sq = jnp.sum(diff * diff, axis=-1)
        k_star = (amp[k] ** 2) * jnp.exp(-0.5 * sq)
        mus.append(k_star.T @ v)

        # Latent-function predictive variance: amp^2 - k_*^T K^{-1} k_*
        w = jsp.linalg.solve_triangular(Lk, k_star, lower=True)
        var_k = (amp[k] ** 2) - jnp.sum(w * w, axis=0)
        vars_a.append(jnp.clip(var_k, a_min=0.0))

    mus = jnp.stack(mus, axis=0)  # (K,S_eval)
    vars_a = jnp.stack(vars_a, axis=0)  # (K,S_eval)

    log_n = log_n_base[None, :] + (mus.T @ Phi)
    var_log_n = vars_a.T @ Phi2
    return np.asarray(log_n), np.asarray(var_log_n)


def poisson_loglik_counts(N: np.ndarray, lam: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    N = np.asarray(N, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    return np.sum(N * np.log(lam + 1e-30) - lam - gammaln(N + 1.0), axis=1)


def poisson_garwood_1sigma(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import chi2

    N = np.asarray(N, dtype=np.float64)
    alpha = 1.0 - 0.6826894921370859  # 1σ central interval
    lo = np.where(N > 0, 0.5 * chi2.ppf(alpha / 2.0, 2.0 * N), 0.0)
    hi = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (N + 1.0))
    return lo, hi


def poisson_mode(lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=np.float64)
    return np.floor(np.maximum(lam, 0.0)).astype(np.int64)


def poisson_median(lam: np.ndarray) -> np.ndarray:
    from scipy.stats import poisson

    lam = np.asarray(lam, dtype=np.float64)
    # Median is the smallest k with CDF(k) >= 0.5; scipy's PPF does that directly.
    med = poisson.ppf(0.5, lam)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(np.int64)


def gp_health(model: dict) -> dict[str, np.ndarray]:
    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32

    delta_mu = np.asarray(model["delta_mu"], dtype=np.float64).reshape(-1)
    delta_sig = np.asarray(model["delta_sig"], dtype=np.float64).reshape(-1) + 1e-12
    delta_train = _as_2d(np.asarray(model["delta_train"], dtype=np.float64))
    delta_t = (jnp.asarray(delta_train, dtype=dtype) - jnp.asarray(delta_mu, dtype=dtype)[None, :]) / jnp.asarray(
        delta_sig, dtype=dtype
    )[None, :]
    amp = jnp.exp(jnp.asarray(model["log_amp"], dtype=dtype))
    ell = jnp.exp(jnp.asarray(model["log_ell"], dtype=dtype))
    jit = jnp.exp(jnp.asarray(model["log_jitter"], dtype=dtype))

    mins = []
    for k in range(int(amp.shape[0])):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        mins.append(jnp.min(jnp.diag(Lk)))
    return {
        "amp": np.asarray(amp),
        "ell": np.asarray(ell),
        "jit": np.asarray(jit),
        "chol_diag_min": np.asarray(jnp.stack(mins, axis=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate conditional HMF emulator on held-out parent spheres.")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--parent-fof", type=Path, default=Path("/cosma7/data/dp004/dc-love2/data/tessera/parent/fof_0011.hdf5"))
    ap.add_argument(
        "--gridder-file",
        type=Path,
        default=None,
        help="Gridder file providing overdensity values. Defaults to the `gridder_file` stored in the model (if present).",
    )
    ap.add_argument(
        "--kernel-radius",
        type=float,
        default=None,
        help="Kernel radius in Mpc (required if not stored in the model and gridder has multiple kernels).",
    )
    ap.add_argument("--kernel-radius-1", type=float, default=None, help="First overdensity kernel radius (Mpc).")
    ap.add_argument("--kernel-radius-2", type=float, default=None, help="Second overdensity kernel radius (Mpc).")
    ap.add_argument("--n-spheres", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1001, help="Seed for held-out sphere selection.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory for plots (default: ../plots relative to this script).")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--mc-delta", type=int, default=2048, help="MC samples for consistency check (0 disables).")
    ap.add_argument("--x64", action="store_true", help="Enable JAX float64 (often fixes prediction Cholesky NaNs).")
    ap.add_argument("--cpu", action="store_true", help="Force JAX to use CPU (set JAX_PLATFORM_NAME=cpu).")
    ap.add_argument("--debug-nans", action="store_true", help="Enable JAX NaN/Inf debugging.")
    ap.add_argument(
        "--pick-counts-by",
        type=str,
        default="ll_pred",
        choices=["ll_pred", "dll"],
        help="Metric used to select example spheres for the individual plots. "
        "'ll_pred' uses per-sphere Poisson log-likelihood; 'dll' uses per-sphere ΔlogL (pred - baseline).",
    )
    ap.add_argument(
        "--pick-counts-percentiles",
        type=float,
        nargs="+",
        default=[50.0, 97.0, 99.0],
        help="Percentiles (0-100) used to choose example spheres for the individual plots (two spheres per percentile).",
    )
    args = ap.parse_args()

    if args.cpu:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if args.x64:
        jax.config.update("jax_enable_x64", True)
    if args.debug_nans:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)

    model = load_emulator(args.model)
    gridder_file = args.gridder_file
    if gridder_file is None and "gridder_file" in model:
        gridder_file = Path(str(model["gridder_file"]))
    if gridder_file is None:
        raise ValueError("Gridder file not provided. Pass --gridder-file or retrain a model that stores it.")

    sphere_radius = args.kernel_radius
    if sphere_radius is None and "kernel_radius" in model:
        sphere_radius = float(model["kernel_radius"])
    if sphere_radius is None:
        sphere_radius = _infer_unique_kernel_radius_from_gridder(gridder_file)
    if sphere_radius is None:
        raise ValueError(
            "Sphere radius not available. Provide `--kernel-radius`, or retrain/export a model that stores it "
            "(or ensure your gridder file contains a single KernelRadius)."
        )

    # Determine the two overdensity smoothing radii.
    kr1 = args.kernel_radius_1
    kr2 = args.kernel_radius_2
    if (kr1 is None or kr2 is None) and "kernel_radii" in model:
        rr = np.asarray(model["kernel_radii"], dtype=np.float64).reshape(-1)
        if rr.size >= 2:
            kr1, kr2 = float(rr[0]), float(rr[1])
    if kr1 is None or kr2 is None:
        if kr2 is None:
            kr2 = float(sphere_radius)
        # Infer the smaller radius from the gridder file.
        r1_inf, r2_inf = infer_two_smallest_kernel_radii(gridder_file)
        kr1 = float(r1_inf) if not np.isclose(float(r1_inf), float(kr2), rtol=0, atol=1e-8) else float(r2_inf)
    kr1, kr2 = float(kr1), float(kr2)
    if kr2 < kr1:
        kr1, kr2 = kr2, kr1
    print(f"kernel_radii {kr1:g} {kr2:g}  sphere_radius {float(sphere_radius):g}")
    log10M_edges = np.asarray(model["log10M_edges"], dtype=np.float64)
    dlog10M = np.diff(log10M_edges)
    log10M_centers = 0.5 * (log10M_edges[:-1] + log10M_edges[1:])
    mass_pivot_msun = float(model.get("mass_pivot_msun", 1e10))

    halo_pos, halo_mass, boxsize, _z = read_swift_fof(args.parent_fof)
    grid_pos, grid_delta1, grid_delta2 = load_gridder_overdensity_pair(
        gridder_file, kernel_radius_1=float(kr1), kernel_radius_2=float(kr2)
    )

    train_idx = None
    if "train_center_idx" in model:
        train_idx = np.asarray(model["train_center_idx"], dtype=np.int64).ravel()
        if train_idx.size == 0:
            train_idx = None
    if train_idx is None:
        idx = _choose_centers_stratified_2d(grid_delta1, grid_delta2, int(args.n_spheres), int(args.seed))
    else:
        idx = _choose_centers_stratified_2d_excluding(
            grid_delta1, grid_delta2, int(args.n_spheres), int(args.seed), train_idx
        )
        print(f"excluded_train_centers {int(train_idx.size)}")
    centers = grid_pos[idx]
    delta_vec = np.stack([np.asarray(grid_delta1, dtype=np.float64)[idx], np.asarray(grid_delta2, dtype=np.float64)[idx]], axis=1)
    # For 1D plots/colouring, keep using the larger-scale overdensity (second component).
    delta = np.asarray(delta_vec[:, 1], dtype=np.float64)

    from scipy.spatial import cKDTree

    L = float(boxsize)
    tree = cKDTree(np.mod(halo_pos, L), boxsize=L)
    hits = tree.query_ball_point(np.mod(centers, L), r=float(sphere_radius), workers=-1)

    log10M = np.log10(np.clip(halo_mass, 1e-300, None)) - np.log10(mass_pivot_msun)
    N = np.zeros((int(args.n_spheres), dlog10M.size), dtype=np.int32)
    for s, ids in enumerate(hits):
        if ids:
            N[s], _ = np.histogram(log10M[np.asarray(ids, dtype=np.int64)], bins=log10M_edges)

    V = float(4.0 / 3.0 * np.pi * float(sphere_radius) ** 3)
    log_n_pred = predict_log_n_batch(model, delta_vec)
    if not np.all(np.isfinite(log_n_pred)):
        h = gp_health(model)
        print("Non-finite predictions detected.")
        print(f"  amp[min,max]=({np.nanmin(h['amp']):.3g},{np.nanmax(h['amp']):.3g})")
        print(f"  ell[min,max]=({np.nanmin(h['ell']):.3g},{np.nanmax(h['ell']):.3g})")
        print(f"  jitter[min,max]=({np.nanmin(h['jit']):.3g},{np.nanmax(h['jit']):.3g})")
        print(f"  chol_diag_min[min,max]=({np.nanmin(h['chol_diag_min']):.3g},{np.nanmax(h['chol_diag_min']):.3g})")
        print("Try re-running with `--x64` and/or `--cpu`.")
        args.no_plots = True
    lam_pred = V * dlog10M[None, :] * np.exp(log_n_pred)

    log_n_base = np.asarray(model["log_n_base"], dtype=np.float64)
    lam_base = V * dlog10M[None, :] * np.exp(log_n_base[None, :])

    ll_pred = poisson_loglik_counts(N, lam_pred)
    ll_base = poisson_loglik_counts(N, lam_base)

    print(f"n_spheres {N.shape[0]}  sphere_R {float(sphere_radius):g} Mpc")
    print(
        f"heldout delta1[min,max] ({float(np.min(delta_vec[:,0])):.3g},{float(np.max(delta_vec[:,0])):.3g})  "
        f"delta2[min,max] ({float(np.min(delta_vec[:,1])):.3g},{float(np.max(delta_vec[:,1])):.3g})"
    )
    print(f"sum N {int(N.sum())}  mean N/sphere {float(N.sum()/N.shape[0]):.3g}")
    print(f"total loglik (pred) {float(ll_pred.sum()):.6g}")
    print(f"total loglik (base) {float(ll_base.sum()):.6g}")
    print(f"delta loglik (pred-base) {float((ll_pred-ll_base).sum()):.6g}")
    print(f"frac spheres improved {float(np.mean(ll_pred > ll_base)):.3f}")

    if int(args.mc_delta) > 0:
        rng = np.random.default_rng(int(args.seed) + 77)
        if grid_delta2.size <= int(args.mc_delta):
            d_mc = np.stack([np.asarray(grid_delta1, dtype=np.float64), np.asarray(grid_delta2, dtype=np.float64)], axis=1)
        else:
            pick = rng.choice(np.arange(int(grid_delta2.size), dtype=np.int64), size=int(args.mc_delta), replace=False)
            d_mc = np.stack([np.asarray(grid_delta1, dtype=np.float64)[pick], np.asarray(grid_delta2, dtype=np.float64)[pick]], axis=1)
        log_n_mc = predict_log_n_batch(model, d_mc)
        n_bar = np.mean(np.exp(log_n_mc), axis=0)
        rel = (n_bar - np.exp(log_n_base)) / np.exp(log_n_base)
        print(f"consistency max|rel| {float(np.max(np.abs(rel))):.3g}")

    if args.no_plots or not np.all(np.isfinite(ll_pred)) or not np.all(np.isfinite(ll_base)):
        return

    outdir = args.outdir
    if outdir is None:
        outdir = Path(__file__).resolve().parent.parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    obs_tot = N.sum(axis=0)
    pred_tot = lam_pred.sum(axis=0)
    base_tot = lam_base.sum(axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.plot(log10M_centers, obs_tot, label="Observed", lw=2)
    ax.plot(log10M_centers, pred_tot, label="Predicted E[N]", lw=2)
    ax.plot(log10M_centers, base_tot, label="Baseline E[N]", lw=1.5, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    ax.set_ylabel("Total counts across spheres")
    ax.legend(frameon=False)
    fig.savefig(outdir / "conditional_hmf_eval_counts.png", dpi=150)
    plt.close(fig)

    dll = ll_pred - ll_base
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    ax.hist(dll, bins=40, histtype="stepfilled", alpha=0.8)
    ax.axvline(0.0, color="k", lw=1)
    ax.set_xlabel(r"$\Delta \log \mathcal{L}$ (pred - baseline) per sphere")
    ax.set_ylabel("Count")
    fig.savefig(outdir / "conditional_hmf_eval_dloglik_hist.png", dpi=150)
    plt.close(fig)

    totN = N.sum(axis=1)
    totLam = lam_pred.sum(axis=1)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    import matplotlib as mpl

    dcol = np.asarray(delta, dtype=np.float64)
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(dcol)), vmax=float(np.nanmax(dcol)))
    sc = ax.scatter(totLam, totN, s=10, alpha=0.7, c=dcol, cmap="viridis", norm=norm)
    lo = 0.0
    hi = max(float(np.max(totLam)), float(np.max(totN)), 1.0)
    ax.plot([lo, hi], [lo, hi], color="k", lw=1)
    ax.set_xlabel("Predicted total E[N] per sphere")
    ax.set_ylabel("Observed total N per sphere")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\delta_R$")
    fig.savefig(outdir / "conditional_hmf_eval_total_counts_scatter.png", dpi=150)
    plt.close(fig)

    # Per-mass-bin predicted-vs-observed count scatter (one subplot per mass bin).
    # Use a compact 3x2 layout when we have ~6 bins (common in tail-focused tests).
    ncols_bin = 2
    nrows_bin = 3 if int(dlog10M.size) <= 6 else int(np.ceil(dlog10M.size / ncols_bin))
    fig, axes = plt.subplots(
        nrows_bin,
        ncols_bin,
        figsize=(6.8, 2.9 * nrows_bin),
        constrained_layout=True,
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])

    for j in range(int(dlog10M.size)):
        ax = axes.flat[j]
        x = np.asarray(lam_pred[:, j], dtype=np.float64)
        y = np.asarray(N[:, j], dtype=np.float64)
        ax.scatter(x, y, s=6, alpha=0.45, c=dcol, cmap="viridis", norm=norm)
        hi = max(float(np.nanmax(x)), float(np.nanmax(y)), 1.0)
        ax.plot([0.0, hi], [0.0, hi], color="k", lw=1, alpha=0.6)
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"log10M={log10M_centers[j]:.2f}", fontsize=9)
        if j % ncols_bin == 0:
            ax.set_ylabel("Observed N")
        if j // ncols_bin == (nrows_bin - 1):
            ax.set_xlabel("Predicted E[N]")

    for ax in axes.flat[int(dlog10M.size) :]:
        ax.axis("off")

    cbar = fig.colorbar(sm, ax=list(axes.flat), pad=0.01, shrink=0.9)
    cbar.set_label(r"$\delta_R$")
    fig.savefig(outdir / "conditional_hmf_eval_counts_by_bin_scatter.png", dpi=150)
    plt.close(fig)

    # Individual HMFs at selected predictive-uncertainty percentiles.
    n_pred = np.exp(log_n_pred)
    n_base = np.exp(log_n_base[None, :])
    n_obs = N / (V * dlog10M[None, :])
    clo, chi = poisson_garwood_1sigma(N)
    n_lo = clo / (V * dlog10M[None, :])
    n_hi = chi / (V * dlog10M[None, :])

    if args.pick_counts_by == "ll_pred":
        # For selection, treat "badness" as -log L so higher percentiles are worse fits.
        metric = -ll_pred
        metric_label = "-ll"
    elif args.pick_counts_by == "dll":
        metric = ll_pred - ll_base
        metric_label = "dll"
    else:
        raise ValueError(f"Unsupported --pick-counts-by={args.pick_counts_by!r}")

    def pick_nearest(err_vals: np.ndarray, pct: float, n_pick: int, used: set[int]) -> list[int]:
        target = float(np.percentile(err_vals, pct))
        order = np.argsort(np.abs(err_vals - target))
        out = []
        for i in order:
            ii = int(i)
            if ii in used:
                continue
            out.append(ii)
            used.add(ii)
            if len(out) == int(n_pick):
                break
        return out

    def pct_label(p: float) -> str:
        # Use p01, p25, p50, p97, p99 style labels.
        pi = int(round(float(p)))
        if abs(float(p) - pi) < 1e-9 and 0 <= pi < 10:
            return f"p{pi:02d}"
        if abs(float(p) - pi) < 1e-9:
            return f"p{pi:d}"
        return f"p{float(p):g}"

    used: set[int] = set()
    picks: list[tuple[str, int]] = []

    # Build a fixed, ordered list of percentiles (best → worst).
    if args.pick_counts_by == "ll_pred":
        # In -ll space, smaller is better; include p01 and p25 as explicit "best-fit" examples.
        pcts = sorted({1.0, 25.0, *[float(x) for x in args.pick_counts_percentiles]})
    else:
        # For dll (pred - baseline), larger is better, so sort percentiles descending for best → worst.
        pcts = sorted({*[float(x) for x in args.pick_counts_percentiles]}, reverse=True)

    for pct in pcts:
        label = pct_label(pct)
        for idx in pick_nearest(metric, float(pct), 2, used):
            picks.append((label, idx))

    ncols = 2
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.6 * nrows), constrained_layout=True, sharex=True, sharey=True, squeeze=False)
    for ax, (label, s) in zip(axes.flat, picks):
        mask_pos = N[s] > 0
        mask_zero = ~mask_pos
        is_first = ax is axes.flat[0]
        ax.errorbar(
            log10M_centers[mask_pos],
            n_obs[s, mask_pos],
            yerr=[n_obs[s, mask_pos] - n_lo[s, mask_pos], n_hi[s, mask_pos] - n_obs[s, mask_pos]],
            fmt="o",
            ms=3,
            lw=0.8,
            capsize=2,
            label="Observed" if is_first else None,
        )
        if np.any(mask_zero):
            # For N=0, plot a 1σ upper limit using Garwood interval (no central value on log-scale).
            y = n_hi[s, mask_zero]
            ax.errorbar(
                log10M_centers[mask_zero],
                y,
                yerr=y * 0.6,
                fmt="v",
                ms=3,
                lw=0.8,
                capsize=0,
                uplims=True,
                alpha=0.9,
                label="Observed (1σ upper limits)" if is_first else None,
            )

        # Emulator mean HMF (density).
        ax.plot(log10M_centers, n_pred[s], lw=2, label="Emulator (mean)" if is_first else None)
        # ax.plot(log10M_centers, n_base[0], lw=1.5, ls="--", label="Baseline (mean)" if is_first else None)

        # Emulator mode/median HMF (convert implied counts to density).
        # mode_counts = poisson_mode(lam_pred[s])
        median_counts = poisson_median(lam_pred[s])
        # n_mode = mode_counts / (V * dlog10M)
        n_median = median_counts / (V * dlog10M)

        # n_mode_plot = np.where(mode_counts > 0, n_mode, np.nan)
        n_median_plot = np.where(median_counts > 0, n_median, np.nan)

        # Choose a sensible y-location for markers indicating exact zeros on a log axis.
        positives: list[np.ndarray] = []
        for arr in [n_pred[s], n_base[0], n_hi[s]]:
            sel = np.asarray(arr)[np.asarray(arr) > 0]
            if sel.size:
                positives.append(sel)
        y_floor = (float(np.min(np.concatenate(positives))) / 5.0) if positives else 1e-12

        # ax.step(log10M_centers, n_mode_plot, where="mid", lw=1.6, ls=":", label="Emulator (mode)" if is_first else None)
        ax.step(
            log10M_centers,
            n_median_plot,
            where="mid",
            lw=1.6,
            ls="-.",
            label="Emulator (median)" if is_first else None,
        )
        # if np.any(mode_counts == 0):
        #     ax.scatter(
        #         log10M_centers[mode_counts == 0],
        #         np.full(int(np.sum(mode_counts == 0)), y_floor),
        #         marker="v",
        #         s=18,
        #         linewidths=0,
        #         alpha=0.7,
        #         color=ax.lines[-2].get_color(),
        #     )
        if np.any(median_counts == 0):
            ax.scatter(
                log10M_centers[median_counts == 0],
                np.full(int(np.sum(median_counts == 0)), y_floor),
                marker="x",
                s=22,
                linewidths=1.0,
                alpha=0.7,
                color=ax.lines[-1].get_color(),
            )
        ax.set_yscale("log")
        ax.set_title(f"{label}: {metric_label}={float(metric[s]):.3g}, δ={delta[s]:.3g}, N={int(N[s].sum())}")

    for ax in axes.flat[len(picks) :]:
        ax.axis("off")

    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$dn/d\log_{10}M\;[\mathrm{Mpc}^{-3}]$")
        ax.set_ylim(1e-5, )
    fig.savefig(outdir / "conditional_hmf_eval_individual_hmfs.png", dpi=150)
    plt.close(fig)

    # Individual COUNT curves (mean/mode/median), which can be exactly zero.
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.6 * nrows), constrained_layout=True, sharex=True, sharey=True, squeeze=False)
    for ax, (label, s) in zip(axes.flat, picks):
        is_first = ax is axes.flat[0]
        # Observed counts with Garwood 1σ interval.
        ax.errorbar(
            log10M_centers,
            N[s],
            yerr=[N[s] - clo[s], chi[s] - N[s]],
            fmt="o",
            ms=3,
            lw=0.8,
            capsize=2,
            label="Observed" if is_first else None,
        )

        # Predicted means (same as previously shown, but in count space).
        ax.step(log10M_centers, lam_pred[s], where="mid", lw=2, label="Emulator (mean)" if is_first else None)
        # ax.step(log10M_centers, lam_base[0], where="mid", lw=1.5, ls="--", label="Baseline (mean)" if is_first else None)

        # # Mode/median implied by the Poisson model.
        # ax.step(
        #     log10M_centers,
        #     poisson_mode(lam_pred[s]),
        #     where="mid",
        #     lw=1.8,
        #     ls=":",
        #     label="Emulator (mode)" if is_first else None,
        # )
        ax.step(
            log10M_centers,
            poisson_median(lam_pred[s]),
            where="mid",
            lw=1.8,
            ls="-.",
            label="Emulator (median)" if is_first else None,
        )

        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"{label}: {metric_label}={float(metric[s]):.3g}, δ={delta[s]:.3g}, N={int(N[s].sum())}")

    for ax in axes.flat[len(picks) :]:
        ax.axis("off")

    axes[0, 0].legend(frameon=False, fontsize=9, ncol=2)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Counts per sphere per bin")
    fig.savefig(outdir / "conditional_hmf_eval_individual_counts.png", dpi=150)
    plt.close(fig)

    # Fit decomposition by mass bin: ΔlogL (pred - baseline) summed across spheres.
    # This aligns with the single-scalar objective (Poisson log-likelihood), but shows which mass bins
    # contribute positive/negative evidence for the emulator relative to the baseline.
    from scipy.special import gammaln

    ll_pred_bin = N * np.log(lam_pred + 1e-30) - lam_pred - gammaln(N + 1.0)
    ll_base_bin = N * np.log(lam_base + 1e-30) - lam_base - gammaln(N + 1.0)
    dll_bin = np.sum(ll_pred_bin - ll_base_bin, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.plot(log10M_centers, dll_bin, lw=2)
    ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    ax.set_ylabel(r"$\Delta \log \mathcal{L}$ (pred - baseline), summed over spheres")
    fig.savefig(outdir / "conditional_hmf_eval_calibration_by_mass.png", dpi=150)
    plt.close(fig)

    # Zero-count calibration by mass bin.
    p0_obs = np.mean(N == 0, axis=0)
    p0_pred = np.mean(np.exp(-lam_pred), axis=0)
    p0_base = np.mean(np.exp(-lam_base), axis=0)
    # Fractions where Poisson mode/median are exactly zero.
    mode0_pred = np.mean(lam_pred < 1.0, axis=0)
    median0_pred = np.mean(lam_pred <= np.log(2.0), axis=0)

    # fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    # ax.plot(log10M_centers, p0_obs, lw=2, label="Observed frac(N=0)")
    # ax.plot(log10M_centers, p0_pred, lw=2, label="Emulator mean P(N=0)")
    # ax.plot(log10M_centers, p0_base, lw=1.5, ls="--", label="Baseline mean P(N=0)")
    # ax.plot(log10M_centers, mode0_pred, lw=1.5, ls=":", label="Emulator frac(mode=0)")
    # ax.plot(log10M_centers, median0_pred, lw=1.5, ls="-.", label="Emulator frac(median=0)")
    # ax.set_ylim(-0.02, 1.02)
    # ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    # ax.set_ylabel("Zero-count probability")
    # ax.legend(frameon=False)
    # fig.savefig(outdir / "conditional_hmf_eval_pzero_by_mass.png", dpi=150)
    # plt.close(fig)

    # Performance vs overdensity: average normalized (negative) log-likelihood in log(1+delta) quantile bins.
    # Using -ll turns "error" into a positive quantity where larger means worse fit.
    #
    # Normalize by "exposure" using the observed total halo count per sphere (sum over mass bins),
    # so overdense/high-count regions don't automatically look worse simply due to more events.
    totN = N.sum(axis=1).astype(np.float64)
    denom = np.maximum(totN, 1.0)
    metric = (-ll_pred) / denom
    metric_base = (-ll_base) / denom
    finite = np.isfinite(metric) & np.isfinite(metric_base) & np.isfinite(delta)
    # Bin in log(1+delta), which matches the common modelling space and avoids over-weighting
    # large overdensities on a linear axis. Clip to keep log1p defined.
    d = np.log1p(np.clip(delta[finite], a_min=-1.0 + 1e-12, a_max=None))
    y = metric[finite]
    yb = metric_base[finite]
    q = np.linspace(0.0, 1.0, 11)
    edges = np.quantile(d, q)
    edges = np.unique(edges)
    if edges.size >= 3:
        mean = []
        lo = []
        hi = []
        mean_b = []
        xmid = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (d >= a) & (d <= b if b == edges[-1] else d < b)
            if not np.any(m):
                continue
            yy = y[m]
            yyb = yb[m]
            mean.append(float(np.mean(yy)))
            lo.append(float(np.quantile(yy, 0.16)))
            hi.append(float(np.quantile(yy, 0.84)))
            mean_b.append(float(np.mean(yyb)))
            xmid.append(float(0.5 * (a + b)))

        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        ax.plot(xmid, mean, marker="o", lw=2, label="Emulator")
        ax.plot(xmid, mean_b, marker="o", lw=1.5, ls="--", alpha=0.9, label="Baseline")
        ax.fill_between(xmid, lo, hi, alpha=0.25, linewidth=0)
        ax.set_xlabel(r"$\log(1+\delta_R)$ (quantile bins)")
        ax.set_ylabel(r"Mean $(-\log \mathcal{L})/\max(N,1)$ per sphere")
        ax.legend(frameon=False)
        fig.savefig(outdir / "conditional_hmf_eval_dloglik_vs_delta.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
