#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _add_repo_root_to_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_root_to_path()

from tessera.emulator.conditional_hmf import (  # noqa: E402
    _choose_centers_stratified,
    _configure_jax,
    _jax_imports,
    _rbf_kernel,
    _sphere_volume,
    load_emulator,
    load_gridder_overdensity,
    read_swift_fof,
)

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
    jax, jnp, jsp = _jax_imports()

    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    Phi = jnp.asarray(model["Phi"], dtype=dtype)
    log_n_base = jnp.asarray(model["log_n_base"], dtype=dtype)
    delta_train = jnp.asarray(model["delta_train"], dtype=dtype)
    delta_mu = float(model["delta_mu"])
    delta_sig = float(model["delta_sig"]) + 1e-12

    amp = jnp.exp(jnp.asarray(model["log_amp"], dtype=dtype))
    ell = jnp.exp(jnp.asarray(model["log_ell"], dtype=dtype))
    jit = jnp.exp(jnp.asarray(model["log_jitter"], dtype=dtype))
    Z = jnp.asarray(model["Z"], dtype=dtype)

    delta_t = (delta_train - delta_mu) / delta_sig
    delta_eval_t = (jnp.asarray(delta_eval, dtype=dtype) - delta_mu) / delta_sig

    mus = []
    for k in range(Phi.shape[0]):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        d = delta_t[:, None] - delta_eval_t[None, :]
        k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
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
    jax, jnp, jsp = _jax_imports()

    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    Phi = jnp.asarray(model["Phi"], dtype=dtype)
    Phi2 = Phi * Phi
    log_n_base = jnp.asarray(model["log_n_base"], dtype=dtype)
    delta_train = jnp.asarray(model["delta_train"], dtype=dtype)
    delta_mu = float(model["delta_mu"])
    delta_sig = float(model["delta_sig"]) + 1e-12

    amp = jnp.exp(jnp.asarray(model["log_amp"], dtype=dtype))
    ell = jnp.exp(jnp.asarray(model["log_ell"], dtype=dtype))
    jit = jnp.exp(jnp.asarray(model["log_jitter"], dtype=dtype))
    Z = jnp.asarray(model["Z"], dtype=dtype)

    delta_t = (delta_train - delta_mu) / delta_sig
    delta_eval_t = (jnp.asarray(delta_eval, dtype=dtype) - delta_mu) / delta_sig

    mus = []
    vars_a = []
    for k in range(Phi.shape[0]):
        Kk = _rbf_kernel(jnp, delta_t, amp[k], ell[k], jit[k])
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        d = delta_t[:, None] - delta_eval_t[None, :]
        k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
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
    jax, jnp, _jsp = _jax_imports()
    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32

    delta_train = jnp.asarray(model["delta_train"], dtype=dtype)
    delta_mu = float(model["delta_mu"])
    delta_sig = float(model["delta_sig"]) + 1e-12
    delta_t = (delta_train - delta_mu) / delta_sig
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
    ap.add_argument("--parent-fof", type=Path, default=Path("/snap7/scratch/dp276/dc-love2/tessera/parent/fof_0011.hdf5"))
    ap.add_argument("--gridder-file", type=Path, default=Path("/snap7/scratch/dp276/dc-love2/tessera/parent/gridder/gridder_output_512.hdf5"))
    ap.add_argument(
        "--kernel-radius",
        type=float,
        default=None,
        help="Kernel radius in Mpc (required if not stored in the model and gridder has multiple kernels).",
    )
    ap.add_argument("--n-spheres", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1001, help="Seed for held-out sphere selection.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory for plots (default: ../plots relative to this script).")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--mc-delta", type=int, default=2048, help="MC samples for consistency check (0 disables).")
    ap.add_argument("--x64", action="store_true", help="Enable JAX float64 (often fixes prediction Cholesky NaNs).")
    ap.add_argument("--cpu", action="store_true", help="Force JAX to use CPU (set JAX_PLATFORM_NAME=cpu).")
    ap.add_argument("--debug-nans", action="store_true", help="Enable JAX NaN/Inf debugging.")
    args = ap.parse_args()

    if args.cpu:
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    _configure_jax(enable_x64=bool(args.x64), debug_nans=bool(args.debug_nans))

    model = load_emulator(args.model)
    kernel_radius = args.kernel_radius
    if kernel_radius is None and "kernel_radius" in model:
        kernel_radius = float(model["kernel_radius"])
    if kernel_radius is None:
        kernel_radius = _infer_unique_kernel_radius_from_gridder(args.gridder_file)
    if kernel_radius is None:
        raise ValueError(
            "Kernel radius not available. Provide `--kernel-radius`, or retrain/export a model that stores it "
            "(or ensure your gridder file contains a single KernelRadius)."
        )
    log10M_edges = np.asarray(model["log10M_edges"], dtype=np.float64)
    dlog10M = np.diff(log10M_edges)
    log10M_centers = 0.5 * (log10M_edges[:-1] + log10M_edges[1:])
    mass_pivot_msun = float(model.get("mass_pivot_msun", 1e10))

    halo_pos, halo_mass, boxsize, _z = read_swift_fof(args.parent_fof)
    grid_pos, grid_delta = load_gridder_overdensity(args.gridder_file, kernel_radius=kernel_radius)

    idx = _choose_centers_stratified(grid_delta, int(args.n_spheres), int(args.seed))
    centers = grid_pos[idx]
    delta = grid_delta[idx]

    from scipy.spatial import cKDTree

    L = float(boxsize)
    tree = cKDTree(np.mod(halo_pos, L), boxsize=L)
    hits = tree.query_ball_point(np.mod(centers, L), r=kernel_radius, workers=-1)

    log10M = np.log10(np.clip(halo_mass, 1e-300, None)) - np.log10(mass_pivot_msun)
    N = np.zeros((int(args.n_spheres), dlog10M.size), dtype=np.int32)
    for s, ids in enumerate(hits):
        if ids:
            N[s], _ = np.histogram(log10M[np.asarray(ids, dtype=np.int64)], bins=log10M_edges)

    V = float(_sphere_volume(kernel_radius))
    log_n_pred, var_log_n_pred = predict_log_n_batch_with_var(model, delta)
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

    print(f"n_spheres {N.shape[0]}  kernel_R {kernel_radius:g} Mpc")
    print(f"heldout delta[min,max] ({delta.min():.3g},{delta.max():.3g})")
    print(f"sum N {int(N.sum())}  mean N/sphere {float(N.sum()/N.shape[0]):.3g}")
    print(f"total loglik (pred) {float(ll_pred.sum()):.6g}")
    print(f"total loglik (base) {float(ll_base.sum()):.6g}")
    print(f"delta loglik (pred-base) {float((ll_pred-ll_base).sum()):.6g}")
    print(f"frac spheres improved {float(np.mean(ll_pred > ll_base)):.3f}")

    if int(args.mc_delta) > 0:
        rng = np.random.default_rng(int(args.seed) + 77)
        if grid_delta.size <= int(args.mc_delta):
            d_mc = np.asarray(grid_delta, dtype=np.float64)
        else:
            d_mc = rng.choice(np.asarray(grid_delta, dtype=np.float64), size=int(args.mc_delta), replace=False)
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
    ax.scatter(totLam, totN, s=10, alpha=0.6)
    lo = 0.0
    hi = max(float(np.max(totLam)), float(np.max(totN)), 1.0)
    ax.plot([lo, hi], [lo, hi], color="k", lw=1)
    ax.set_xlabel("Predicted total E[N] per sphere")
    ax.set_ylabel("Observed total N per sphere")
    fig.savefig(outdir / "conditional_hmf_eval_total_counts_scatter.png", dpi=150)
    plt.close(fig)

    # Individual HMFs at selected predictive-uncertainty percentiles.
    n_pred = np.exp(log_n_pred)
    n_base = np.exp(log_n_base[None, :])
    n_obs = N / (V * dlog10M[None, :])
    clo, chi = poisson_garwood_1sigma(N)
    n_lo = clo / (V * dlog10M[None, :])
    n_hi = chi / (V * dlog10M[None, :])

    # Choose example spheres by an estimated predictive uncertainty level.
    # We use a relative 1σ error implied by var(log n): sqrt(exp(var)-1), and then
    # take the median across mass bins as a single per-sphere scalar.
    rel_sigma = np.sqrt(np.expm1(np.maximum(var_log_n_pred, 0.0)))
    err_sphere = np.nanmedian(rel_sigma, axis=1)

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

    used: set[int] = set()
    picks: list[tuple[str, int]] = []
    for label, pct in [("e50", 50.0), ("e97", 97.0), ("e99", 99.0)]:
        for idx in pick_nearest(err_sphere, pct, 2, used):
            picks.append((label, idx))

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True, sharex=True, sharey=True)
    for ax, (label, s) in zip(axes.flat, picks):
        mask_pos = N[s] > 0
        mask_zero = ~mask_pos
        is_first = ax is axes.flat[0]
        if np.any(mask_pos):
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
        ax.plot(log10M_centers, n_base[0], lw=1.5, ls="--", label="Baseline (mean)" if is_first else None)

        # Emulator mode/median HMF (convert implied counts to density).
        mode_counts = poisson_mode(lam_pred[s])
        median_counts = poisson_median(lam_pred[s])
        n_mode = mode_counts / (V * dlog10M)
        n_median = median_counts / (V * dlog10M)

        n_mode_plot = np.where(mode_counts > 0, n_mode, np.nan)
        n_median_plot = np.where(median_counts > 0, n_median, np.nan)

        # Choose a sensible y-location for markers indicating exact zeros on a log axis.
        positives: list[np.ndarray] = []
        for arr in [n_pred[s], n_base[0], n_hi[s]]:
            sel = np.asarray(arr)[np.asarray(arr) > 0]
            if sel.size:
                positives.append(sel)
        y_floor = (float(np.min(np.concatenate(positives))) / 5.0) if positives else 1e-12

        ax.step(log10M_centers, n_mode_plot, where="mid", lw=1.6, ls=":", label="Emulator (mode)" if is_first else None)
        ax.step(
            log10M_centers,
            n_median_plot,
            where="mid",
            lw=1.6,
            ls="-.",
            label="Emulator (median)" if is_first else None,
        )
        if np.any(mode_counts == 0):
            ax.scatter(
                log10M_centers[mode_counts == 0],
                np.full(int(np.sum(mode_counts == 0)), y_floor),
                marker="v",
                s=18,
                linewidths=0,
                alpha=0.7,
                color=ax.lines[-2].get_color(),
            )
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
        ax.set_title(f"{label}: relerr={err_sphere[s]:.3g}, δ={delta[s]:.3g}, N={int(N[s].sum())}")
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$dn/d\log_{10}M\;[\mathrm{Mpc}^{-3}]$")
    fig.savefig(outdir / "conditional_hmf_eval_individual_hmfs.png", dpi=150)
    plt.close(fig)

    # Individual COUNT curves (mean/mode/median), which can be exactly zero.
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5), constrained_layout=True, sharex=True, sharey=True)
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
        ax.step(log10M_centers, lam_base[0], where="mid", lw=1.5, ls="--", label="Baseline (mean)" if is_first else None)

        # Mode/median implied by the Poisson model.
        ax.step(
            log10M_centers,
            poisson_mode(lam_pred[s]),
            where="mid",
            lw=1.8,
            ls=":",
            label="Emulator (mode)" if is_first else None,
        )
        ax.step(
            log10M_centers,
            poisson_median(lam_pred[s]),
            where="mid",
            lw=1.8,
            ls="-.",
            label="Emulator (median)" if is_first else None,
        )

        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"{label}: relerr={err_sphere[s]:.3g}, δ={delta[s]:.3g}, N={int(N[s].sum())}")

    axes[0, 0].legend(frameon=False, fontsize=9, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Counts per sphere per bin")
    fig.savefig(outdir / "conditional_hmf_eval_individual_counts.png", dpi=150)
    plt.close(fig)

    # Calibration by mass bin: mean standardized residuals.
    denom_pred = np.sqrt(np.maximum(lam_pred, 1e-30))
    denom_base = np.sqrt(np.maximum(lam_base, 1e-30))
    r_pred = (N - lam_pred) / denom_pred
    r_base = (N - lam_base) / denom_base
    r_pred_m = np.nanmean(r_pred, axis=0)
    r_base_m = np.nanmean(r_base, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.plot(log10M_centers, r_pred_m, lw=2, label="Emulator")
    ax.plot(log10M_centers, r_base_m, lw=1.5, ls="--", label="Baseline")
    ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    ax.set_ylabel(r"Mean $(N-\lambda)/\sqrt{\lambda}$")
    ax.legend(frameon=False)
    fig.savefig(outdir / "conditional_hmf_eval_calibration_by_mass.png", dpi=150)
    plt.close(fig)

    # Zero-count calibration by mass bin.
    p0_obs = np.mean(N == 0, axis=0)
    p0_pred = np.mean(np.exp(-lam_pred), axis=0)
    p0_base = np.mean(np.exp(-lam_base), axis=0)
    # Fractions where Poisson mode/median are exactly zero.
    mode0_pred = np.mean(lam_pred < 1.0, axis=0)
    median0_pred = np.mean(lam_pred <= np.log(2.0), axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(log10M_centers, p0_obs, lw=2, label="Observed frac(N=0)")
    ax.plot(log10M_centers, p0_pred, lw=2, label="Emulator mean P(N=0)")
    ax.plot(log10M_centers, p0_base, lw=1.5, ls="--", label="Baseline mean P(N=0)")
    ax.plot(log10M_centers, mode0_pred, lw=1.5, ls=":", label="Emulator frac(mode=0)")
    ax.plot(log10M_centers, median0_pred, lw=1.5, ls="-.", label="Emulator frac(median=0)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    ax.set_ylabel("Zero-count probability")
    ax.legend(frameon=False)
    fig.savefig(outdir / "conditional_hmf_eval_pzero_by_mass.png", dpi=150)
    plt.close(fig)

    # Performance vs overdensity: median ΔlogL in δ-quantile bins.
    dll = ll_pred - ll_base
    finite = np.isfinite(dll) & np.isfinite(delta)
    d = delta[finite]
    y = dll[finite]
    q = np.linspace(0.0, 1.0, 11)
    edges = np.quantile(d, q)
    edges = np.unique(edges)
    if edges.size >= 3:
        med = []
        lo = []
        hi = []
        xmid = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (d >= a) & (d <= b if b == edges[-1] else d < b)
            if not np.any(m):
                continue
            yy = y[m]
            med.append(float(np.median(yy)))
            lo.append(float(np.quantile(yy, 0.16)))
            hi.append(float(np.quantile(yy, 0.84)))
            xmid.append(float(0.5 * (a + b)))

        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        ax.axhline(0.0, color="k", lw=1, alpha=0.6)
        ax.plot(xmid, med, marker="o", lw=2)
        ax.fill_between(xmid, lo, hi, alpha=0.25, linewidth=0)
        ax.set_xlabel(r"$\delta_R$ (quantile bins)")
        ax.set_ylabel(r"Median $\Delta \log \mathcal{L}$ (pred - baseline)")
        fig.savefig(outdir / "conditional_hmf_eval_dloglik_vs_delta.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
