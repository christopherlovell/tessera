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
    # This file lives at tessera/emulator/*.py. Add the repo root (two levels up)
    # so `import tessera...` works when executed as a script.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_root_to_path()

from tessera.emulator.conditional_hmf import load_emulator, load_gridder_overdensity


def fit_overdensity_edgeworth(delta: np.ndarray, *, clip_min: float = -1.0 + 1e-12) -> dict[str, float | int]:
    """
    Fit an Edgeworth expansion for overdensity by modelling:

        y = log(1 + delta)

    and approximating the PDF of y via an Edgeworth expansion around a Normal with
    parameters (mu, sigma) plus higher moments (skewness, excess kurtosis).
    """
    delta = np.asarray(delta, dtype=np.float64).ravel()
    finite = np.isfinite(delta)
    valid = finite & (delta > float(clip_min))
    n_total = int(delta.size)
    n_valid = int(np.sum(valid))
    if n_valid < 2:
        raise ValueError(f"Not enough valid delta samples to fit: n_valid={n_valid} (n_total={n_total})")

    y = np.log1p(delta[valid])
    mu = float(np.mean(y))
    sigma = float(np.std(y, ddof=0))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0.0:
        raise FloatingPointError(f"Non-finite fit: mu={mu}, sigma={sigma}")

    z = (y - mu) / sigma
    skew = float(np.mean(z**3))
    kurt_excess = float(np.mean(z**4) - 3.0)
    if not np.isfinite(skew) or not np.isfinite(kurt_excess):
        raise FloatingPointError(f"Non-finite higher moments: skew={skew}, kurt_excess={kurt_excess}")
    return {"mu": mu, "sigma": sigma, "skew": skew, "kurt_excess": kurt_excess, "n_total": n_total, "n_used": n_valid}


def _edgeworth_correction(z: np.ndarray, *, skew: float, kurt_excess: float) -> np.ndarray:
    """
    Edgeworth correction factor w(z) multiplying the standard Normal pdf φ(z).

    Uses a common 4th-order expansion:
        w = 1 + (γ1/6) H3 + (γ2/24) H4 + (γ1^2/72) H6

    where γ1 is skewness and γ2 is excess kurtosis, and Hn are probabilists' Hermite polynomials.
    """
    z = np.asarray(z, dtype=np.float64)
    g1 = float(skew)
    g2 = float(kurt_excess)
    H3 = z**3 - 3.0 * z
    H4 = z**4 - 6.0 * z**2 + 3.0
    H6 = z**6 - 15.0 * z**4 + 45.0 * z**2 - 15.0
    return 1.0 + (g1 / 6.0) * H3 + (g2 / 24.0) * H4 + (g1 * g1 / 72.0) * H6


def _edgeworth_pdf_y(y: np.ndarray, *, mu: float, sigma: float, skew: float, kurt_excess: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    sig = float(sigma)
    z = (y - float(mu)) / sig
    phi = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    w = _edgeworth_correction(z, skew=float(skew), kurt_excess=float(kurt_excess))
    return (phi * w) / sig


def _edgeworth_pdf_log10_overdensity(x: np.ndarray, *, mu: float, sigma: float, skew: float, kurt_excess: float) -> np.ndarray:
    """
    PDF of x = log10(1+delta) induced by the Edgeworth model for y=log(1+delta).
    """
    x = np.asarray(x, dtype=np.float64)
    ln10 = np.log(10.0)
    y = x * ln10
    return _edgeworth_pdf_y(y, mu=float(mu), sigma=float(sigma), skew=float(skew), kurt_excess=float(kurt_excess)) * ln10


def _sample_y_normal(*, mu: float, sigma: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(loc=float(mu), scale=float(sigma), size=int(n))


def _infer_unique_kernel_radius_from_gridder(path: Path) -> float | None:
    """
    Best-effort inference of KernelRadius when not provided.

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


def plot_overdensity_distribution_with_fit(
    delta: np.ndarray,
    *,
    mu: float,
    sigma: float,
    skew: float,
    kurt_excess: float,
    out: Path,
    delta_selected: np.ndarray | None = None,
    nbins: int = 120,
    clip_min: float = -1.0 + 1e-12,
) -> None:
    """
    Make a PDF/CDF plot of log10(1+delta) and overlay the implied Normal fit.

    If log(1+delta) ~ Normal(mu, sigma^2), then log10(1+delta) ~ Normal(mu/ln(10), (sigma/ln(10))^2).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = np.asarray(delta, dtype=np.float64).ravel()
    finite = np.isfinite(delta)
    valid = finite & (delta > float(clip_min))
    if not np.any(valid):
        raise ValueError("No valid delta samples for plotting.")

    x = np.log10(1.0 + delta[valid])

    x_sel = None
    if delta_selected is not None:
        ds = np.asarray(delta_selected, dtype=np.float64).ravel()
        m = np.isfinite(ds) & (ds > float(clip_min))
        if np.any(m):
            x_sel = np.log10(1.0 + ds[m])

    xmin, xmax = float(np.min(x)), float(np.max(x))
    xx = np.linspace(xmin, xmax, 600)
    pdf = _edgeworth_pdf_log10_overdensity(xx, mu=float(mu), sigma=float(sigma), skew=float(skew), kurt_excess=float(kurt_excess))
    # Edgeworth expansions can go negative in the tails; clip for plotting and renormalize on the grid.
    pdf = np.where(np.isfinite(pdf), pdf, 0.0)
    pdf = np.maximum(pdf, 0.0)
    area = float(np.trapezoid(pdf, xx))
    if area > 0.0:
        pdf = pdf / area
    cdf = np.cumsum(pdf) * (xx[1] - xx[0])
    cdf = np.clip(cdf, 0.0, 1.0)

    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax_pdf.hist(x, bins=int(nbins), density=True, histtype="stepfilled", alpha=0.25, color="tab:blue")
    ax_pdf.hist(x, bins=int(nbins), density=True, histtype="step", lw=1.5, color="tab:blue", label="Measured")
    if x_sel is not None:
        ax_pdf.hist(
            x_sel,
            bins=int(nbins),
            density=True,
            histtype="step",
            lw=1.5,
            color="tab:green",
            label=f"Selected spheres (S={int(x_sel.size)})",
        )
    ax_pdf.plot(xx, pdf, color="tab:orange", lw=2.0, label="Edgeworth fit")
    ax_pdf.set_xlabel("log10(1 + overdensity)")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.legend(loc="best", frameon=False)

    xs = np.sort(x)
    ys = (np.arange(xs.size, dtype=np.float64) + 1.0) / float(xs.size)
    ax_cdf.plot(xs, ys, color="tab:blue", lw=1.5, label="Measured")
    if x_sel is not None:
        xs2 = np.sort(x_sel)
        ys2 = (np.arange(xs2.size, dtype=np.float64) + 1.0) / float(xs2.size)
        ax_cdf.plot(xs2, ys2, color="tab:green", lw=1.5, label="Selected spheres")
    ax_cdf.plot(xx, cdf, color="tab:orange", lw=2.0, label="Edgeworth fit")
    ax_cdf.set_xlabel("log10(1 + overdensity)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.grid(True, alpha=0.2)
    ax_cdf.legend(loc="best", frameon=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_overdensity_distribution(
    delta: np.ndarray,
    *,
    out: Path,
    delta_selected: np.ndarray | None = None,
    nbins: int = 120,
    clip_min: float = -1.0 + 1e-12,
) -> None:
    """
    Plot the empirical PDF/CDF of log10(1+delta) without any parametric fit overlay.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta = np.asarray(delta, dtype=np.float64).ravel()
    finite = np.isfinite(delta)
    valid = finite & (delta > float(clip_min))
    if not np.any(valid):
        raise ValueError("No valid delta samples for plotting.")

    x = np.log10(1.0 + delta[valid])

    x_sel = None
    if delta_selected is not None:
        ds = np.asarray(delta_selected, dtype=np.float64).ravel()
        m = np.isfinite(ds) & (ds > float(clip_min))
        if np.any(m):
            x_sel = np.log10(1.0 + ds[m])

    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax_pdf.hist(x, bins=int(nbins), density=True, histtype="stepfilled", alpha=0.25, color="tab:blue")
    ax_pdf.hist(x, bins=int(nbins), density=True, histtype="step", lw=1.5, color="tab:blue", label="Measured")
    if x_sel is not None:
        ax_pdf.hist(
            x_sel,
            bins=int(nbins),
            density=True,
            histtype="step",
            lw=1.5,
            color="tab:green",
            label=f"Selected spheres (S={int(x_sel.size)})",
        )
    ax_pdf.set_xlabel("log10(1 + overdensity)")
    ax_pdf.set_ylabel("PDF")
    ax_pdf.legend(loc="best", frameon=False)

    xs = np.sort(x)
    ys = (np.arange(xs.size, dtype=np.float64) + 1.0) / float(xs.size)
    ax_cdf.plot(xs, ys, color="tab:blue", lw=1.5, label="Measured")
    if x_sel is not None:
        xs2 = np.sort(x_sel)
        ys2 = (np.arange(xs2.size, dtype=np.float64) + 1.0) / float(xs2.size)
        ax_cdf.plot(xs2, ys2, color="tab:green", lw=1.5, label="Selected spheres")
    ax_cdf.set_xlabel("log10(1 + overdensity)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.grid(True, alpha=0.2)
    ax_cdf.legend(loc="best", frameon=False)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_global_hmf_comparison(
    *,
    log10M: np.ndarray,
    n_pred: np.ndarray,
    n_pred_lo: np.ndarray | None = None,
    n_pred_hi: np.ndarray | None = None,
    n_true: np.ndarray,
    n_true_lo: np.ndarray,
    n_true_hi: np.ndarray,
    n_base: np.ndarray | None = None,
    out: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log10M = np.asarray(log10M, dtype=np.float64)
    n_pred = np.asarray(n_pred, dtype=np.float64)
    n_pred_lo = None if n_pred_lo is None else np.asarray(n_pred_lo, dtype=np.float64)
    n_pred_hi = None if n_pred_hi is None else np.asarray(n_pred_hi, dtype=np.float64)
    n_true = np.asarray(n_true, dtype=np.float64)
    n_true_lo = np.asarray(n_true_lo, dtype=np.float64)
    n_true_hi = np.asarray(n_true_hi, dtype=np.float64)
    n_base = None if n_base is None else np.asarray(n_base, dtype=np.float64)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7.5), constrained_layout=True, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(log10M, n_true, lw=2, label="True (full box)")
    ax.fill_between(log10M, n_true_lo, n_true_hi, alpha=0.25, linewidth=0, label="True (Garwood 1σ)")
    pred_line = ax.plot(log10M, n_pred, lw=2, label="Emulator ⟨n(M|δ)⟩")[0]
    pred_color = pred_line.get_color()
    if n_pred_lo is not None and n_pred_hi is not None:
        ax.fill_between(log10M, n_pred_lo, n_pred_hi, alpha=0.18, linewidth=0, color=pred_color, label="Emulator (GP 68%)")
    if n_base is not None:
        ax.plot(log10M, n_base, lw=2, ls="--", color="0.35", label="Baseline model")
    ax.set_yscale("log")
    ax.set_ylabel(r"$dn/d\log_{10}M\;[\mathrm{Mpc}^{-3}]$")
    ax.legend(frameon=False)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = n_pred / n_true
        # Shot-noise (Poisson) uncertainty on the *true* curve only (Garwood interval).
        # Plot as a band around 1 in ratio space.
        shot_lo = n_true_lo / n_true
        shot_hi = n_true_hi / n_true
        gp_lo = None if n_pred_lo is None else (n_pred_lo / n_true)
        gp_hi = None if n_pred_hi is None else (n_pred_hi / n_true)
        ratio_base = None if n_base is None else (n_base / n_true)
    axr.axhline(1.0, color="k", lw=1, alpha=0.6)
    axr.fill_between(log10M, shot_lo, shot_hi, alpha=0.18, linewidth=0, color="k", label="Shot noise (Garwood 1σ)")
    if gp_lo is not None and gp_hi is not None:
        axr.fill_between(log10M, gp_lo, gp_hi, alpha=0.12, linewidth=0, color=pred_color, label="Emulator (GP 68%)")
    axr.plot(log10M, ratio, lw=2, color=pred_color)
    if ratio_base is not None:
        axr.plot(log10M, ratio_base, lw=2, ls="--", color="0.35", label="Baseline / true")
    axr.set_xlabel(r"$\log_{10}(M / 10^{10}\,M_\odot)$")
    axr.set_ylabel("Ratio")
    y_max = 2.0
    for arr in [ratio, shot_hi, ratio_base]:
        if arr is None:
            continue
        if np.any(np.isfinite(arr)):
            y_max = max(y_max, float(np.nanmax(arr)) * 1.05)
    axr.set_ylim(0.0, min(10.0, y_max))
    axr.grid(True, alpha=0.2)
    axr.legend(frameon=False, loc="best")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _predict_log_n_batch(model: dict, delta_eval: np.ndarray) -> np.ndarray:
    # Local copy of the evaluation predictor: returns log n(M|delta) for each delta.
    from tessera.emulator.conditional_hmf import _kernel_cross, _kernel_matrix

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
    gp_kernel = str(model.get("gp_kernel", "rbf")).lower().strip()

    delta_t = (delta_train - delta_mu) / delta_sig
    delta_eval_t = (jnp.asarray(delta_eval, dtype=dtype) - delta_mu) / delta_sig

    mus = []
    for k in range(Phi.shape[0]):
        Kk = _kernel_matrix(jnp, delta_t, amp[k], ell[k], jit[k], kind=gp_kernel)
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        k_star = _kernel_cross(jnp, delta_t, delta_eval_t, amp[k], ell[k], kind=gp_kernel)
        mus.append(k_star.T @ v)
    mus = jnp.stack(mus, axis=0)  # (K, S_eval)

    log_n = log_n_base[None, :] + (mus.T @ Phi)
    return np.asarray(log_n)


def _gp_posterior_cache(model: dict) -> dict[str, object]:
    """
    Build cached GP quantities that depend only on the trained model and training deltas.

    This avoids recomputing Cholesky factors for every delta batch when integrating.
    """
    from tessera.emulator.conditional_hmf import _kernel_matrix

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
    gp_kernel = str(model.get("gp_kernel", "rbf")).lower().strip()

    delta_t = (delta_train - delta_mu) / delta_sig

    Ls = []
    vs = []
    for k in range(Phi.shape[0]):
        Kk = _kernel_matrix(jnp, delta_t, amp[k], ell[k], jit[k], kind=gp_kernel)
        Lk = jnp.linalg.cholesky(Kk)
        v = jsp.linalg.solve_triangular(Lk.T, Z[k], lower=False)
        Ls.append(Lk)
        vs.append(v)

    return {
        "dtype": dtype,
        "Phi": Phi,
        "Phi2": Phi * Phi,
        "log_n_base": log_n_base,
        "delta_t": delta_t,
        "delta_mu": delta_mu,
        "delta_sig": delta_sig,
        "amp": amp,
        "ell": ell,
        "jit": jit,
        "gp_kernel": gp_kernel,
        "Ls": tuple(Ls),
        "vs": tuple(vs),
    }


def _predict_log_n_batch_mean_var(cache: dict[str, object], delta_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return predictive mean and variance of log n(M|delta) under the GP posterior.

    Notes:
    - Uses the standard GP conditional variance with K_train that includes the learned jitter term.
    - Returns the variance of the latent GP function (does not add jitter again).
    - Assumes independent GPs per mass-basis coefficient (as in training).
    """
    dtype = cache["dtype"]  # type: ignore[assignment]
    Phi = cache["Phi"]  # type: ignore[assignment]
    Phi2 = cache["Phi2"]  # type: ignore[assignment]
    log_n_base = cache["log_n_base"]  # type: ignore[assignment]
    delta_t = cache["delta_t"]  # type: ignore[assignment]
    delta_mu = float(cache["delta_mu"])  # type: ignore[arg-type]
    delta_sig = float(cache["delta_sig"])  # type: ignore[arg-type]
    amp = cache["amp"]  # type: ignore[assignment]
    ell = cache["ell"]  # type: ignore[assignment]
    Ls = cache["Ls"]  # type: ignore[assignment]
    vs = cache["vs"]  # type: ignore[assignment]
    gp_kernel = str(cache.get("gp_kernel", "rbf")).lower().strip()

    delta_eval_t = (jnp.asarray(delta_eval, dtype=dtype) - delta_mu) / delta_sig

    mu_coeffs = []
    var_coeffs = []
    for k in range(Phi.shape[0]):
        Lk = Ls[k]
        v = vs[k]
        from tessera.emulator.conditional_hmf import _kernel_cross
        k_star = _kernel_cross(jnp, delta_t, delta_eval_t, amp[k], ell[k], kind=gp_kernel)
        mu_k = k_star.T @ v

        w = jsp.linalg.solve_triangular(Lk, k_star, lower=True)
        var_k = (amp[k] ** 2) - jnp.sum(w * w, axis=0)
        var_k = jnp.clip(var_k, 0.0, jnp.inf)

        mu_coeffs.append(mu_k)
        var_coeffs.append(var_k)

    mu_coeffs = jnp.stack(mu_coeffs, axis=1)  # (B, K)
    var_coeffs = jnp.stack(var_coeffs, axis=1)  # (B, K)
    mean_log_n = log_n_base[None, :] + (mu_coeffs @ Phi)
    var_log_n = var_coeffs @ Phi2
    return np.asarray(mean_log_n), np.asarray(var_log_n)


def integrate_global_hmf_mc(
    model: dict,
    *,
    mu: float,
    sigma: float,
    skew: float,
    kurt_excess: float,
    n_mc: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte-Carlo estimate of:
        n_global(M) = E_{p(delta)}[ n(M|delta) ].
    """
    y = _sample_y_normal(mu=float(mu), sigma=float(sigma), n=int(n_mc), seed=int(seed))
    z = (y - float(mu)) / float(sigma)
    w = _edgeworth_correction(z, skew=float(skew), kurt_excess=float(kurt_excess))
    w = np.where(np.isfinite(w), w, 0.0)
    nneg = int(np.sum(w < 0.0))
    if nneg:
        # Keep the integration stable/positive by clipping negative weights.
        w = np.maximum(w, 0.0)
        print(f"edgeworth_negative_weight_frac {nneg / float(w.size):.6g}")
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        raise FloatingPointError("Edgeworth weights are not usable (sum <= 0).")

    delta = np.exp(y) - 1.0
    cache = _gp_posterior_cache(model)
    mean_log_n, var_log_n = _predict_log_n_batch_mean_var(cache, delta)  # (n_mc,J), (n_mc,J)
    # Lognormal correction: E[exp(X)] for X~N(mean,var) is exp(mean + 0.5 var).
    n_mean = np.exp(mean_log_n + 0.5 * var_log_n)
    n_global = np.sum((w[:, None] / sw) * n_mean, axis=0)
    edges = np.asarray(model["log10M_edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.asarray(n_global, dtype=np.float64)


def integrate_global_hmf_empirical(
    model: dict,
    *,
    delta_samples: np.ndarray,
    batch_size: int = 4096,
    gp_draws: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Empirical estimate of:
        n_global(M) = E_{empirical p(delta)}[ n(M|delta) ]

    using the delta samples directly (e.g. from the full gridder field).
    """
    delta_samples = np.asarray(delta_samples, dtype=np.float64).ravel()
    delta_samples = delta_samples[np.isfinite(delta_samples)]
    if delta_samples.size == 0:
        raise ValueError("No finite delta samples provided for empirical integration.")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    edges = np.asarray(model["log10M_edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    cache = _gp_posterior_cache(model)
    J = int(centers.size)
    acc = np.zeros((J,), dtype=np.float64)
    draws = int(gp_draws)
    draw_sums = None
    rng = None
    if draws > 0:
        draw_sums = np.zeros((draws, J), dtype=np.float64)
        rng = np.random.default_rng(int(seed) + 1234)

    n_total = int(delta_samples.size)
    for start in range(0, n_total, batch_size):
        d = delta_samples[start : start + batch_size]
        mean_log_n, var_log_n = _predict_log_n_batch_mean_var(cache, d)  # (B,J), (B,J)
        n_mean = np.exp(mean_log_n + 0.5 * var_log_n)
        acc = acc + np.sum(n_mean, axis=0)

        if draws > 0 and draw_sums is not None and rng is not None:
            # Monte Carlo propagation of GP predictive uncertainty, assuming per-delta independence.
            sd = np.sqrt(np.maximum(var_log_n, 0.0))
            # Chunk draws to keep memory bounded.
            chunk = 16
            B, Jb = mean_log_n.shape
            for ds0 in range(0, draws, chunk):
                ds1 = min(draws, ds0 + chunk)
                eps = rng.normal(size=(ds1 - ds0, B, Jb)).astype(np.float64)
                n_draw = np.exp(mean_log_n[None, :, :] + sd[None, :, :] * eps)
                draw_sums[ds0:ds1, :] += np.sum(n_draw, axis=1)

    n_global = acc / float(n_total)

    n_lo = None
    n_hi = None
    if draws > 0 and draw_sums is not None:
        n_draws = draw_sums / float(n_total)
        n_lo = np.percentile(n_draws, 16.0, axis=0)
        n_hi = np.percentile(n_draws, 84.0, axis=0)

    return centers, np.asarray(n_global, dtype=np.float64), n_lo, n_hi


def main() -> None:
    ap = argparse.ArgumentParser(description="Integrate conditional HMF emulator over a fitted p(delta) to get a global HMF.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fd = sub.add_parser("fit-delta", help="Fit p(delta) from a gridder overdensity field (Edgeworth in y=log(1+delta)).")
    fd.add_argument("--model", type=Path, default=None, help="Model file to take defaults from (gridder_file, kernel_radius).")
    fd.add_argument(
        "--gridder-file",
        type=Path,
        default=None,
        help="Gridder file to fit p(delta) from (defaults to the model's `gridder_file` if present).",
    )
    fd.add_argument(
        "--kernel-radius",
        type=float,
        default=None,
        help="Kernel radius in Mpc (defaults to the model's `kernel_radius` if present).",
    )
    fd.add_argument("--clip-min", type=float, default=-1.0 + 1e-12)
    fd.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG for the overdensity PDF/CDF with fit (defaults to ../plots/global_hmf_overdensity_fit_<stem>.png).",
    )
    fd.add_argument("--nbins", type=int, default=120, help="Histogram bins for log10(1+delta).")

    ig = sub.add_parser("integrate", help="Integrate the emulator over the fitted p(delta) to estimate the global HMF.")
    ig.add_argument("--model", type=Path, required=True)
    ig.add_argument(
        "--gridder-file",
        type=Path,
        default=None,
        help="Gridder file used to estimate p(delta). Defaults to the `gridder_file` stored in the model (if present).",
    )
    ig.add_argument("--kernel-radius", type=float, default=None, help="Kernel radius in Mpc (defaults to the model value if available).")
    ig.add_argument("--n-mc", type=int, default=8192, help="Monte Carlo samples for the delta integral.")
    ig.add_argument("--seed", type=int, default=0)
    ig.add_argument("--x64", action="store_true", help="Enable JAX float64 (often improves stability).")
    ig.add_argument("--cpu", action="store_true", help="Force JAX to use CPU (set JAX_PLATFORM_NAME=cpu).")
    ig.add_argument(
        "--delta-method",
        type=str,
        default="empirical",
        choices=["empirical", "edgeworth"],
        help="How to obtain p(delta) for the global integral. 'empirical' uses the gridder overdensity samples directly; "
        "'edgeworth' fits an Edgeworth expansion in y=log(1+delta) and integrates using importance sampling.",
    )
    ig.add_argument(
        "--delta-subsample",
        type=int,
        default=0,
        help="If >0, randomly subsample this many delta values from the gridder field for the empirical integral/plot.",
    )
    ig.add_argument(
        "--gp-draws",
        type=int,
        default=64,
        help="If >0, draw from the GP predictive distribution to propagate emulator uncertainty into the global HMF "
        "(reports a 68% band). Set to 0 to disable.",
    )
    ig.add_argument("--batch-size", type=int, default=4096, help="Batch size for evaluating the emulator over delta samples.")
    ig.add_argument(
        "--delta-fit-plot",
        type=Path,
        default=None,
        help="Output PNG for the overdensity PDF/CDF with fit (defaults to ../plots/global_hmf_overdensity_fit_<modelstem>.png).",
    )
    ig.add_argument("--delta-fit-nbins", type=int, default=120, help="Histogram bins for log10(1+delta) fit plot.")
    ig.add_argument(
        "--parent-fof",
        type=Path,
        default=Path("/cosma7/data/dp004/dc-love2/data/tessera/parent/fof_0011.hdf5"),
        help="FOF catalogue for the full box, used to compute a 'true' global HMF for comparison.",
    )
    ig.add_argument(
        "--hmf-plot",
        type=Path,
        default=None,
        help="Output PNG for predicted vs true global HMF (defaults to ../plots/global_hmf_compare_<modelstem>.png).",
    )

    args = ap.parse_args()

    if getattr(args, "cpu", False):
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if getattr(args, "x64", False):
        import jax

        jax.config.update("jax_enable_x64", True)

    if args.cmd == "fit-delta":
        model = load_emulator(args.model) if args.model is not None else {}
        gridder_file = args.gridder_file
        if gridder_file is None and "gridder_file" in model:
            gridder_file = Path(str(model["gridder_file"]))
        if gridder_file is None:
            raise ValueError("Gridder file not provided. Pass --gridder-file or --model with `gridder_file` saved.")

        kernel_radius = args.kernel_radius
        if kernel_radius is None and "kernel_radius" in model:
            kernel_radius = float(model["kernel_radius"])
        if kernel_radius is None:
            kernel_radius = _infer_unique_kernel_radius_from_gridder(gridder_file)
        if kernel_radius is None:
            raise ValueError(
                "Kernel radius not available. Pass --kernel-radius, provide a model that stores it, "
                "or ensure the gridder file contains a single KernelRadius."
            )

        _pos, grid_delta = load_gridder_overdensity(gridder_file, kernel_radius=float(kernel_radius))
        fit = fit_overdensity_edgeworth(grid_delta, clip_min=float(args.clip_min))
        print(f"mu {fit['mu']:.8e}")
        print(f"sigma {fit['sigma']:.8e}")
        print(f"skew {fit['skew']:.8e}")
        print(f"kurt_excess {fit['kurt_excess']:.8e}")
        print(f"n_used {int(fit['n_used'])}  n_total {int(fit['n_total'])}")

        stem = (Path(args.model).stem if args.model is not None else Path(gridder_file).stem)
        outdir = Path(__file__).resolve().parent.parent / "plots"
        out = args.out if args.out is not None else (outdir / f"global_hmf_overdensity_fit_{stem}.png")
        plot_overdensity_distribution_with_fit(
            grid_delta,
            mu=float(fit["mu"]),
            sigma=float(fit["sigma"]),
            skew=float(fit["skew"]),
            kurt_excess=float(fit["kurt_excess"]),
            out=Path(out),
            nbins=int(args.nbins),
            clip_min=float(args.clip_min),
        )
        print(f"Saved {out}")
        return

    if args.cmd == "integrate":
        model = load_emulator(args.model)
        kernel_radius = args.kernel_radius
        if kernel_radius is None and "kernel_radius" in model:
            kernel_radius = float(model["kernel_radius"])
        if kernel_radius is None:
            raise ValueError("Kernel radius not available. Provide --kernel-radius or use a model that stores it.")

        gridder_file = args.gridder_file
        if gridder_file is None and "gridder_file" in model:
            gridder_file = Path(str(model["gridder_file"]))
        if gridder_file is None:
            raise ValueError("Gridder file not provided. Pass --gridder-file or retrain to save it into the model.")

        _pos, grid_delta = load_gridder_overdensity(gridder_file, kernel_radius=float(kernel_radius))
        outdir = Path(__file__).resolve().parent.parent / "plots"

        delta_used = np.asarray(grid_delta, dtype=np.float64)
        if int(args.delta_subsample) > 0 and delta_used.size > int(args.delta_subsample):
            rng = np.random.default_rng(int(args.seed) + 12345)
            delta_used = rng.choice(delta_used, size=int(args.delta_subsample), replace=False)

        plot_path = args.delta_fit_plot if args.delta_fit_plot is not None else (outdir / f"global_hmf_overdensity_fit_{Path(args.model).stem}.png")
        delta_selected = None
        if "delta_train" in model:
            delta_selected = np.asarray(model["delta_train"], dtype=np.float64)
        if args.delta_method == "empirical":
            plot_overdensity_distribution(delta_used, out=Path(plot_path), nbins=int(args.delta_fit_nbins), delta_selected=delta_selected)
            print(f"Saved {plot_path}")
            log10M, n_global, n_lo, n_hi = integrate_global_hmf_empirical(
                model,
                delta_samples=delta_used,
                batch_size=int(args.batch_size),
                gp_draws=int(args.gp_draws),
                seed=int(args.seed),
            )
        elif args.delta_method == "edgeworth":
            fit = fit_overdensity_edgeworth(delta_used)
            plot_overdensity_distribution_with_fit(
                delta_used,
                mu=float(fit["mu"]),
                sigma=float(fit["sigma"]),
                skew=float(fit["skew"]),
                kurt_excess=float(fit["kurt_excess"]),
                out=Path(plot_path),
                delta_selected=delta_selected,
                nbins=int(args.delta_fit_nbins),
            )
            print(f"Saved {plot_path}")
            # Edgeworth integration includes GP lognormal correction but does not currently provide a band.
            n_lo, n_hi = None, None
            log10M, n_global = integrate_global_hmf_mc(
                model,
                mu=float(fit["mu"]),
                sigma=float(fit["sigma"]),
                skew=float(fit["skew"]),
                kurt_excess=float(fit["kurt_excess"]),
                n_mc=int(args.n_mc),
                seed=int(args.seed),
            )
        else:
            raise ValueError(f"Unsupported --delta-method={args.delta_method!r}")

        # Optional diagnostic: compare to the "true" global HMF computed from the full box.
        try:
            from tessera.emulator.conditional_hmf import read_swift_fof
        except Exception:
            read_swift_fof = None

        if read_swift_fof is not None and args.parent_fof is not None:
            halo_pos, halo_mass, boxsize, _z = read_swift_fof(Path(args.parent_fof))
            _ = halo_pos  # unused
            edges = np.asarray(model["log10M_edges"], dtype=np.float64)
            pivot = float(model.get("mass_pivot_msun", 1e10))
            log10M_all = np.log10(np.clip(np.asarray(halo_mass, dtype=np.float64), 1e-300, None)) - np.log10(pivot)
            counts, _ = np.histogram(log10M_all, bins=edges)
            dlog10M = np.diff(edges)
            n_true = counts / (float(boxsize) ** 3 * dlog10M)
            print("full_box_counts_per_bin " + " ".join(str(int(c)) for c in np.asarray(counts).tolist()))

            # Garwood 68% (1σ) interval for Poisson counts, then convert to density.
            from scipy.stats import chi2

            alpha = 1.0 - 0.6826894921370859
            c = np.asarray(counts, dtype=np.float64)
            lo_c = np.where(c > 0, 0.5 * chi2.ppf(alpha / 2.0, 2.0 * c), 0.0)
            hi_c = 0.5 * chi2.ppf(1.0 - alpha / 2.0, 2.0 * (c + 1.0))
            vol = float(boxsize) ** 3
            n_true_lo = lo_c / (vol * dlog10M)
            n_true_hi = hi_c / (vol * dlog10M)

            outdir = Path(__file__).resolve().parent.parent / "plots"
            hmf_plot = args.hmf_plot if args.hmf_plot is not None else (outdir / f"global_hmf_compare_{Path(args.model).stem}.png")
            n_base = np.exp(np.asarray(model["log_n_base"], dtype=np.float64))
            plot_global_hmf_comparison(
                log10M=log10M,
                n_pred=n_global,
                n_pred_lo=n_lo,
                n_pred_hi=n_hi,
                n_true=n_true,
                n_true_lo=n_true_lo,
                n_true_hi=n_true_hi,
                n_base=n_base,
                out=Path(hmf_plot),
            )
            print(f"Saved {hmf_plot}")

        # Print: log10(M/1e10 Msun) center, global dn/dlog10M, log of same.
        for x, n in zip(log10M, n_global):
            print(f"{x:.8e} {n:.8e} {np.log(n + 1e-300):.8e}")
        return


if __name__ == "__main__":
    main()
