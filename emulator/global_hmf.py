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

from tessera.emulator.conditional_hmf import (
    infer_two_smallest_kernel_radii,
    load_emulator,
    load_gridder_overdensity,
    load_gridder_overdensity_pair,
)


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


def plot_overdensity_distribution_2d(
    delta12: np.ndarray,
    *,
    out: Path,
    delta_selected: np.ndarray | None = None,
    nbins: int = 160,
    clip_min: float = -1.0 + 1e-12,
    label1: str = r"$\log_{10}(1+\delta_1)$",
    label2: str = r"$\log_{10}(1+\delta_2)$",
) -> None:
    """
    Plot a 2D histogram of (log10(1+delta1), log10(1+delta2)) with marginal PDFs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    delta12 = np.asarray(delta12, dtype=np.float64)
    if delta12.ndim != 2 or delta12.shape[1] != 2:
        raise ValueError("delta12 must have shape (N,2)")
    m = (
        np.isfinite(delta12[:, 0])
        & np.isfinite(delta12[:, 1])
        & (delta12[:, 0] > float(clip_min))
        & (delta12[:, 1] > float(clip_min))
    )
    if not np.any(m):
        raise ValueError("No valid delta samples for 2D plotting.")
    d1 = np.clip(delta12[m, 0], float(clip_min), None)
    d2 = np.clip(delta12[m, 1], float(clip_min), None)
    x1 = np.log10(1.0 + d1)
    x2 = np.log10(1.0 + d2)

    nb = int(nbins)
    H, xedges, yedges = np.histogram2d(x1, x2, bins=nb)
    corr = float(np.corrcoef(x1, x2)[0, 1]) if x1.size >= 2 else float("nan")

    xs1 = None
    xs2 = None
    if delta_selected is not None:
        ds = np.asarray(delta_selected, dtype=np.float64)
        if ds.ndim == 2 and ds.shape[1] == 2:
            ms = (
                np.isfinite(ds[:, 0])
                & np.isfinite(ds[:, 1])
                & (ds[:, 0] > float(clip_min))
                & (ds[:, 1] > float(clip_min))
            )
            if np.any(ms):
                xs1 = np.log10(1.0 + np.clip(ds[ms, 0], float(clip_min), None))
                xs2 = np.log10(1.0 + np.clip(ds[ms, 1], float(clip_min), None))

    fig = plt.figure(figsize=(8.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4.0, 1.4], height_ratios=[1.4, 4.0], wspace=0.05, hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    vmax = float(np.max(H)) if H.size else 1.0
    norm = LogNorm(vmin=1.0, vmax=max(vmax, 1.0))
    im = ax.pcolormesh(xedges, yedges, H.T, shading="auto", norm=norm, cmap="viridis")
    cbar = fig.colorbar(im, ax=[ax, ax_top, ax_right], pad=0.01, fraction=0.046)
    cbar.set_label("Count per bin")

    if xs1 is not None and xs2 is not None:
        ax.scatter(xs1, xs2, s=10, c="tab:orange", alpha=0.45, linewidths=0, label=f"Selected (S={int(xs1.size)})")
        ax.legend(loc="best", frameon=False)

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(f"Overdensity 2D histogram (N={int(x1.size):,}, corr={corr:.3f})")

    ax_top.hist(x1, bins=xedges, density=True, histtype="step", lw=1.5, color="tab:blue", label="Parent")
    if xs1 is not None:
        ax_top.hist(xs1, bins=xedges, density=True, histtype="step", lw=1.5, color="tab:orange", label="Selected")
    ax_top.set_ylabel("PDF")
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.grid(True, alpha=0.15)
    ax_top.legend(loc="best", frameon=False)

    ax_right.hist(x2, bins=yedges, density=True, histtype="step", lw=1.5, color="tab:blue", orientation="horizontal")
    if xs2 is not None:
        ax_right.hist(xs2, bins=yedges, density=True, histtype="step", lw=1.5, color="tab:orange", orientation="horizontal")
    ax_right.set_xlabel("PDF")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.grid(True, alpha=0.15)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_global_hmf_comparison(
    *,
    log10M: np.ndarray,
    n_pred: np.ndarray,
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
    n_true = np.asarray(n_true, dtype=np.float64)
    n_true_lo = np.asarray(n_true_lo, dtype=np.float64)
    n_true_hi = np.asarray(n_true_hi, dtype=np.float64)
    n_base = None if n_base is None else np.asarray(n_base, dtype=np.float64)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.5, 7.5), constrained_layout=True, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(log10M, n_true, lw=2, label="True (full box)")
    ax.fill_between(log10M, n_true_lo, n_true_hi, alpha=0.25, linewidth=0, label="True (Garwood 1σ)")
    ax.plot(log10M, n_pred, lw=2, label="Emulator ⟨n(M|δ)⟩")
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
        ratio_base = None if n_base is None else (n_base / n_true)
    axr.axhline(1.0, color="k", lw=1, alpha=0.6)
    axr.fill_between(log10M, shot_lo, shot_hi, alpha=0.18, linewidth=0, color="k", label="Shot noise (Garwood 1σ)")
    axr.plot(log10M, ratio, lw=2)
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
    from tessera.emulator.conditional_hmf import _rbf_kernel

    dtype = jnp.float64 if bool(jax.config.read("jax_enable_x64")) else jnp.float32
    Phi = jnp.asarray(model["Phi"], dtype=dtype)
    log_n_base = jnp.asarray(model["log_n_base"], dtype=dtype)
    delta_mu = np.asarray(model["delta_mu"], dtype=np.float64).reshape(-1)
    delta_sig = np.asarray(model["delta_sig"], dtype=np.float64).reshape(-1) + 1e-12
    delta_train = np.asarray(model["delta_train"], dtype=np.float64)
    delta_eval = np.asarray(delta_eval, dtype=np.float64)
    if delta_train.ndim == 1:
        delta_train = delta_train[:, None]
    if delta_eval.ndim == 1:
        delta_eval = delta_eval[:, None]
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
    mus = jnp.stack(mus, axis=0)  # (K, S_eval)

    log_n = log_n_base[None, :] + (mus.T @ Phi)
    return np.asarray(log_n)


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
    dmu = np.asarray(model.get("delta_mu", np.asarray([0.0])), dtype=np.float64).reshape(-1)
    if dmu.size != 1:
        raise ValueError("Edgeworth integration is only implemented for 1D overdensity models.")
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
    log_n = _predict_log_n_batch(model, delta)  # (n_mc, J)
    n_global = np.sum((w[:, None] / sw) * np.exp(log_n), axis=0)
    edges = np.asarray(model["log10M_edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, np.asarray(n_global, dtype=np.float64)


def integrate_global_hmf_empirical(
    model: dict,
    *,
    delta_samples: np.ndarray,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Empirical estimate of:
        n_global(M) = E_{empirical p(delta)}[ n(M|delta) ]

    using the delta samples directly (e.g. from the full gridder field).
    """
    delta_samples = np.asarray(delta_samples, dtype=np.float64)
    if delta_samples.ndim == 1:
        delta_samples = delta_samples.reshape(-1, 1)
    if delta_samples.ndim != 2:
        raise ValueError("delta_samples must have shape (N,) or (N,D)")
    m = np.all(np.isfinite(delta_samples), axis=1)
    delta_samples = delta_samples[m]
    if delta_samples.shape[0] == 0:
        raise ValueError("No finite delta samples provided for empirical integration.")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    edges = np.asarray(model["log10M_edges"], dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    acc = None
    n_total = int(delta_samples.shape[0])
    for start in range(0, n_total, batch_size):
        d = delta_samples[start : start + batch_size]
        log_n = _predict_log_n_batch(model, d)  # (B, J)
        n = np.exp(log_n)
        s = np.sum(n, axis=0)
        acc = s if acc is None else (acc + s)

    n_global = acc / float(n_total)
    return centers, np.asarray(n_global, dtype=np.float64)


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
        # sphere radius is not needed for the integral; keep as optional metadata.
        gridder_file = args.gridder_file
        if gridder_file is None and "gridder_file" in model:
            gridder_file = Path(str(model["gridder_file"]))
        if gridder_file is None:
            raise ValueError("Gridder file not provided. Pass --gridder-file or retrain to save it into the model.")

        # Determine the overdensity kernel radii to use for sampling p(delta).
        kr1 = None
        kr2 = None
        if "kernel_radii" in model:
            rr = np.asarray(model["kernel_radii"], dtype=np.float64).reshape(-1)
            if rr.size >= 2:
                kr1, kr2 = float(rr[0]), float(rr[1])
        if kr1 is None or kr2 is None:
            kr1, kr2 = infer_two_smallest_kernel_radii(gridder_file)
        if float(kr2) < float(kr1):
            kr1, kr2 = kr2, kr1
        print(f"kernel_radii {float(kr1):g} {float(kr2):g}")

        _pos, grid_delta1, grid_delta2 = load_gridder_overdensity_pair(
            gridder_file, kernel_radius_1=float(kr1), kernel_radius_2=float(kr2)
        )
        outdir = Path(__file__).resolve().parent.parent / "plots"

        delta_used = np.stack([np.asarray(grid_delta1, dtype=np.float64), np.asarray(grid_delta2, dtype=np.float64)], axis=1)
        if int(args.delta_subsample) > 0 and delta_used.shape[0] > int(args.delta_subsample):
            rng = np.random.default_rng(int(args.seed) + 12345)
            pick = rng.choice(np.arange(int(delta_used.shape[0]), dtype=np.int64), size=int(args.delta_subsample), replace=False)
            delta_used = delta_used[pick]

        plot_path = args.delta_fit_plot if args.delta_fit_plot is not None else (outdir / f"global_hmf_overdensity_fit_{Path(args.model).stem}.png")
        dmu = np.asarray(model.get("delta_mu", np.asarray([0.0])), dtype=np.float64).reshape(-1)
        delta_selected = None
        if "delta_train" in model:
            delta_selected = np.asarray(model["delta_train"], dtype=np.float64)
        if args.delta_method == "empirical":
            if dmu.size == 1:
                # For 1D models, use the second kernel as the default delta axis in this 2-kernel loader.
                delta_1d = np.asarray(delta_used[:, 1], dtype=np.float64).reshape(-1)
                sel_1d = None
                if delta_selected is not None:
                    ds = np.asarray(delta_selected, dtype=np.float64)
                    sel_1d = ds if ds.ndim == 1 else ds[:, -1]
                plot_overdensity_distribution(
                    delta_1d,
                    out=Path(plot_path),
                    nbins=int(args.delta_fit_nbins),
                    delta_selected=sel_1d,
                )
            else:
                plot_overdensity_distribution_2d(
                    delta_used,
                    out=Path(plot_path),
                    nbins=int(args.delta_fit_nbins),
                    delta_selected=delta_selected,
                    label1=rf"$\log_{{10}}(1+\delta_{{R={float(kr1):g}}})$",
                    label2=rf"$\log_{{10}}(1+\delta_{{R={float(kr2):g}}})$",
                )
            print(f"Saved {plot_path}")
            log10M, n_global = integrate_global_hmf_empirical(
                model, delta_samples=(delta_1d if dmu.size == 1 else delta_used), batch_size=int(args.batch_size)
            )
        elif args.delta_method == "edgeworth":
            if dmu.size != 1:
                raise ValueError("Edgeworth delta-method is only implemented for 1D overdensity models.")
            delta_1d = np.asarray(delta_used[:, 1], dtype=np.float64).reshape(-1)
            fit = fit_overdensity_edgeworth(delta_1d)
            plot_overdensity_distribution_with_fit(
                delta_1d,
                mu=float(fit["mu"]),
                sigma=float(fit["sigma"]),
                skew=float(fit["skew"]),
                kurt_excess=float(fit["kurt_excess"]),
                out=Path(plot_path),
                delta_selected=(None if delta_selected is None else (np.asarray(delta_selected) if np.asarray(delta_selected).ndim == 1 else np.asarray(delta_selected)[:, -1])),
                nbins=int(args.delta_fit_nbins),
            )
            print(f"Saved {plot_path}")
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
