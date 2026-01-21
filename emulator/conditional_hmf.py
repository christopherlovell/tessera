#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masses_msun = np.asarray(masses_msun, dtype=np.float64)
    log10M = np.log10(np.clip(masses_msun, 1e-300, None)) - np.log10(MASS_PIVOT_MSUN)
    if log10M_min is None:
        log10M_min = float(np.min(log10M))
    if log10M_max is None:
        log10M_max = float(np.max(log10M))
    edges = np.linspace(log10M_min, log10M_max, int(n_bins) + 1)
    dlog10M = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, dlog10M


def hmf_dn_dlog10M(masses_msun: np.ndarray, volume_mpc3: float, log10M_edges: np.ndarray) -> np.ndarray:
    log10M = np.log10(np.clip(np.asarray(masses_msun, dtype=np.float64), 1e-300, None)) - np.log10(MASS_PIVOT_MSUN)
    counts, _ = np.histogram(log10M, bins=np.asarray(log10M_edges, dtype=np.float64))
    dlog10M = np.diff(log10M_edges)
    return counts / (float(volume_mpc3) * dlog10M)


def _sphere_volume(R: float) -> float:
    return float(4.0 / 3.0 * np.pi * R**3)


def _choose_centers_stratified(delta: np.ndarray, n: int, seed: int) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64)
    n = int(n)
    if n <= 0 or n > delta.size:
        raise ValueError(f"n={n} invalid for delta.size={delta.size}")
    order = np.argsort(delta)
    rng = np.random.default_rng(int(seed))
    edges = np.linspace(0, delta.size, n + 1, dtype=np.int64)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        j = int(rng.integers(int(a), max(int(a) + 1, int(b))))
        idx.append(int(order[j]))
    return np.asarray(idx, dtype=np.int64)


@dataclass(frozen=True)
class ParentDataset:
    delta: np.ndarray  # (S,)
    V: np.ndarray  # (S,)
    N: np.ndarray  # (S,J) int
    log10M_edges: np.ndarray  # (J+1,)
    log10M_centers: np.ndarray  # (J,)
    dlog10M: np.ndarray  # (J,)
    log_n_base: np.ndarray  # (J,)
    delta_grid: np.ndarray  # (Q,) for p(delta)


def build_parent_dataset(
    *,
    parent_fof: Path,
    gridder_file: Path,
    kernel_radius: float,
    n_spheres: int = 512,
    seed: int = 0,
    n_mass_bins: int = 25,
    log10M_min: float | None = None,
    log10M_max: float | None = None,
    n_delta_q: int = 2048,
) -> ParentDataset:
    centres_h, masses_h, boxsize, _z = read_swift_fof(parent_fof)
    grid_pos, grid_delta = load_gridder_overdensity(gridder_file, kernel_radius=float(kernel_radius))

    log10M_edges, log10M_centers, dlog10M = log10_mass_bins(
        masses_h, n_bins=int(n_mass_bins), log10M_min=log10M_min, log10M_max=log10M_max
    )
    log_n_base = np.log(hmf_dn_dlog10M(masses_h, boxsize**3, log10M_edges) + 1e-300)

    center_idx = _choose_centers_stratified(grid_delta, int(n_spheres), int(seed))
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

    V = np.full(int(n_spheres), _sphere_volume(float(kernel_radius)), dtype=np.float64)

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
        log10M_edges=np.asarray(log10M_edges, dtype=np.float64),
        log10M_centers=np.asarray(log10M_centers, dtype=np.float64),
        dlog10M=np.asarray(dlog10M, dtype=np.float64),
        log_n_base=np.asarray(log_n_base, dtype=np.float64),
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


def _jax_imports():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy as jsp
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "JAX is required for training/prediction; load an environment with `jax` installed."
        ) from exc
    return jax, jnp, jsp


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
    jit: bool = True,
):
    jax, jnp, jsp = _jax_imports()

    N = jnp.asarray(N)
    V = jnp.asarray(V)
    dlog10M = jnp.asarray(dlog10M)
    log_n_base = jnp.asarray(log_n_base)
    Phi = jnp.asarray(Phi)
    delta = jnp.asarray(delta)
    delta_q = jnp.asarray(delta_q)
    w_q = jnp.full((delta_q.shape[0],), 1.0 / float(delta_q.shape[0]), dtype=delta_q.dtype)

    mu_d = jnp.mean(delta)
    sig_d = jnp.std(delta) + 1e-12
    delta_t = (delta - mu_d) / sig_d
    delta_q_t = (delta_q - mu_d) / sig_d

    S, J = N.shape
    K = Phi.shape[0]

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

        mus = []
        for k in range(K):
            d = delta_t[:, None] - delta_q_t[None, :]
            k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
            mus.append(k_star.T @ Vvec[k])
        mus = jnp.stack(mus, axis=0)  # (K,Q)

        log_n_q = log_n_base[None, :] + (mus.T @ Phi)
        n_bar = (w_q[:, None] * jnp.exp(log_n_q)).sum(axis=0)
        penalty = float(alpha_cons) * jnp.sum((jnp.log(n_bar + 1e-30) - log_n_base) ** 2)
        return -(ll + lp) + penalty

    aux = {"delta_mu": mu_d, "delta_sig": sig_d, "delta_train_t": delta_t}
    return (jax.jit(loss) if jit else loss), aux


def train_map(
    loss_fn,
    *,
    K: int,
    S: int,
    steps: int = 5000,
    lr: float = 1e-2,
    seed: int = 0,
    jit: bool = True,
    init_jitter: float = 1e-2,
    clip_grad_norm: float = 10.0,
    print_every: int = 200,
):
    jax, jnp, _jsp = _jax_imports()

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
        params = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, mhat, vhat)
        return params, m, v, t, val
    step = jax.jit(step) if jit else step

    val0 = float(loss_fn(params))
    if not np.isfinite(val0):
        raise FloatingPointError(
            "Initial loss is non-finite. Try `--x64 --init-jitter 1e-1 --no-jit --debug-nans` "
            "and/or reduce `--alpha-cons`."
        )

    for i in range(int(steps)):
        params, m, v, t, val = step(params, m, v, t)
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
            print(i, float(val))
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
            f.create_dataset("Phi", data=np.asarray(Phi, dtype=np.float64))
            f.create_dataset("log10M_edges", data=np.asarray(log10M_edges, dtype=np.float64))
            f.create_dataset("log_n_base", data=np.asarray(log_n_base, dtype=np.float64))
            f.create_dataset("delta_train", data=np.asarray(delta_train, dtype=np.float64))
            f.create_dataset("log_amp", data=np.asarray(params["log_amp"]))
            f.create_dataset("log_ell", data=np.asarray(params["log_ell"]))
            f.create_dataset("log_jitter", data=np.asarray(params["log_jitter"]))
            f.create_dataset("Z", data=np.asarray(params["Z"]))
        return

    np.savez(
        out,
        Phi=np.asarray(Phi, dtype=np.float64),
        log10M_edges=np.asarray(log10M_edges, dtype=np.float64),
        mass_log_base=np.asarray(10, dtype=np.int64),
        mass_pivot_msun=np.asarray(MASS_PIVOT_MSUN, dtype=np.float64),
        log_n_base=np.asarray(log_n_base, dtype=np.float64),
        delta_mu=float(delta_mu),
        delta_sig=float(delta_sig),
        delta_train=np.asarray(delta_train, dtype=np.float64),
        kernel_radius=float(kernel_radius),
        log_amp=np.asarray(params["log_amp"]),
        log_ell=np.asarray(params["log_ell"]),
        log_jitter=np.asarray(params["log_jitter"]),
        Z=np.asarray(params["Z"]),
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
    return out


def predict_log_n(model: dict[str, np.ndarray | float], delta: float) -> tuple[np.ndarray, np.ndarray]:
    jax, jnp, jsp = _jax_imports()

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


def _configure_jax(*, enable_x64: bool, debug_nans: bool) -> None:
    jax, _jnp, _jsp = _jax_imports()
    if enable_x64:
        jax.config.update("jax_enable_x64", True)
    if debug_nans:
        jax.config.update("jax_debug_nans", True)
        jax.config.update("jax_debug_infs", True)


def _print_data_summary(ds: ParentDataset, Phi: np.ndarray | None) -> None:
    def rng(x):
        x = np.asarray(x)
        return float(np.nanmin(x)), float(np.nanmax(x))

    print(f"S={ds.delta.size} J={ds.dlog10M.size}")
    print(f"delta[min,max]={rng(ds.delta)} std={float(np.std(ds.delta)):.6g}")
    print(f"counts[min,max]={rng(ds.N)} total={int(ds.N.sum())}")
    print(f"log_n_base[min,max]={rng(ds.log_n_base)}")
    print(f"delta_grid[min,max]={rng(ds.delta_grid)}")
    if Phi is not None:
        print(f"Phi shape={Phi.shape} Phi[min,max]={rng(Phi)}")


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

    mus = []
    for k in range(Phi.shape[0]):
        d = delta_t[:, None] - delta_q_t[None, :]
        k_star = (amp[k] ** 2) * jnp.exp(-0.5 * (d**2) / (ell[k] ** 2))
        mus.append(k_star.T @ Vvec[k])
    mus = jnp.stack(mus, axis=0)
    log_n_q = log_n_base[None, :] + (mus.T @ Phi)
    n_bar = (w_q[:, None] * jnp.exp(log_n_q)).sum(axis=0)
    penalty = float(alpha_cons) * jnp.sum((jnp.log(n_bar + 1e-30) - log_n_base) ** 2)

    loss = -(ll + lp) + penalty
    stats = {
        "loss": loss,
        "ll": ll,
        "lp": lp,
        "penalty": penalty,
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
    tr.add_argument("--parent-fof", type=Path, default=Path("/snap7/scratch/dp276/dc-love2/tessera/parent/fof_0011.hdf5"))
    tr.add_argument("--gridder-file", type=Path, default=Path("/snap7/scratch/dp276/dc-love2/tessera/parent/gridder/gridder_output_512.hdf5"))
    tr.add_argument("--kernel-radius", type=float, default=15.0)
    tr.add_argument("--n-spheres", type=int, default=512)
    tr.add_argument("--n-mass-bins", type=int, default=25)
    tr.add_argument("--K", type=int, default=6, help="Number of PCA modes (excluding intercept).")
    tr.add_argument("--steps", type=int, default=5000)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--alpha-cons", type=float, default=512.0)
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
        if args.debug_nans or args.x64:
            _configure_jax(enable_x64=bool(args.x64), debug_nans=bool(args.debug_nans))
        ds = build_parent_dataset(
            parent_fof=args.parent_fof,
            gridder_file=args.gridder_file,
            kernel_radius=float(args.kernel_radius),
            n_spheres=int(args.n_spheres),
            seed=int(args.seed),
            n_mass_bins=int(args.n_mass_bins),
        )
        Phi = pca_mass_basis(ds.N, ds.V, ds.dlog10M, ds.log_n_base, K=int(args.K), add_intercept=True)
        if args.check_data:
            _print_data_summary(ds, Phi)
            return
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
                params=params0,
            )
            for k in ["loss", "ll", "lp", "penalty", "logdet", "lam_min", "lam_max", "log_n_min", "log_n_max", "Ldiag_min"]:
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
            jit=not bool(args.no_jit),
        )
        params = train_map(
            loss_fn,
            K=Phi.shape[0],
            S=ds.delta.size,
            steps=int(args.steps),
            lr=float(args.lr),
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
            kernel_radius=float(args.kernel_radius),
        )
        print(f"Wrote {args.out}")
    elif args.cmd == "predict":
        _configure_jax(enable_x64=False, debug_nans=False)
        model = load_emulator(args.model)
        log10M, log_n = predict_log_n(model, float(args.delta))
        for x, y in zip(log10M, log_n):
            print(f"{x:.8e} {y:.8e}")


if __name__ == "__main__":
    main()
