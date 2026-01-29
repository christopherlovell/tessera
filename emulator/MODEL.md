# Conditional HMF emulator: model summary

This document summarizes the **statistical model** implemented in `emulator/conditional_hmf.py` (training), `emulator/evaluate_conditional_hmf.py` (evaluation), and `emulator/global_hmf.py` (integration to a global HMF).

## Data and notation

For each selected region (sphere) \(s \in \{1,\dots,S\}\):

- Overdensity: \(\delta_s\) (defined by the gridder at kernel radius \(R\)).
- Volume: \(V_s\).
- Mass bins \(j \in \{1,\dots,J\}\) with edges \(\log_{10} M_{j}\) and widths
  \(\Delta \log_{10} M_j \equiv \log_{10} M_{j+1} - \log_{10} M_j\).
- Observed halo counts in each bin: \(N_{s j} \in \mathbb{N}_0\).

We model the **halo mass function** in each region in terms of a binwise density per log-mass:
\[
n_{s j} \equiv \left.\frac{dn}{d\log_{10} M}\right|_{s,j}.
\]

## Conditional Poisson likelihood

Counts are modelled as conditionally independent Poisson draws:
\[
N_{s j} \mid \lambda_{s j} \sim \mathrm{Poisson}(\lambda_{s j}),
\qquad
\lambda_{s j} = V_s \, \Delta \log_{10} M_j \, n_{s j}.
\]

Equivalently, defining \(\ell(\cdot)\) as the log-likelihood summed over \((s,j)\),
\[
\log p(N \mid \lambda)
=
\sum_{s=1}^S \sum_{j=1}^J
\left[N_{s j}\log(\lambda_{s j} + \epsilon) - \lambda_{s j} - \log(N_{s j}!)\right],
\]
with a small \(\epsilon\) for numerical stability in the \(\log\).

## Baseline + GP residual parameterization

The conditional HMF is parameterized as a **baseline** (unconditional) term plus a \(\delta\)-dependent deviation:
\[
\log n_{s j} = \log n^{\mathrm{base}}_{j} + r_{s j}.
\]

### Mass basis

Residuals \(r_{s j}\) are represented using a fixed set of basis functions over mass bins:
\[
r_{s j} = \sum_{k=1}^{K_\phi} a_{s k}\,\Phi_{k j}.
\]

Here \(\Phi \in \mathbb{R}^{K_\phi \times J}\) is the chosen mass basis:

- **PCA basis** (default): \(\Phi\) is learned from the training spheres via an SVD of log-density residuals.
- **B-spline basis** (`--mass-basis bspline`): \(\Phi\) is a B-spline basis evaluated at the mass-bin centers (with degree `--bspline-degree`).

In both cases the model uses the same likelihood and GP machinery; only \(\Phi\) changes.

### GP over overdensity for coefficient functions

Each mass-basis coefficient is a function of overdensity:
\[
a_k(\delta) \sim \mathcal{GP}\!\left(0,\, k_{\theta_k}(\delta,\delta')\right),
\qquad k=1,\dots,K_\phi.
\]

The kernel is chosen by `--gp-kernel`:

- RBF (squared exponential): \(k(\delta,\delta') = \sigma^2 \exp\!\left[-\tfrac{(\delta-\delta')^2}{2\ell^2}\right]\)
- Matérn \(5/2\):
\[
k(\delta,\delta') = \sigma^2
\left(1 + \sqrt{5}r + \frac{5}{3}r^2\right)\exp(-\sqrt{5}r),
\quad r=\frac{|\delta-\delta'|}{\ell}.
\]

Each GP has its own hyperparameters \(\theta_k = (\sigma_k,\ell_k,\mathrm{jitter}_k)\).

## MAP training via a whitened GP parameterization

Rather than sampling latent GP values directly, the code uses a **whitened** representation per mode \(k\):

- Let \(K_k(\theta_k)\in\mathbb{R}^{S\times S}\) be the kernel matrix on training \(\{\delta_s\}\) including jitter.
- Compute \(L_k\) such that \(K_k = L_k L_k^\top\) (Cholesky factorization).
- Introduce whitened latent variables \(z_k \sim \mathcal{N}(0, I)\) and set:
\[
a_k = L_k z_k.
\]

The objective is a MAP loss:
\[
\mathcal{L}(\Theta, Z)
= -\log p(N \mid \lambda(\Theta,Z)) \;-\; \log p(Z)
\;+\; \text{(optional penalties)}.
\]

The log prior \(\log p(Z)\) plus the Jacobian term implied by \(a=Lz\) yields:
\[
\log p(a\mid \theta) \equiv \log p(z) - \sum_i \log (L_{ii}) + \mathrm{const}.
\]

This is implemented as
\[
\log p = -\frac{1}{2}\|Z\|^2 - \sum_k \sum_i \log (L^{(k)}_{ii}).
\]

The model is optimized with Adam and optional learning-rate schedules.

## Optional objectives

### Tail objective

An optional “tail” term encourages matching cumulative abundance above a threshold mass bin:
\[
\mathcal{L}_{\mathrm{tail}} =
\sum_s \left[\log\!\Big(\sum_{j\ge j_0}\lambda_{s j} + \varepsilon\Big) - \log\!\Big(\sum_{j\ge j_0}N_{s j} + \varepsilon\Big)\right]^2,
\]
weighted by `--beta-tail`. The threshold is set by `--tail-top-bins`.

### Zero-bin leakage penalty

To penalize predicting nonzero expected counts when \(N_{sj}=0\), the log-likelihood’s \(-\lambda_{sj}\) term for zero bins can be upweighted by `--zero-bin-lam-weight`:
\[
\log p(N\mid \lambda)\;\mapsto\;
\log p(N\mid \lambda)\;-\;(\omega-1)\sum_{s,j:\,N_{sj}=0}\lambda_{sj},
\]
where \(\omega =\) `zero_bin_lam_weight` and \(\omega>1\) increases the penalty.

## Evaluation

Evaluation draws held-out spheres from the parent box (optionally excluding training centers), computes observed counts \(N_{sj}\), and compares:

- baseline-only prediction \(\lambda^{\mathrm{base}}_{sj}\),
- emulator prediction \(\lambda^{\mathrm{pred}}_{sj}\),

using per-sphere Poisson log-likelihood and various diagnostic plots (scatter per bin, HMF-vs-\(\delta\), etc.).

## Global HMF by integrating over the overdensity distribution

The global HMF is obtained by integrating the conditional prediction over the global overdensity distribution:
\[
n_{\mathrm{global}}(M) = \int n(M\mid \delta)\,p(\delta)\,d\delta
\approx \frac{1}{N}\sum_{i=1}^N n(M\mid \delta_i),
\]
using either:

- an **empirical** estimate \(p(\delta)\) from gridder samples, or
- a parametric Edgeworth approximation in \(y=\log(1+\delta)\).

When propagating GP uncertainty for the global HMF, the code treats \(\log n(M\mid\delta)\) as approximately Normal at fixed \(\delta\), so
\[
\mathbb{E}[n \mid \delta] \approx \exp\!\left(\mu_{\log n}(\delta) + \tfrac{1}{2}\sigma^2_{\log n}(\delta)\right),
\]
and can optionally Monte Carlo sample from the predictive \(\log n\) to produce a global 68% band.

