Conditional HMF emulator (WIP).

Defaults are wired to the Tessera parent run at `/snap7/scratch/dp276/dc-love2/tessera/parent` and its gridder output in `parent/gridder/`.

Train (requires `jax`):
`python tessera/emulator/conditional_hmf.py train`

If you don't have access to a global/full-box HMF (e.g. training from zoom regions), train with a baseline
estimated from the sampled spheres:
`python tessera/emulator/conditional_hmf.py train --baseline spheres`

If training hits NaNs, start with:
`python tessera/emulator/conditional_hmf.py train --check-data`
`python tessera/emulator/conditional_hmf.py train --x64 --init-jitter 1e-1 --no-jit --debug-nans --steps 1`

To disable the consistency penalty (i.e. do not enforce that integrating over p(δ) recovers the baseline), set:
`python tessera/emulator/conditional_hmf.py train --alpha-cons 0`

Training sphere selection enforces non-overlapping spheres by default (centre separation >= 2R). To allow overlaps:
`python tessera/emulator/conditional_hmf.py train --allow-overlap`

To force coverage of the very high-mass tail, you can also include centres nearest to the top-N most massive halos:
`python tessera/emulator/conditional_hmf.py train --include-top-halos 32`

To emphasize the high-mass tail during training, you can add a tail objective on cumulative counts above a mass threshold:
`python tessera/emulator/conditional_hmf.py train --beta-tail 1.0 --tail-top-bins 2`

Predict (prints `log10(M/1e10Msun)_center  log_n` per bin):
`python tessera/emulator/conditional_hmf.py predict --model conditional_hmf_emulator.h5 --delta 0.5`

Evaluate on held-out parent spheres (prints likelihood diagnostics + writes plots):
`python tessera/emulator/evaluate_conditional_hmf.py --model conditional_hmf_emulator.h5`

If the model file contains `train_center_idx`, evaluation will exclude those training centres when sampling
evaluation spheres (to reduce train/test overlap).

Recover a global HMF by integrating the conditional emulator over a model for the global overdensity PDF p(δ):
`python tessera/emulator/global_hmf.py fit-delta --gridder-file ... --kernel-radius 15`
`python tessera/emulator/global_hmf.py integrate --model conditional_hmf_emulator.h5 --gridder-file ...`

If the model stores `gridder_file` (newly trained models do), you can omit `--gridder-file` and it will be used by default:
`python tessera/emulator/global_hmf.py integrate --model conditional_hmf_emulator.h5`

Similarly, `fit-delta` can take defaults from the model:
`python tessera/emulator/global_hmf.py fit-delta --model conditional_hmf_emulator.h5`

Both `fit-delta` and `integrate` also write a PDF/CDF plot of log10(1+δ) with the fitted curve to `tessera/plots/`
by default (override with `--out` for `fit-delta` or `--delta-fit-plot` for `integrate`).

`integrate` also writes a diagnostic comparison of the predicted global HMF against the full-box ("true") HMF
using `--parent-fof` (default points at the parent run FOF catalogue); override the output path with `--hmf-plot`.

By default, `integrate` uses an empirical p(δ) (directly from the gridder samples). For a parametric tail model, use:
`python tessera/emulator/global_hmf.py integrate --model conditional_hmf_emulator.h5 --delta-method edgeworth`

If you see NaNs during evaluation, try:
`python tessera/emulator/evaluate_conditional_hmf.py --model conditional_hmf_emulator.h5 --x64`
or CPU:
`JAX_PLATFORM_NAME=cpu python tessera/emulator/evaluate_conditional_hmf.py --model conditional_hmf_emulator.h5 --x64`

Plots write to `tessera/plots/` by default (i.e. `../plots` relative to the script); override with `--outdir`.
