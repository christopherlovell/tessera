Conditional HMF emulator (WIP).

Defaults are wired to the Tessera parent run at `/snap7/scratch/dp276/dc-love2/tessera/parent` and its gridder output in `parent/gridder/`.

Train (requires `jax`):
`python tessera/emulator/conditional_hmf.py train`

If training hits NaNs, start with:
`python tessera/emulator/conditional_hmf.py train --check-data`
`python tessera/emulator/conditional_hmf.py train --x64 --init-jitter 1e-1 --no-jit --debug-nans --steps 1`

Predict (prints `log10(M/1e10Msun)_center  log_n` per bin):
`python tessera/emulator/conditional_hmf.py predict --model conditional_hmf_emulator.h5 --delta 0.5`

Evaluate on held-out parent spheres (prints likelihood diagnostics + writes plots):
`python tessera/emulator/evaluate_conditional_hmf.py --model conditional_hmf_emulator.h5`

If you see NaNs during evaluation, try:
`python tessera/emulator/evaluate_conditional_hmf.py --model conditional_hmf_emulator.h5 --x64`
or CPU:
`JAX_PLATFORM_NAME=cpu python tessera/emulator/evaluate_conditional_hmf.py --model conditional_hmf_emulator.h5 --x64`

Plots write to `tessera/plots/` by default (i.e. `../plots` relative to the script); override with `--outdir`.
