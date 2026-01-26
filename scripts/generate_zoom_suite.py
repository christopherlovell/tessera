#!/usr/bin/env python
"""
Run `generate_zoom_ics.py` repeatedly from a YAML specification to create a reproducible suite of zoom regions.

Example YAML
-----------
out_base: /snap8/scratch/dp004/dc-love2/tessera
common:
  machine: cosma8
  snap: /path/to/z0_snapshot.hdf5
  ic_snap: /path/to/parent_ic_snapshot.hdf5
  parent_config: configs/music2_parent_tessera.conf
  template: configs/music2_zoom.conf
  swift_template: configs/swift_zoom_params.yaml
regions:
  - label: kernel_A
    select_mode: kernel
    grid: /path/to/gridder_output.hdf5
    kernel_index: 0
    target_logdelta: 1.0
    target_nhigh: 1e8
  - label: halo_14
    select_mode: halo
    fof: /path/to/fof_catalogue.hdf5
    halo_mmin: 1e4
    halo_mmax: 3e4
    selection_radius: 5.0
    halo_seed: 123
    target_nhigh: 2e8
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from methods import existing_labels


ROOT = Path(__file__).resolve().parent.parent
GEN_SCRIPT = ROOT / "scripts" / "generate_zoom_ics.py"

_REPO_RELATIVE_PATH_KEYS = {"parent_config", "template", "swift_template"}


_KEY_TO_FLAG: dict[str, str] = {
    "select_mode": "--select-mode",
    "label": "--label",
    "grid": "--grid",
    "snap": "--snap",
    "ic_snap": "--ic-snap",
    "machine": "--machine",
    "kernel_index": "--kernel-index",
    "kernel_radius": "--kernel-radius",
    "target_logdelta": "--target-logdelta",
    "target_rank": "--target-rank",
    "max_tries": "--max-tries",
    "selection_radius": "--selection-radius",
    "out_base": "--out-base",
    "index": "--index",
    "parent_config": "--parent-config",
    "template": "--template",
    "out_config": "--out-config",
    "levelmin": "--levelmin",
    "levelmax": "--levelmax",
    "out_ics": "--out-ics",
    "plot_out": "--plot-out",
    "fof": "--fof",
    "halo_mmin": "--halo-mmin",
    "halo_mmax": "--halo-mmax",
    "halo_seed": "--halo-seed",
    "halo_rank": "--halo-rank",
    "halo_boundary": "--halo-boundary",
    "check_boundary_haloes": "--check-boundary-haloes",
    "target_nhigh": "--target-nhigh",
    "target_nhigh_rtol": "--target-nhigh-rtol",
    "pad_mpc": "--pad-mpc",
    "pad_min_mpc": "--pad-min-mpc",
    "pad_max_mpc": "--pad-max-mpc",
    "pad_tune_iters": "--pad-tune-iters",
    "allow_overlap": "--allow-overlap",
    "overlap_buffer_mpc": "--overlap-buffer-mpc",
    "swift_template": "--swift-template",
    "metadata": "--metadata",
}


def _normalize_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _resolve_path_value(spec_dir: Path, key: str, value) -> str:
    if value is None:
        return ""
    if key not in _REPO_RELATIVE_PATH_KEYS:
        return str(value)
    p = Path(str(value))
    if p.is_absolute():
        return str(p)
    cand1 = (spec_dir / p)
    if cand1.exists():
        return str(cand1)
    cand2 = (ROOT / p)
    if cand2.exists():
        return str(cand2)
    return str(cand1)


def _build_argv(params: dict, *, spec_dir: Path) -> list[str]:
    args: list[str] = []
    normalized = {_normalize_key(k): v for k, v in params.items()}

    for k in normalized:
        if k not in _KEY_TO_FLAG and k not in {"name"}:
            raise ValueError(f"Unknown key in suite spec: {k!r}")

    if "label" not in normalized and "name" in normalized:
        normalized["label"] = normalized["name"]

    for k, flag in _KEY_TO_FLAG.items():
        if k not in normalized:
            continue
        v = normalized[k]
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                args.append(flag)
            continue
        if k in _REPO_RELATIVE_PATH_KEYS:
            args.extend([flag, _resolve_path_value(spec_dir, k, v)])
        else:
            args.extend([flag, str(v)])

    return args


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a suite of zoom regions from a YAML spec.")
    ap.add_argument("spec", type=Path, help="YAML suite specification.")
    ap.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    ap.add_argument("--resume", action="store_true", help="Skip regions whose label already exists under out_base.")
    ap.add_argument("--out-base", type=Path, help="Override out_base from the YAML file.")
    args = ap.parse_args()
    spec_dir = args.spec.resolve().parent

    data = yaml.safe_load(args.spec.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{args.spec}: expected a YAML mapping at the root")
    common = data.get("common", {})
    regions = data.get("regions", None)
    if not isinstance(common, dict):
        raise ValueError(f"{args.spec}: 'common' must be a mapping")
    if not isinstance(regions, list) or not regions:
        raise ValueError(f"{args.spec}: 'regions' must be a non-empty list")

    out_base = args.out_base if args.out_base is not None else data.get("out_base", common.get("out_base", None))
    if out_base is None:
        raise ValueError(f"{args.spec}: missing out_base (set at top-level, in common, or pass --out-base)")
    out_base = Path(out_base)

    seen_labels = existing_labels(out_base) if args.resume else set()

    for i, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"{args.spec}: regions[{i}] must be a mapping")
        merged = dict(common)
        merged.update(region)
        merged["out_base"] = str(out_base)

        label = merged.get("label", merged.get("name", None))
        if args.resume:
            if not isinstance(label, str) or not label:
                raise ValueError(f"{args.spec}: regions[{i}] missing 'label' required for --resume")
            if label in seen_labels:
                print(f"[skip] {label}")
                continue

        argv = [args.python, str(GEN_SCRIPT)] + _build_argv(merged, spec_dir=spec_dir)
        cmd_str = " ".join(shlex.quote(a) for a in argv)
        if args.dry_run:
            print(cmd_str)
            continue
        print(f"[run] {label or f'region_{i}'}")
        subprocess.run(argv, check=True)


if __name__ == "__main__":
    main()
