#!/usr/bin/env python
"""
Build SWIFT inputs for the Tessera parent run:
  1) Write a SWIFT YAML using cosmology/box/spacing from the MUSIC2 config.
  2) Write a Slurm submission script that runs MUSIC2 then SWIFT with that YAML.
"""

import argparse
import configparser
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent


def read_music_cosmo(conf_path: Path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    with conf_path.open() as fh:
        cfg.read_file(fh)
    cos = cfg["cosmology"]
    setup = cfg["setup"]
    omega_m = float(str(cos["Omega_m"]).split()[0])
    omega_b = float(str(cos.get("Omega_b", 0.0)).split()[0])
    h = float(str(cos["H0"]).split()[0]) / 100.0
    box = float(str(setup["boxlength"]).split()[0])
    gridres = int(str(setup["gridres"]).split()[0])
    return omega_m, omega_b, h, box, gridres


def write_swift_yaml(template: Path, ic_path: Path, out_path: Path,
                     h: float, omega_m: float, omega_b: float,
                     box: float, gridres: int):
    lines = template.read_text().splitlines()
    updated = []
    omega_lambda = 1.0 - omega_m  # flat
    omega_cdm = omega_m - omega_b
    spacing = box / gridres
    soft = spacing / 25.0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Omega_cdm"):
            updated.append(f"  Omega_cdm:      {omega_cdm:.8f}")
        elif stripped.startswith("Omega_lambda"):
            updated.append(f"  Omega_lambda:   {omega_lambda:.8f}")
        elif stripped.startswith("Omega_b"):
            updated.append(f"  Omega_b:        {omega_b:.8f}")
        elif stripped.startswith("h:"):
            updated.append(f"  h:              {h:.8f}")
        elif stripped.startswith("comoving_DM_softening"):
            updated.append(f"  comoving_DM_softening:     {soft:.4f}")
        elif stripped.startswith("max_physical_DM_softening"):
            updated.append(f"  max_physical_DM_softening: {soft:.4f}")
        elif stripped.startswith("file_name"):
            updated.append(f"  file_name:                   {ic_path}")
        else:
            updated.append(line)
    out_path.write_text("\n".join(updated) + "\n")
    return soft


def write_submit_script(music_conf: Path, ics: Path, swift_yaml: Path, out_path: Path,
                        music_bin: Path, swift_bin: Path,
                        partition: str = "cosma7", account: str = "dp276",
                        time: str = "24:00:00", cpus: int = 28):
    run_dir = ics.parent
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    script = dedent(f"""\
    #!/bin/bash -l
    #SBATCH -J tessera_swift
    #SBATCH --nodes=8
    #SBATCH --ntasks-per-node=1
    #SBATCH --cpus-per-task={cpus}
    #SBATCH -o {log_dir}/tessera_swift.%J.out
    #SBATCH -e {log_dir}/tessera_swift.%J.err
    #SBATCH -p {partition}
    #SBATCH -A {account}
    #SBATCH --exclusive
    #SBATCH -t {time}

    set -euo pipefail

    MUSIC_BIN="{music_bin}"
    SWIFT_BIN="{swift_bin}"
    MUSIC_CONF="{music_conf}"
    ICS="{ics}"
    YAML="{swift_yaml}"
    RUN_DIR="{run_dir}"
    LOG_FILE="{log_dir}/tessera_swift.${{SLURM_JOB_ID}}.log"

    module purge
    module load intel_comp/2025.1.1 compiler-rt tbb umf compiler mpi
    module load parmetis/4.0.3-64bit ucx/1.18.1 fftw/3.3.10cosma7 parallel_hdf5/1.14.4 gsl/2.8 metis/5.1.0-64bit

    mkdir -p "${{RUN_DIR}}"
    cd "${{RUN_DIR}}"

    if [ -f "${{ICS}}" ]; then
      echo "IC file ${{ICS}} exists, skipping MUSIC2."
    else
      echo "Running MUSIC2..."
      "${{MUSIC_BIN}}" "${{MUSIC_CONF}}" 2>&1 | tee "${{LOG_FILE}}"
    fi

    echo "Running SWIFT..."
    mpirun -np 8 "${{SWIFT_BIN}}" \\
      --cosmology --fof --self-gravity --threads={cpus} \\
      "${{YAML}}" 2>&1 | tee -a "${{LOG_FILE}}"
    """)

    out_path.write_text(script)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Write SWIFT YAML and submission script for Tessera parent.")
    ap.add_argument("--music-conf", type=Path, default=ROOT / "music2_parent_tessera.conf",
                    help="MUSIC2 config to read cosmology/box/grid from.")
    ap.add_argument("--ics", type=Path, required=True, help="Path to MUSIC2 ICs (SWIFT format).")
    ap.add_argument("--template", type=Path, default=ROOT.parent / "configs" / "swift_params.yaml",
                    help="SWIFT YAML template to start from.")
    ap.add_argument("--outdir", type=Path, default=ROOT, help="Output directory for YAML and submission script.")
    ap.add_argument("--output-times", type=Path, default=ROOT / "output_times.txt",
                    help="Optional output_list file to copy alongside the YAML.")
    ap.add_argument("--music-bin", type=Path, default=Path("/cosma7/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC"),
                    help="Path to MUSIC2 binary.")
    ap.add_argument("--swift-bin", type=Path, default=Path("/cosma7/data/dp004/dc-love2/codes/SWIFT/swift_mpi"),
                    help="Path to SWIFT binary.")
    ap.add_argument("--partition", default="cosma7")
    ap.add_argument("--account", default="dp276")
    ap.add_argument("--time", default="24:00:00")
    ap.add_argument("--cpus", type=int, default=28)
    args = ap.parse_args()

    omega_m, omega_b, h, box, gridres = read_music_cosmo(args.music_conf)
    args.outdir.mkdir(parents=True, exist_ok=True)
    yaml_out = args.outdir / "swift_params.yaml"
    submit_out = args.outdir / "submit_swift_parent.slurm"

    soft = write_swift_yaml(args.template, args.ics, yaml_out,
                            h=h, omega_m=omega_m, omega_b=omega_b, box=box, gridres=gridres)
    submit_path = write_submit_script(
        music_conf=args.music_conf,
        ics=args.ics,
        swift_yaml=yaml_out,
        out_path=submit_out,
        music_bin=args.music_bin,
        swift_bin=args.swift_bin,
        partition=args.partition,
        account=args.account,
        time=args.time,
        cpus=args.cpus,
    )
    # Copy output_times.txt if provided and exists
    if args.output_times.exists():
        target_times = args.outdir / args.output_times.name
        target_times.write_text(args.output_times.read_text())
        print(f"Copied output times to {target_times}")

    # Copy output_times.txt if provided and exists
    if args.output_times.exists():
        target_times = args.outdir / args.output_times.name
        target_times.write_text(args.output_times.read_text())
        print(f"Copied output times to {target_times}")

    # Copy the MUSIC config for reference
    target_music = args.outdir / args.music_conf.name
    target_music.write_text(args.music_conf.read_text())
    print(f"Copied MUSIC config to {target_music}")

    print(f"Wrote SWIFT params to {yaml_out} (Omega_m={omega_m}, Omega_b={omega_b}, h={h}, soft={soft:.4f} Mpc/h)")
    print(f"Wrote submission script to {submit_path}")
    if args.ics.exists():
        print(f"IC file {args.ics} already exists; MUSIC2 run will be skipped in the submit script.")


if __name__ == "__main__":
    main()
