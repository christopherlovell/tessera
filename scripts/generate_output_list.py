#!/usr/bin/env python
"""
Generate a SWIFT output_list file with scale-factor checkpoints.

- Cosmology is read from a MUSIC2 config (Omega_m, Omega_b, H0).
- Times are spaced uniformly in cosmic time with an early-time supersampling
  for z > z_split.
- Outputs are written in SWIFT's output_list format (lines of "aexp: <a>").
"""

import argparse
import configparser
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value


def read_cosmo(conf_path: Path):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    with conf_path.open() as fh:
        cfg.read_file(fh)
    cos = cfg["cosmology"]
    omega_m = float(str(cos["Omega_m"]).split()[0])
    omega_b = float(str(cos.get("Omega_b", 0.0)).split()[0])
    h = float(str(cos["H0"]).split()[0]) / 100.0
    cosmo = FlatLambdaCDM(H0=h * 100 * u.km / u.s / u.Mpc, Om0=omega_m, Ob0=omega_b)
    return cosmo


def build_times(cosmo, z_start: float, z_split: float, nsnap: int, supersample: int):
    t_start = cosmo.age(z_start).to(u.Gyr).value
    t_end = cosmo.age(0).to(u.Gyr).value
    t_split = cosmo.age(z_split).to(u.Gyr).value

    dt = (t_end - t_start) / nsnap
    dt_fine = dt / supersample

    t_base = np.arange(t_start, t_end + 0.5 * dt, dt)
    t_fine = np.arange(t_start, t_split + 0.5 * dt_fine, dt_fine)

    times = np.unique(np.concatenate([t_base, t_fine]))
    times.sort()
    return times


def times_to_scale_factors(cosmo, times_gyr):
    a_list = []
    for t in times_gyr:
        # z_at_value can struggle extremely close to z=0; clip to a tiny floor.
        try:
            z = z_at_value(cosmo.age, t * u.Gyr)
            # Guard against extremely small negative/positive z around 0
            if z < 1e-8:
                z = 0.0
        except Exception:
            z = 0.0
        a = 1.0 / (1.0 + z)
        a_list.append(float(a))
    return np.array(a_list)


def main():
    ap = argparse.ArgumentParser(description="Generate SWIFT output_list file (aexp list) with early-time supersampling.")
    ap.add_argument("--music-conf", type=Path, required=True, help="MUSIC2 config to read cosmology from.")
    ap.add_argument("--out", type=Path, required=True, help="Output file path for SWIFT output_list.")
    ap.add_argument("--z-start", type=float, default=20.0, help="Starting redshift for outputs.")
    ap.add_argument("--z-split", type=float, default=5.0, help="Redshift above which to supersample.")
    ap.add_argument("--nsnap", type=int, default=90, help="Approximate number of snapshots (uniform time spacing).")
    ap.add_argument("--supersample", type=int, default=3, help="Factor to refine dt for z>z_split.")
    args = ap.parse_args()

    cosmo = read_cosmo(args.music_conf)
    times = build_times(cosmo, args.z_start, args.z_split, args.nsnap, args.supersample)
    a = times_to_scale_factors(cosmo, times)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        fh.write(f"# Scale Factor\n")
        for aval in a:
            fh.write(f"{aval:.8f}\n")

    print(f"Wrote {len(a)} outputs to {args.out}")
    # Show a few samples for eyeballing
    for i in [0, 1, 2, len(a)//2, len(a)-10, len(a)-3, len(a)-2, len(a)-1]:
        z = 1.0 / a[i] - 1.0
        print(f"{i:4d}: a={a[i]:.6f}, z~{z:.3f}")


if __name__ == "__main__":
    main()
