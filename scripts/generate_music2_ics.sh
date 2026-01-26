#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--machine cosma7|cosma8] [--no-modules] [MUSIC2_CONFIG]

Options:
  -m, --machine    Select machine defaults (cosma7|cosma8). Default: cosma7
  --no-modules     Skip 'module purge/load' (use current environment)
  -h, --help       Show this help

Env overrides:
  MUSIC_BIN         Path to MUSIC2 executable (overrides machine default)
  MACHINE           Same as --machine
EOF
}

machine="${MACHINE:-cosma7}"
no_modules=0
cfg_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--machine)
      [[ $# -ge 2 ]] || { echo "ERROR: --machine requires a value" >&2; usage >&2; exit 2; }
      machine="$2"
      shift 2
      ;;
    --cosma7)
      machine="cosma7"
      shift
      ;;
    --cosma8)
      machine="cosma8"
      shift
      ;;
    --no-modules)
      no_modules=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      cfg_path="$1"
      shift
      ;;
  esac
done

case "$machine" in
  cosma7|cosma8) ;;
  *)
    echo "ERROR: unsupported --machine: $machine (expected cosma7 or cosma8)" >&2
    exit 2
    ;;
esac

MUSIC_BIN_DEFAULT_COSMA7="/cosma7/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC"
MUSIC_BIN_DEFAULT_COSMA8="/cosma8/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC"
if [[ "$machine" == "cosma7" ]]; then
  MUSIC_BIN_DEFAULT="$MUSIC_BIN_DEFAULT_COSMA7"
else
  MUSIC_BIN_DEFAULT="$MUSIC_BIN_DEFAULT_COSMA8"
fi
MUSIC_BIN="${MUSIC_BIN:-$MUSIC_BIN_DEFAULT}"

CFG="${cfg_path:-"$ROOT/configs/music2_parent_064.conf"}"

if [[ "$no_modules" -eq 0 ]]; then
  if command -v module >/dev/null 2>&1; then
    module purge
    if [[ "$machine" == "cosma7" ]]; then
      module load intel_comp/2025.1.1 compiler-rt tbb umf compiler mpi
      module load parmetis/4.0.3-64bit ucx/1.18.1 fftw/3.3.10cosma7 parallel_hdf5/1.14.4 gsl/2.8 metis/5.1.0-64bit
    elif [[ "$machine" == "cosma8" ]]; then
      module load intel_comp/2024.2.0 compiler-rt tbb compiler mpi ucx/1.17.0 parallel_hdf5/1.14.4 fftw/3.3.10 parmetis/4.0.3-64bit gsl/2.8
    else
      echo "ERROR: unsupported machine for module loading: $machine" >&2
      exit 2
    fi
  else
    echo "WARNING: 'module' command not found; skipping module setup." >&2
  fi
fi

if [[ ! -x "$MUSIC_BIN" ]]; then
  echo "MUSIC binary not found/executable at: $MUSIC_BIN" >&2
  exit 1
fi

echo "Running MUSIC with config: $CFG"
"$MUSIC_BIN" "$CFG"

# Add missing Masses datasets if needed
# if [[ -f "$ROOT/scripts/add_missing_masses.py" ]]; then
#   python "$ROOT/scripts/add_missing_masses.py" "$(awk -F'=| ' '/filename/{print $2}' "$CFG")"
# fi
