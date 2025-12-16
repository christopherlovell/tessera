#!/usr/bin/env bash
module purge
module load intel_comp/2025.1.1 compiler-rt tbb umf compiler mpi
module load parmetis/4.0.3-64bit ucx/1.18.1 fftw/3.3.10cosma7 parallel_hdf5/1.14.4 gsl/2.8 metis/5.1.0-64bit

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MUSIC_BIN="${MUSIC_BIN:-/cosma7/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC}"
CFG="${1:-"$ROOT/configs/music2_parent_064.conf"}"

if [[ ! -x "$MUSIC_BIN" ]]; then
  echo "MUSIC binary not found/executable at: $MUSIC_BIN" >&2
  exit 1
fi

mkdir -p "$ROOT/data"
echo "Running MUSIC with config: $CFG"
"$MUSIC_BIN" "$CFG"

# Add missing Masses datasets if needed
# if [[ -f "$ROOT/scripts/add_missing_masses.py" ]]; then
#   python "$ROOT/scripts/add_missing_masses.py" "$(awk -F'=| ' '/filename/{print $2}' "$CFG")"
# fi
