#!/bin/bash -l
#SBATCH -J swift_zoom
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -o logs/swift_zoom.%J.out
#SBATCH -e logs/swift_zoom.%J.err
#SBATCH -p cosma7
#SBATCH -A dp276
#SBATCH --exclusive
#SBATCH -t 24:00:00

set -euo pipefail

MUSIC_BIN="${MUSIC_BIN:-/cosma7/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC}"
SWIFT_BIN="${SWIFT_BIN:-/cosma7/data/dp004/dc-love2/codes/SWIFT/swift_mpi}"
# Default to the directory where `sbatch` was invoked from (Slurm copies the script to a spool
# directory on the compute node, so using $BASH_SOURCE[0] would incorrectly point to /var/slurm/...).
RUN_DIR="${RUN_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MUSIC_CONF="${MUSIC_CONF:-${RUN_DIR}/music2_zoom.conf}"
YAML="${YAML:-${RUN_DIR}/swift_zoom_params.yaml}"
RUN_LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${RUN_LOG_DIR}/swift_zoom.${SLURM_JOB_ID}.log"

module purge
module load intel_comp/2025.1.1 compiler-rt tbb umf compiler mpi
module load parmetis/4.0.3-64bit ucx/1.18.1 fftw/3.3.10cosma7 parallel_hdf5/1.14.4 gsl/2.8 metis/5.1.0-64bit

mkdir -p "${RUN_LOG_DIR}"

extract_music2_filename() {
  # Extract `filename = ...` from [output] section of MUSIC2 INI config.
  # Handles trailing comments and whitespace.
  awk '
    BEGIN { in_output=0 }
    /^\[output\]/ { in_output=1; next }
    /^\[/ { in_output=0 }
    in_output && $0 ~ /^filename[ \t]*=/ {
      sub(/^[^=]*=/, "", $0)
      sub(/[ \t]*#.*/, "", $0)
      gsub(/^[ \t]+|[ \t]+$/, "", $0)
      print $0
      exit
    }
  ' "$1"
}

echo "Using SWIFT_BIN=${SWIFT_BIN}"
echo "Using YAML=${YAML}"
echo "Run directory: ${RUN_DIR}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

echo "Initial working directory: $(pwd)"
cd "${RUN_DIR}"
echo "Working directory after cd: $(pwd)"

# Optionally generate ICs if missing (mirrors `tessera/generate_music2_ics.sh` logic).
if [ -f "${MUSIC_CONF}" ]; then
  ICS_PATH="$(extract_music2_filename "${MUSIC_CONF}")"
  if [ -n "${ICS_PATH}" ] && [ ! -f "${ICS_PATH}" ]; then
    echo "ICs not found at ${ICS_PATH}, running MUSIC2..."
    "${MUSIC_BIN}" "${MUSIC_CONF}" 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "ICs present (or filename not found); skipping MUSIC2."
  fi
else
  echo "MUSIC2 config ${MUSIC_CONF} not found; skipping MUSIC2."
fi

mpirun -np 4 "${SWIFT_BIN}" \
  --cosmology \
  --self-gravity \
  --zoom \
  --threads="${SLURM_CPUS_PER_TASK}" \
  "${YAML}" 2>&1 | tee "${LOG_FILE}"
