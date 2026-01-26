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

# cosma7 defaults (override via env vars if needed)
MUSIC_BIN="${MUSIC_BIN:-/cosma7/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC}"
SWIFT_BIN="${SWIFT_BIN:-/cosma7/data/dp004/dc-love2/codes/SWIFT/swift_mpi}"

RUN_DIR="${RUN_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
YAML="${YAML:-${RUN_DIR}/swift_zoom_params.yaml}"
RUN_LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${RUN_LOG_DIR}/swift_zoom.${SLURM_JOB_ID}.log"

module purge
module load intel_comp/2025.1.1 compiler-rt tbb umf compiler mpi
module load parmetis/4.0.3-64bit ucx/1.18.1 fftw/3.3.10cosma7 parallel_hdf5/1.14.4 gsl/2.8 metis/5.1.0-64bit

mkdir -p "${RUN_LOG_DIR}"

echo "Using SWIFT_BIN=${SWIFT_BIN}"
echo "Using YAML=${YAML}"
echo "Run directory: ${RUN_DIR}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

NPROCS="${SLURM_NTASKS:-}"
if [[ -z "${NPROCS}" ]]; then
  NPROCS="${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}"
fi
echo "MPI tasks: ${NPROCS}"

echo "Initial working directory: $(pwd)"
cd "${RUN_DIR}"
echo "Working directory after cd: $(pwd)"

MUSIC_CONF="${MUSIC_CONF:-${RUN_DIR}/music2_zoom.conf}"
echo "Using MUSIC_CONF=${MUSIC_CONF}"

IC_PATH=""
if [[ -f "${YAML}" ]]; then
  IC_PATH="$(
    awk -F: '
      $0 ~ /^[[:space:]]*file_name[[:space:]]*:/ {
        sub(/^[^:]*:/, "", $0)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
        gsub(/^"|"$/, "", $0)
        gsub(/^'\''|'\''$/, "", $0)
        print $0
        exit
      }
    ' "${YAML}"
  )"
fi

if [[ -n "${IC_PATH}" && -f "${IC_PATH}" ]]; then
  echo "ICs present at ${IC_PATH}; skipping MUSIC2."
elif [[ -f "${MUSIC_CONF}" ]]; then
  echo "Running MUSIC2..."
  "${MUSIC_BIN}" "${MUSIC_CONF}" 2>&1 | tee -a "${LOG_FILE}"
else
  echo "MUSIC2 config ${MUSIC_CONF} not found; skipping MUSIC2."
fi

mpirun -np "${NPROCS}" "${SWIFT_BIN}" \
  --cosmology \
  --self-gravity \
  --zoom \
  --fof \
  --threads="${SLURM_CPUS_PER_TASK}" \
  "${YAML}" 2>&1 | tee -a "${LOG_FILE}"
