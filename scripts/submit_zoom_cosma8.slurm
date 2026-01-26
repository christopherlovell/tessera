#!/bin/bash -l
#SBATCH -J swift_zoom
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -o logs/swift_zoom.%J.out
#SBATCH -e logs/swift_zoom.%J.err
#SBATCH -p cosma8
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH -t 24:00:00

set -euo pipefail

MUSIC_BIN="${MUSIC_BIN:-/cosma8/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC}"
SWIFT_BIN="${SWIFT_BIN:-/cosma8/data/dp004/dc-love2/codes/swiftsim/swift_mpi}"
RUN_DIR="${RUN_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
YAML="${YAML:-${RUN_DIR}/swift_zoom_params.yaml}"
RUN_LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${RUN_LOG_DIR}/swift_zoom.${SLURM_JOB_ID}.log"

module purge
module load intel_comp/2024.2.0 compiler-rt tbb compiler mpi
module load ucx/1.17.0 parallel_hdf5/1.14.4 fftw/3.3.10 parmetis/4.0.3-64bit gsl/2.8

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

# mpirun -np "${NPROCS}" "${SWIFT_BIN}" \
/cosma8/data/dp004/dc-love2/codes/swiftsim/swift \
  --cosmology \
  --self-gravity \
  --zoom \
  --fof \
  --threads="${SLURM_CPUS_PER_TASK}" \
  "${YAML}" 2>&1 | tee -a "${LOG_FILE}"
