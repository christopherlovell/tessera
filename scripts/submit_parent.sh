#!/bin/bash -l
#SBATCH -J swift_parent
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH -p cosma7
#SBATCH -A dp276
#SBATCH -t 24:00:00
#SBATCH -o /snap7/scratch/dp276/dc-love2/tessera/logs/swift_parent.out
#SBATCH -e /snap7/scratch/dp276/dc-love2/tessera/logs/swift_parent.err

module purge
module load intel_comp/2025.1.1 compiler-rt tbb umf compiler mpi
module load parmetis/4.0.3-64bit ucx/1.18.1 fftw/3.3.10cosma7 parallel_hdf5/1.14.4 gsl/2.8 metis/5.1.0-64bit

set -euo pipefail
RUN_DIR="/snap7/scratch/dp276/dc-love2/tessera/parent/"
mkdir -p "$RUN_DIR/logs"
cd "$RUN_DIR"

if [ ! -f "$RUN_DIR/music2_ICs_parent.hdf5" ]; then
  echo "IC file missing, running MUSIC2..."
  /cosma7/data/dp004/dc-love2/codes/MUSIC2/build/MUSIC "$RUN_DIR/music2_parent.conf"
fi

echo "Running SWIFT..."
mpirun -np 8 /cosma7/data/dp004/dc-love2/codes/SWIFT/swift_mpi \
  --cosmology \
  --self-gravity \
  --threads=28 \
  "$RUN_DIR/swift_params.yaml"
