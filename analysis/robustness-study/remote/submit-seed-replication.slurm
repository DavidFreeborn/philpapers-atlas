#!/bin/bash
#SBATCH --job-name=pp_robust_seeds
#SBATCH --output=/projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/logs/seed_%A_%a.out
#SBATCH --error=/projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/logs/seed_%A_%a.err
#SBATCH --array=0-14%5
#SBATCH --time=02:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --partition=short

set -euo pipefail
module load python/3.13
source /projects/ComputationalPhilosophyLab/LitReview/atlas/config/venv/bin/activate
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -u \
  /projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/remote_seed_replication.py \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --output /projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/seed-runs
