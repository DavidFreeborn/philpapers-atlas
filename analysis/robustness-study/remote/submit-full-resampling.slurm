#!/bin/bash
#SBATCH --job-name=pp_full_resample
#SBATCH --output=/projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/logs/full_resample_%A_%a.out
#SBATCH --error=/projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/logs/full_resample_%A_%a.err
#SBATCH --array=0-19%10
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --partition=short

set -euo pipefail
module load python/3.13
source /projects/ComputationalPhilosophyLab/LitReview/atlas/config/venv/bin/activate
export NUMBA_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u \
  /projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/remote_full_resampling.py \
  --repeat "${SLURM_ARRAY_TASK_ID}" \
  --output /projects/ComputationalPhilosophyLab/LitReview/robustness_study_20260827/full-resampling
