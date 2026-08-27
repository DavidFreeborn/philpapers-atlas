# Reproducing the robustness study

## Environment

The local study environment is stored outside version control at
`analysis/robustness-study/work/venv312`. Its relevant package versions are fixed in
[`config.yaml`](config.yaml): Python 3.12.13, NumPy 1.26.4, SciPy 1.17.1,
scikit-learn 1.8.0, umap-learn 0.5.11, hdbscan 0.8.42 and pynndescent 0.6.0.
The Explorer jobs activate the project environment at
`/projects/ComputationalPhilosophyLab/LitReview/atlas/config/venv` and record their
actual versions in every `run.json`.

Large recovered sources and run checkpoints belong under
`analysis/robustness-study/work/`, which is ignored by Git. Verify their hashes
against `results/source-manifest.json` before running an experiment.

In the commands below, `python` means the pinned interpreter:

```powershell
$studyPython = 'analysis/robustness-study/work/venv312/Scripts/python.exe'
```

## Audit and controlled scans

```powershell
& $studyPython scripts/robustness_study.py audit
& $studyPython scripts/robustness_study.py umap-seeds --pca-dimensions 100 --jobs 3
& $studyPython scripts/robustness_study.py summarize-seeds --pca-dimensions 100
& $studyPython scripts/robustness_study.py hdbscan-grid --pca-dimensions 100 --jobs 3
& $studyPython scripts/robustness_study.py summarize-grid --pca-dimensions 100

& $studyPython scripts/robustness_study.py umap-seeds --pca-dimensions 50 --jobs 3
& $studyPython scripts/robustness_study.py summarize-seeds --pca-dimensions 50
& $studyPython scripts/robustness_study.py hdbscan-grid --pca-dimensions 50 --jobs 3
& $studyPython scripts/robustness_study.py summarize-grid --pca-dimensions 50

& $studyPython scripts/robustness_followup.py sensitivity --jobs 3
& $studyPython scripts/robustness_followup.py summarize-sensitivity
```

Completed cells are checkpointed atomically and reused unless `--force` is given.

## Linux seed replication

Upload `scripts/remote_seed_replication.py` and
`analysis/robustness-study/remote/submit-seed-replication.slurm` to the study
directory on Explorer, submit the array, and recover `seed-runs/` to
`work/remote-linux/seed-runs/`. Then run:

```powershell
& $studyPython scripts/summarize_remote_replication.py
```

The completed replication job was Slurm job 9750381.

## Resampling

The fixed-PCA diagnostic is local:

```powershell
& $studyPython scripts/robustness_followup.py resample --jobs 3
& $studyPython scripts/robustness_followup.py summarize-resample
```

The primary full-pipeline refits use
`scripts/remote_full_resampling.py` and
`analysis/robustness-study/remote/submit-full-resampling.slurm`. Recover the
completed `full-resampling/` directory to
`work/remote-linux/full-resampling/`, then run:

```powershell
& $studyPython scripts/summarize_full_resampling.py
```

The submitted full-refit array is Slurm job 9750727.

The historical 2D HDBSCAN settings have their own implementation check:

```powershell
& $studyPython scripts/robustness_2d_replication.py seeds --jobs 3
& $studyPython scripts/robustness_2d_replication.py summarize
& $studyPython scripts/robustness_2d_replication.py resamples --jobs 3
```

## Held-out semantic and consensus analysis

Do not run these commands until [`finalists.yaml`](finalists.yaml) has been frozen.

```powershell
& $studyPython scripts/robustness_semantic.py prepare
& $studyPython scripts/robustness_semantic.py evaluate
& $studyPython scripts/robustness_semantic.py consensus
& $studyPython scripts/robustness_finalize.py
```

`prepare` builds the original-SPECTER neighbour graph and performs the exact-neighbour
audit. `evaluate` writes held-out SPECTER and text results plus 100 permutation-null
runs per lens. `consensus` reads the full-pipeline subsample outputs, except for the
historical display lens, whose result is explicitly conditional on its fixed map.

## Verification

```powershell
& $studyPython -m pytest -q tests/test_robustness_study.py tests/test_robustness_semantic.py
& $studyPython scripts/verify_robustness_outputs.py
npm run lint
npm run build:pages
npm run test:3d
npm run test:3d:data
```

The final integrity pass must regenerate figures exclusively from saved CSV files,
verify every expected run count, rerun the provenance audit, and compare a sample of
saved HDBSCAN labels with fresh refits.
