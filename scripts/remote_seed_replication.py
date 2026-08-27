#!/usr/bin/env python3
"""Linux replication branch for selected PhilPapers UMAP seeds.

This script is intended for the Explorer Slurm environment. It deliberately
duplicates the primary fit in a minimal form so platform effects can be audited
without importing local-path orchestration code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import hdbscan
import numpy as np
import umap


JOBS = [(dimension, seed) for dimension in (2, 10, 30) for seed in (11, 42, 73, 211, 547)]
PCA_PATH = Path(
    "/projects/ComputationalPhilosophyLab/LitReview/data/output/umap_input_vectors/pca_100d_vectors.npy"
)
PACKAGES = ("numpy", "scipy", "scikit-learn", "umap-learn", "hdbscan", "pynndescent")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dimension, seed = JOBS[args.task_id]
    output = args.output / f"d{dimension}" / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        print(f"Already complete: dimension={dimension}, seed={seed}")
        return

    pca = np.load(PCA_PATH, mmap_mode="r")
    started = perf_counter()
    embedding = umap.UMAP(
        n_components=dimension,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=False,
        verbose=False,
    ).fit_transform(pca).astype(np.float32, copy=False)
    umap_seconds = perf_counter() - started
    if embedding.shape != (69_400, dimension) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid UMAP output: {embedding.shape}")

    started = perf_counter()
    model = hdbscan.HDBSCAN(
        min_cluster_size=200,
        min_samples=15,
        metric="euclidean",
        cluster_selection_method="eom",
        gen_min_span_tree=True,
        prediction_data=False,
        core_dist_n_jobs=1,
    ).fit(embedding)
    hdbscan_seconds = perf_counter() - started
    labels = model.labels_.astype(np.int16, copy=False)
    assigned = labels >= 0
    counts = np.unique(labels[assigned], return_counts=True)[1]
    embedding_path = output / "embedding.npy"
    labels_path = output / "labels.npy"
    atomic_npy(embedding_path, embedding)
    atomic_npy(labels_path, labels)
    atomic_json(
        output / "run.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "dimension": dimension,
            "seed": seed,
            "n_papers": len(labels),
            "cluster_count": int(np.unique(labels[assigned]).size),
            "noise_count": int((~assigned).sum()),
            "noise_pct": float((~assigned).mean() * 100),
            "relative_validity": float(model.relative_validity_),
            "persistence_mean": float(np.mean(model.cluster_persistence_)),
            "membership_mean_assigned": float(np.mean(model.probabilities_[assigned])),
            "cluster_size_min": int(counts.min()),
            "cluster_size_median": float(np.median(counts)),
            "cluster_size_max": int(counts.max()),
            "umap_seconds": umap_seconds,
            "hdbscan_seconds": hdbscan_seconds,
            "embedding_sha256": sha256_file(embedding_path),
            "labels_sha256": sha256_file(labels_path),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        },
    )
    print(f"Complete: dimension={dimension}, seed={seed}")


if __name__ == "__main__":
    main()

