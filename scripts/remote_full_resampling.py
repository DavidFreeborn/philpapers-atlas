#!/usr/bin/env python3
"""Full SPECTER→PCA→UMAP→HDBSCAN subsample refit for Explorer Slurm."""

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
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.decomposition import PCA


RAW_EMBEDDINGS = Path(
    "/projects/ComputationalPhilosophyLab/LitReview/atlas/data/pp_embs_allenai_specter_first_pool/embeddings.npy"
)
ENGLISH_INDICES = Path(
    "/projects/ComputationalPhilosophyLab/LitReview/data/output/"
    "pp_embs_allenai_specter_first_pool_pca_100_umap_30_hdbs_200_ms_15/english_indices.npy"
)
N_PAPERS = 69_400
FRACTION = 0.8
SEED_START = 104_729
PACKAGES = ("numpy", "scipy", "scikit-learn", "umap-learn", "hdbscan", "pynndescent")
CONFIGS = (
    {"id": "current_30d_42", "dimension": 30, "mcs": 200, "ms": 15, "implementation": "hdbscan"},
    {"id": "typical_30d_49", "dimension": 30, "mcs": 200, "ms": 15, "implementation": "hdbscan"},
    {"id": "robust_20d_49", "dimension": 20, "mcs": 150, "ms": 50, "implementation": "hdbscan"},
    {"id": "robust_20d_29", "dimension": 20, "mcs": 300, "ms": 50, "implementation": "hdbscan"},
    {"id": "direct_2d_70", "dimension": 2, "mcs": 200, "ms": 15, "implementation": "hdbscan"},
    {"id": "direct_2d_historical_params", "dimension": 2, "mcs": 150, "ms": 50, "implementation": "sklearn"},
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fit_labels(coordinates: np.ndarray, config: dict, fitted_mcs: int) -> tuple[np.ndarray, dict]:
    started = perf_counter()
    if config["implementation"] == "sklearn":
        model = SklearnHDBSCAN(
            min_cluster_size=fitted_mcs,
            min_samples=config["ms"],
            metric="euclidean",
            cluster_selection_method="eom",
            algorithm="kd_tree",
            leaf_size=60,
            n_jobs=4,
            copy=True,
        ).fit(coordinates)
        relative_validity = None
    else:
        model = hdbscan.HDBSCAN(
            min_cluster_size=fitted_mcs,
            min_samples=config["ms"],
            metric="euclidean",
            cluster_selection_method="eom",
            gen_min_span_tree=True,
            core_dist_n_jobs=4,
        ).fit(coordinates)
        relative_validity = float(model.relative_validity_)
    labels = model.labels_.astype(np.int16, copy=False)
    assigned = labels != -1
    return labels, {
        "cluster_count": int(np.unique(labels[assigned]).size),
        "noise_pct": float(np.mean(~assigned) * 100),
        "coverage": float(np.mean(assigned)),
        "membership_mean_assigned": float(model.probabilities_[assigned].mean()),
        "relative_validity": relative_validity,
        "fit_seconds": perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.repeat < 20:
        raise ValueError("repeat must be in [0, 19]")
    output = args.output / f"repeat{args.repeat:02d}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run.json").exists():
        print(f"Already complete: repeat={args.repeat}")
        return

    seed = SEED_START + args.repeat
    rng = np.random.default_rng(seed)
    sample_count = round(N_PAPERS * FRACTION)
    indices = np.sort(rng.choice(N_PAPERS, size=sample_count, replace=False)).astype(np.int32)
    raw = np.load(RAW_EMBEDDINGS, mmap_mode="r")
    english = np.load(ENGLISH_INDICES)
    if raw.shape != (73_440, 768) or english.shape != (N_PAPERS,):
        raise ValueError(f"Unexpected source shapes: raw={raw.shape}, english={english.shape}")
    values = np.asarray(raw[english[indices]], dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    pca_started = perf_counter()
    pca = PCA(n_components=100, svd_solver="randomized", random_state=seed)
    reduced = pca.fit_transform(values).astype(np.float32, copy=False)
    pca_seconds = perf_counter() - pca_started
    if reduced.shape != (sample_count, 100) or not np.isfinite(reduced).all():
        raise ValueError("Invalid PCA output")

    fits = []
    umap_metrics = {}
    for dimension in (2, 20, 30):
        started = perf_counter()
        coordinates = umap.UMAP(
            n_components=dimension,
            n_neighbors=15,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
            low_memory=False,
            verbose=False,
        ).fit_transform(reduced).astype(np.float32, copy=False)
        umap_metrics[str(dimension)] = {"seconds": perf_counter() - started}
        unique_configs = {
            (config["mcs"], config["ms"], config["implementation"]): config
            for config in CONFIGS
            if config["dimension"] == dimension
        }
        fitted_cache = {}
        for config in unique_configs.values():
            for strategy, fitted_mcs in (("scaled", round(config["mcs"] * FRACTION)), ("fixed", config["mcs"])):
                key = (config["mcs"], config["ms"], config["implementation"], strategy)
                labels, metrics = fit_labels(coordinates, config, fitted_mcs)
                fitted_cache[key] = labels
                fits.append(
                    {
                        "dimension": dimension,
                        "base_min_cluster_size": config["mcs"],
                        "fitted_min_cluster_size": fitted_mcs,
                        "min_samples": config["ms"],
                        "implementation": config["implementation"],
                        "strategy": strategy,
                        **metrics,
                    }
                )
        for config in (value for value in CONFIGS if value["dimension"] == dimension):
            for strategy in ("scaled", "fixed"):
                key = (config["mcs"], config["ms"], config["implementation"], strategy)
                atomic_npy(output / f"labels-{config['id']}-{strategy}.npy", fitted_cache[key])

    atomic_npy(output / "indices.npy", indices)
    atomic_json(
        output / "run.json",
        {
            "repeat": args.repeat,
            "seed": seed,
            "sample_count": sample_count,
            "fraction": FRACTION,
            "pca_refitted": True,
            "pca_solver": "randomized",
            "pca_random_state": seed,
            "pca_seconds": pca_seconds,
            "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
            "umap": umap_metrics,
            "fits": fits,
            "source_hashes": {
                "embeddings": sha256_file(RAW_EMBEDDINGS),
                "english_indices": sha256_file(ENGLISH_INDICES),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Complete repeat={args.repeat}; PCA={pca_seconds:.1f}s")


if __name__ == "__main__":
    main()
