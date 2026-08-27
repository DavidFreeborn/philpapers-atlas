#!/usr/bin/env python3
"""Checkpointed experiments for the PhilPapers clustering robustness study.

Run with the pinned environment documented in analysis/robustness-study/PLAN.md.
Large arrays and per-run labels are written below the ignored work directory;
compact, auditable tables are written to the tracked results directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score

from robustness_metrics import (
    NOISE,
    atomic_json,
    atomic_npy,
    best_match_jaccard,
    hdbscan_summary,
    optimal_label_mapping,
    pairwise_partition_rows,
    pareto_flags,
    partition_comparison,
    percentile_interval,
    sha256_file,
    stability_band,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
WORK = STUDY / "work"
SOURCE = WORK / "source"
RESULTS = STUDY / "results"
CONFIG_PATH = STUDY / "config.yaml"
N_PAPERS = 69_400
EXPECTED_HASHES = {
    "embeddings.npy": "0f98bf9256b205b94dde8e8ab9078b55312c55e591bcafabd97bbdb446f4721f",
    "english_indices.npy": "4ae97ff835edb0b5591ff72df8954e784c5b9a6a6f6794225ac280cb794afd2c",
    "preferred-cache/pca100.npy": "ddabc6dba24d264e16cac434c059f9dcb2d5b169fee36cea8b771ac7cd1feb20",
    "preferred-cache/umap30.npy": "d97c16cc9a152fcd1ffa2379d842f330d1e1440a982d3e4408ad36bf5df20047",
    "preferred-cache/umap2.npy": "efb1359d841bb67cb9a6fa4a74490efb87db51a95c0faa74819e85250d6765c4",
    "canonical/pca100.npy": "ddabc6dba24d264e16cac434c059f9dcb2d5b169fee36cea8b771ac7cd1feb20",
    "canonical/umap20.npy": "0b06d353f1888de0119698534b624d673a205570e1b37292af8c7c8e85336607",
    "canonical/umap25.npy": "f0eba27f33910c9be003b82d89374c39c5266fd1fa9853a5c5faa922f0bf74b8",
    "canonical/umap30.npy": "61ae465ed22afb1aca82fb1e3a8e92a1d543db497ea03076049b0b7df4aab9a8",
    "preferred-labels.npy": "41528899916696751b903b7d852430b088a162d184bf78c3b9da1c39b4040125",
    "unified-preferred/labels.npy": "41528899916696751b903b7d852430b088a162d184bf78c3b9da1c39b4040125",
}
PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "umap-learn",
    "hdbscan",
    "pynndescent",
    "pandas",
    "openpyxl",
    "matplotlib",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def ensure_finite(name: str, array: np.ndarray, shape: tuple[int, ...]) -> None:
    if array.shape != shape:
        raise ValueError(f"{name}: expected {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite values")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
        extras = sorted(set().union(*(row.keys() for row in rows)) - set(fieldnames))
        fieldnames.extend(extras)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def package_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}


def fit_hdbscan(
    coordinates: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    selection_method: str = "eom",
) -> tuple[object, np.ndarray, dict]:
    import hdbscan

    started = perf_counter()
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method=selection_method,
        gen_min_span_tree=True,
        prediction_data=False,
        core_dist_n_jobs=1,
    ).fit(coordinates)
    duration = perf_counter() - started
    labels = model.labels_.astype(np.int16, copy=False)
    summary = hdbscan_summary(model, labels, duration)
    return model, labels, summary


def filtered_paper_ids() -> list[str]:
    document_paths = [
        line.strip()
        for line in (SOURCE / "document_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    english = np.load(SOURCE / "english_indices.npy")
    if english.shape != (N_PAPERS,) or int(english.max()) >= len(document_paths):
        raise ValueError("Invalid English indices")
    return [Path(document_paths[int(index)]).stem for index in english]


def approximate_neighbor_overlap(
    left: np.ndarray,
    right: np.ndarray,
    neighbors: int,
    seed: int,
) -> dict[str, float]:
    from pynndescent import NNDescent

    left_index = NNDescent(left, n_neighbors=neighbors + 1, metric="euclidean", random_state=seed, n_jobs=-1)
    right_index = NNDescent(right, n_neighbors=neighbors + 1, metric="euclidean", random_state=seed, n_jobs=-1)
    left_neighbors = left_index.neighbor_graph[0][:, 1 : neighbors + 1]
    right_neighbors = right_index.neighbor_graph[0][:, 1 : neighbors + 1]
    overlaps = np.empty(len(left), dtype=np.float32)
    for index in range(len(left)):
        overlaps[index] = len(set(left_neighbors[index]).intersection(right_neighbors[index])) / neighbors
    return {
        "mean": float(overlaps.mean()),
        "median": float(np.median(overlaps)),
        "q05": float(np.quantile(overlaps, 0.05)),
        "q95": float(np.quantile(overlaps, 0.95)),
    }


def command_audit(_: argparse.Namespace) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    entries = []
    for path in manifest_files:
        relative = path.relative_to(SOURCE).as_posix()
        digest = sha256_file(path)
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "expected_sha256": EXPECTED_HASHES.get(relative),
                "hash_verified": EXPECTED_HASHES.get(relative) in {None, digest},
            }
        )
    failures = [entry for entry in entries if not entry["hash_verified"]]
    if failures:
        raise ValueError(f"Source hash failures: {failures}")

    pca_old = np.load(SOURCE / "preferred-cache" / "pca100.npy")
    pca_new = np.load(SOURCE / "canonical" / "pca100.npy")
    umap_old = np.load(SOURCE / "preferred-cache" / "umap30.npy")
    umap_new = np.load(SOURCE / "canonical" / "umap30.npy")
    display = np.load(SOURCE / "preferred-cache" / "umap2.npy")
    labels = np.load(SOURCE / "preferred-labels.npy")
    unified_labels = np.load(SOURCE / "unified-preferred" / "labels.npy")
    embeddings = np.load(SOURCE / "embeddings.npy", mmap_mode="r")
    ensure_finite("PCA100", pca_new, (N_PAPERS, 100))
    ensure_finite("canonical UMAP30", umap_new, (N_PAPERS, 30))
    ensure_finite("historical UMAP30", umap_old, (N_PAPERS, 30))
    ensure_finite("historical UMAP2", display, (N_PAPERS, 2))
    if embeddings.shape != (73_440, 768):
        raise ValueError(f"Unexpected source embeddings: {embeddings.shape}")
    if not np.array_equal(pca_old, pca_new):
        raise ValueError("Recovered PCA arrays are not identical")
    if not np.array_equal(labels, unified_labels):
        raise ValueError("Published and unified labels differ")

    _, refit_labels, canonical_fit = fit_hdbscan(umap_new, 200, 15)
    _, old_labels, historical_fit = fit_hdbscan(umap_old, 200, 15)
    canonical_fit["ari_to_published"] = float(adjusted_rand_score(labels, refit_labels))
    historical_fit["ari_to_published"] = float(adjusted_rand_score(labels, old_labels))
    if canonical_fit["ari_to_published"] != 1.0:
        raise ValueError("Pinned environment failed exact 42-label reproduction")

    map_data = json.loads((ROOT / "public" / "data" / "map.json").read_text(encoding="utf-8"))
    map_coordinates = np.asarray(map_data["points"], dtype=np.float32)[:, :2]
    if not np.allclose(map_coordinates, display, rtol=0, atol=1e-5):
        raise ValueError("Public map coordinates do not match recovered display within JSON precision")
    catalog = json.loads((ROOT / "public" / "data" / "lenses" / "catalog.json").read_text(encoding="utf-8"))
    default_lens = next(lens for lens in catalog["lenses"] if lens["id"] == catalog["defaultLens"])
    public_labels = np.fromfile(ROOT / "public" / default_lens["labelsFile"], dtype="<i2")
    if not np.array_equal(labels.astype(np.int16), public_labels):
        raise ValueError("Public default lens labels differ from the recovered labels")

    ids = filtered_paper_ids()
    search = json.loads((ROOT / "public" / "data" / "search.json").read_text(encoding="utf-8"))
    atlas_ids = [row[3].rsplit("/", 1)[-1] for row in search]
    if ids != atlas_ids:
        first = next(index for index, pair in enumerate(zip(ids, atlas_ids)) if pair[0] != pair[1])
        raise ValueError(f"Paper-order mismatch at {first}: {ids[first]} != {atlas_ids[first]}")

    rng = np.random.default_rng(8675309)
    audit_sample = np.sort(rng.choice(N_PAPERS, size=4_000, replace=False))
    trust = {
        "historical_umap30_from_pca100": float(
            trustworthiness(pca_new[audit_sample], umap_old[audit_sample], n_neighbors=15, metric="cosine")
        ),
        "canonical_umap30_from_pca100": float(
            trustworthiness(pca_new[audit_sample], umap_new[audit_sample], n_neighbors=15, metric="cosine")
        ),
        "historical_umap2_from_pca100": float(
            trustworthiness(pca_new[audit_sample], display[audit_sample], n_neighbors=15, metric="cosine")
        ),
        "historical_umap2_from_historical_umap30": float(
            trustworthiness(umap_old[audit_sample], display[audit_sample], n_neighbors=15, metric="euclidean")
        ),
        "sample_size": int(len(audit_sample)),
    }

    lineage = {
        "generated_at": utc_now(),
        "paper_order_exact": True,
        "pca_arrays_byte_identical": True,
        "umap30_arrays_byte_identical": bool(np.array_equal(umap_old, umap_new)),
        "umap30_flat_correlation": float(np.corrcoef(umap_old.ravel(), umap_new.ravel())[0, 1]),
        "umap30_mean_absolute_difference": float(np.abs(umap_old - umap_new).mean()),
        "canonical_fit": canonical_fit,
        "historical_fit": historical_fit,
        "trustworthiness_sample": trust,
        "neighbor_overlap": {
            "k15": approximate_neighbor_overlap(umap_old, umap_new, 15, 8675309),
            "k50": approximate_neighbor_overlap(umap_old, umap_new, 50, 8675309),
        },
    }
    atomic_json(RESULTS / "lineage-audit.json", lineage)
    atomic_json(
        RESULTS / "source-manifest.json",
        {
            "generated_at": utc_now(),
            "root": "analysis/robustness-study/work/source",
            "remote_project": "/projects/ComputationalPhilosophyLab/LitReview",
            "files": entries,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": package_versions(),
            },
        },
    )
    print(json.dumps(lineage, indent=2))


def umap_run_directory(pca_dimensions: int, dimension: int, seed: int) -> Path:
    return WORK / "runs" / f"umap-pca{pca_dimensions}" / f"d{dimension}" / f"seed{seed}"


def run_umap_seed(
    pca_path: str,
    pca_dimensions: int,
    dimension: int,
    seed: int,
    umap_config: dict,
    hdbscan_config: dict,
    force: bool,
) -> dict:
    directory = umap_run_directory(pca_dimensions, dimension, seed)
    meta_path = directory / "run.json"
    embedding_path = directory / "embedding.npy"
    labels_path = directory / "labels.npy"
    if not force and meta_path.exists() and embedding_path.exists() and labels_path.exists():
        result = json.loads(meta_path.read_text(encoding="utf-8"))
        result["cached"] = True
        return result

    import umap

    coordinates = np.load(pca_path, mmap_mode="r")
    if coordinates.shape != (N_PAPERS, pca_dimensions):
        raise ValueError(f"Unexpected PCA input shape {coordinates.shape}")
    started = perf_counter()
    reducer = umap.UMAP(
        n_components=dimension,
        n_neighbors=int(umap_config["n_neighbors"]),
        min_dist=float(umap_config["min_dist"]),
        metric=umap_config["metric"],
        random_state=seed,
        low_memory=bool(umap_config["low_memory"]),
        verbose=False,
    )
    embedding = reducer.fit_transform(coordinates).astype(np.float32, copy=False)
    umap_seconds = perf_counter() - started
    ensure_finite("UMAP output", embedding, (N_PAPERS, dimension))
    _, labels, summary = fit_hdbscan(
        embedding,
        int(hdbscan_config["min_cluster_size"]),
        int(hdbscan_config["min_samples"]),
        hdbscan_config["cluster_selection_method"],
    )
    atomic_npy(embedding_path, embedding)
    atomic_npy(labels_path, labels)
    result = {
        "completed_at": utc_now(),
        "pca_dimensions": pca_dimensions,
        "dimension": dimension,
        "seed": seed,
        "umap_n_neighbors": int(umap_config["n_neighbors"]),
        "umap_min_dist": float(umap_config["min_dist"]),
        "umap_metric": umap_config["metric"],
        "umap_low_memory": bool(umap_config["low_memory"]),
        "umap_seconds": float(umap_seconds),
        "embedding_sha256": sha256_file(embedding_path),
        "labels_sha256": sha256_file(labels_path),
        **summary,
    }
    atomic_json(meta_path, result)
    result["cached"] = False
    return result


def command_umap_seeds(args: argparse.Namespace) -> None:
    config = load_config()
    primary = config["umap_primary"]
    pca_dimensions = int(args.pca_dimensions)
    if pca_dimensions == 100:
        pca_path = SOURCE / "canonical" / "pca100.npy"
        dimensions = [int(value) for value in primary["dimensions"]]
        seeds = [int(value) for value in primary["seeds"]]
    elif pca_dimensions == 50:
        pca_path = SOURCE / "canonical-pca50" / "pca50.npy"
        dimensions = [int(value) for value in config["pca_sensitivity"]["umap_dimensions"]]
        seeds = [42, 211, 547]
    else:
        raise ValueError("Only the preregistered PCA50 and PCA100 branches are supported")
    jobs = [(dimension, seed) for dimension in dimensions for seed in seeds]
    print(f"Running/checking {len(jobs)} UMAP-to-HDBSCAN jobs with {args.jobs} workers", flush=True)
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(run_umap_seed)(
            str(pca_path),
            pca_dimensions,
            dimension,
            seed,
            primary,
            config["hdbscan_reference"],
            args.force,
        )
        for dimension, seed in jobs
    )
    cached = sum(bool(row["cached"]) for row in rows)
    print(f"Complete: {len(rows) - cached} fitted, {cached} cached", flush=True)


def canonical_embedding_path(dimension: int) -> Path | None:
    path = SOURCE / "canonical" / f"umap{dimension}.npy"
    return path if path.exists() else None


def command_summarize_seeds(args: argparse.Namespace) -> None:
    config = load_config()
    primary = config["umap_primary"]
    pca_dimensions = int(args.pca_dimensions)
    dimensions = (
        [int(value) for value in primary["dimensions"]]
        if pca_dimensions == 100
        else [int(value) for value in config["pca_sensitivity"]["umap_dimensions"]]
    )
    seeds = [int(value) for value in primary["seeds"]] if pca_dimensions == 100 else [42, 211, 547]
    run_rows: list[dict] = []
    pair_rows: list[dict] = []
    dimension_rows: list[dict] = []
    cluster_rows: list[dict] = []
    reproducibility_rows: list[dict] = []

    for dimension in dimensions:
        names = []
        labels_list = []
        metadata = []
        for seed in seeds:
            directory = umap_run_directory(pca_dimensions, dimension, seed)
            meta_path = directory / "run.json"
            labels_path = directory / "labels.npy"
            if not meta_path.exists() or not labels_path.exists():
                raise FileNotFoundError(f"Missing completed seed run: {directory}")
            row = json.loads(meta_path.read_text(encoding="utf-8"))
            names.append(f"seed{seed}")
            labels_list.append(np.load(labels_path))
            metadata.append(row)
            run_rows.append(row)

        comparisons = pairwise_partition_rows(names, labels_list)
        for row in comparisons:
            row.update({"pca_dimensions": pca_dimensions, "dimension": dimension})
        pair_rows.extend(comparisons)
        mean_ari = {
            name: float(
                np.mean(
                    [
                        row["ari_all"]
                        for row in comparisons
                        if row["left"] == name or row["right"] == name
                    ]
                )
            )
            for name in names
        }
        medoid_name = max(mean_ari, key=mean_ari.get)
        medoid_position = names.index(medoid_name)
        medoid_labels = labels_list[medoid_position]
        per_paper_agreement = np.ones(N_PAPERS, dtype=np.float32)
        for name, labels in zip(names, labels_list, strict=True):
            if name == medoid_name:
                continue
            mapping = optimal_label_mapping(medoid_labels, labels)
            mapped = np.full(N_PAPERS, NOISE, dtype=np.int16)
            for source, target in mapping.items():
                mapped[labels == source] = target
            per_paper_agreement += mapped == medoid_labels
        per_paper_agreement /= len(labels_list)
        atomic_npy(
            WORK / "summaries" / f"pca{pca_dimensions}-d{dimension}-paper-seed-agreement.npy",
            per_paper_agreement,
        )

        cluster_matches: dict[int, list[float]] = defaultdict(list)
        for name, labels in zip(names, labels_list, strict=True):
            if name == medoid_name:
                continue
            for match in best_match_jaccard(medoid_labels, labels):
                cluster_matches[int(match["reference_cluster"])].append(float(match["jaccard"]))
        medoid_sizes = Counter(int(value) for value in medoid_labels if value != NOISE)
        for cluster_id in sorted(cluster_matches):
            values = cluster_matches[cluster_id]
            low, high = percentile_interval(values)
            mean = float(np.mean(values))
            cluster_rows.append(
                {
                    "pca_dimensions": pca_dimensions,
                    "dimension": dimension,
                    "medoid_seed": int(medoid_name.removeprefix("seed")),
                    "cluster": cluster_id,
                    "size": medoid_sizes[cluster_id],
                    "seed_jaccard_mean": mean,
                    "seed_jaccard_median": float(np.median(values)),
                    "seed_jaccard_min": float(np.min(values)),
                    "seed_jaccard_ci_low": low,
                    "seed_jaccard_ci_high": high,
                    "stability_band": stability_band(mean),
                }
            )

        ari_values = [float(row["ari_all"]) for row in comparisons]
        ami_values = [float(row["ami_all"]) for row in comparisons]
        jaccard_values = [float(row["cluster_jaccard_weighted"]) for row in comparisons]
        dimension_rows.append(
            {
                "pca_dimensions": pca_dimensions,
                "dimension": dimension,
                "runs": len(labels_list),
                "medoid_seed": int(medoid_name.removeprefix("seed")),
                "cluster_count_median": float(np.median([row["cluster_count"] for row in metadata])),
                "cluster_count_min": int(min(row["cluster_count"] for row in metadata)),
                "cluster_count_max": int(max(row["cluster_count"] for row in metadata)),
                "noise_pct_median": float(np.median([row["noise_pct"] for row in metadata])),
                "relative_validity_median": float(np.median([row["relative_validity"] for row in metadata])),
                "relative_validity_min": float(min(row["relative_validity"] for row in metadata)),
                "relative_validity_max": float(max(row["relative_validity"] for row in metadata)),
                "pairwise_ari_mean": float(np.mean(ari_values)),
                "pairwise_ari_median": float(np.median(ari_values)),
                "pairwise_ari_min": float(np.min(ari_values)),
                "pairwise_ami_mean": float(np.mean(ami_values)),
                "pairwise_cluster_jaccard_weighted_mean": float(np.mean(jaccard_values)),
                "paper_assignment_agreement_mean": float(per_paper_agreement.mean()),
                "paper_assignment_agreement_q10": float(np.quantile(per_paper_agreement, 0.10)),
                "paper_assignment_agreement_q50": float(np.median(per_paper_agreement)),
                "paper_assignment_agreement_q90": float(np.quantile(per_paper_agreement, 0.90)),
            }
        )

        canonical_path = canonical_embedding_path(dimension) if pca_dimensions == 100 else None
        if canonical_path is not None and 42 in seeds:
            generated_dir = umap_run_directory(pca_dimensions, dimension, 42)
            generated = np.load(generated_dir / "embedding.npy")
            canonical = np.load(canonical_path)
            _, canonical_labels, _ = fit_hdbscan(canonical, 200, 15)
            generated_labels = np.load(generated_dir / "labels.npy")
            reproducibility_rows.append(
                {
                    "dimension": dimension,
                    "byte_identical": bool(np.array_equal(generated, canonical)),
                    "max_absolute_difference": float(np.max(np.abs(generated - canonical))),
                    "mean_absolute_difference": float(np.mean(np.abs(generated - canonical))),
                    "embedding_flat_correlation": float(np.corrcoef(generated.ravel(), canonical.ravel())[0, 1]),
                    "clustering_ari": float(adjusted_rand_score(canonical_labels, generated_labels)),
                }
            )

    write_csv(RESULTS / f"pca{pca_dimensions}-seed-runs.csv", run_rows)
    write_csv(RESULTS / f"pca{pca_dimensions}-seed-pairwise.csv", pair_rows)
    write_csv(RESULTS / f"pca{pca_dimensions}-seed-summary.csv", dimension_rows)
    write_csv(RESULTS / f"pca{pca_dimensions}-seed-cluster-stability.csv", cluster_rows)
    if reproducibility_rows:
        write_csv(RESULTS / f"pca{pca_dimensions}-canonical-reproduction.csv", reproducibility_rows)
    print(json.dumps(dimension_rows, indent=2))


def grid_directory(pca_dimensions: int, dimension: int, mcs: int, ms: int, method: str) -> Path:
    return (
        WORK
        / "runs"
        / f"hdbscan-grid-pca{pca_dimensions}"
        / f"d{dimension}"
        / f"mcs{mcs}-ms{ms}-{method}"
    )


def run_grid_cell(
    embedding_path: str,
    pca_dimensions: int,
    dimension: int,
    mcs: int,
    ms: int,
    method: str,
    force: bool,
) -> dict:
    directory = grid_directory(pca_dimensions, dimension, mcs, ms, method)
    meta_path = directory / "run.json"
    labels_path = directory / "labels.npy"
    if not force and meta_path.exists() and labels_path.exists():
        result = json.loads(meta_path.read_text(encoding="utf-8"))
        result["cached"] = True
        return result
    embedding = np.load(embedding_path, mmap_mode="r")
    _, labels, summary = fit_hdbscan(embedding, mcs, ms, method)
    atomic_npy(labels_path, labels)
    result = {
        "completed_at": utc_now(),
        "pca_dimensions": pca_dimensions,
        "dimension": dimension,
        "seed": 42,
        "min_cluster_size": mcs,
        "min_samples": ms,
        "selection_method": method,
        "embedding_path": str(Path(embedding_path).resolve().relative_to(ROOT)),
        "embedding_sha256": sha256_file(Path(embedding_path)),
        "labels_sha256": sha256_file(labels_path),
        **summary,
    }
    atomic_json(meta_path, result)
    result["cached"] = False
    return result


def command_hdbscan_grid(args: argparse.Namespace) -> None:
    config = load_config()
    pca_dimensions = int(args.pca_dimensions)
    if pca_dimensions == 100:
        dimensions = [int(value) for value in config["umap_primary"]["dimensions"]]
    elif pca_dimensions == 50:
        dimensions = [int(value) for value in config["pca_sensitivity"]["umap_dimensions"]]
    else:
        raise ValueError("Unsupported PCA dimension")
    grid = config["hdbscan_grid"]
    cells = []
    for dimension in dimensions:
        canonical_root = SOURCE / ("canonical" if pca_dimensions == 100 else "canonical-pca50")
        canonical_path = canonical_root / f"umap{dimension}.npy"
        embedding_path = (
            canonical_path
            if canonical_path.exists()
            else umap_run_directory(pca_dimensions, dimension, 42) / "embedding.npy"
        )
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing seed-42 embedding: {embedding_path}")
        for mcs in grid["min_cluster_size"]:
            for ms in grid["min_samples"]:
                cells.append((str(embedding_path), dimension, int(mcs), int(ms), "eom"))
        cells.append((str(embedding_path), dimension, 200, 15, "leaf"))
    print(f"Running/checking {len(cells)} HDBSCAN grid cells with {args.jobs} workers", flush=True)
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(run_grid_cell)(
            path,
            pca_dimensions,
            dimension,
            mcs,
            ms,
            method,
            args.force,
        )
        for path, dimension, mcs, ms, method in cells
    )
    cached = sum(bool(row["cached"]) for row in rows)
    print(f"Complete: {len(rows) - cached} fitted, {cached} cached", flush=True)


def command_summarize_grid(args: argparse.Namespace) -> None:
    config = load_config()
    pca_dimensions = int(args.pca_dimensions)
    dimensions = (
        [int(value) for value in config["umap_primary"]["dimensions"]]
        if pca_dimensions == 100
        else [int(value) for value in config["pca_sensitivity"]["umap_dimensions"]]
    )
    grid = config["hdbscan_grid"]
    rows: list[dict] = []
    labels_by_key: dict[tuple[int, int, int, str], np.ndarray] = {}
    for dimension in dimensions:
        for method in ("eom", "leaf"):
            combinations = (
                [(int(mcs), int(ms)) for mcs in grid["min_cluster_size"] for ms in grid["min_samples"]]
                if method == "eom"
                else [(200, 15)]
            )
            for mcs, ms in combinations:
                directory = grid_directory(pca_dimensions, dimension, mcs, ms, method)
                if not (directory / "run.json").exists():
                    raise FileNotFoundError(f"Missing grid cell: {directory}")
                row = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                rows.append(row)
                labels_by_key[(dimension, mcs, ms, method)] = np.load(directory / "labels.npy")

    mcs_values = [int(value) for value in grid["min_cluster_size"]]
    ms_values = [int(value) for value in grid["min_samples"]]
    for row in rows:
        if row["selection_method"] != "eom":
            row["adjacent_parameter_ari_mean"] = ""
            row["adjacent_parameter_ari_min"] = ""
            continue
        dimension = int(row["dimension"])
        mcs = int(row["min_cluster_size"])
        ms = int(row["min_samples"])
        neighbors = []
        mcs_position = mcs_values.index(mcs)
        ms_position = ms_values.index(ms)
        for delta_mcs, delta_ms in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_mcs_position = mcs_position + delta_mcs
            next_ms_position = ms_position + delta_ms
            if 0 <= next_mcs_position < len(mcs_values) and 0 <= next_ms_position < len(ms_values):
                other_key = (
                    dimension,
                    mcs_values[next_mcs_position],
                    ms_values[next_ms_position],
                    "eom",
                )
                current = labels_by_key[(dimension, mcs, ms, "eom")]
                neighbors.append(float(adjusted_rand_score(current, labels_by_key[other_key])))
        row["adjacent_parameter_ari_mean"] = float(np.mean(neighbors))
        row["adjacent_parameter_ari_min"] = float(np.min(neighbors))

    for dimension in dimensions:
        eligible = [
            row
            for row in rows
            if row["dimension"] == dimension
            and row["selection_method"] == "eom"
            and int(grid["substantive_min_clusters"]) <= row["cluster_count"] <= int(grid["substantive_max_clusters"])
        ]
        flags = pareto_flags(
            eligible,
            {
                "relative_validity": "max",
                "coverage": "max",
                "persistence_weighted_coverage": "max",
                "adjacent_parameter_ari_mean": "max",
            },
        )
        for row, flag in zip(eligible, flags, strict=True):
            row["pareto_within_dimension"] = flag
    for row in rows:
        row.setdefault("pareto_within_dimension", False)
        row["substantive_non_degenerate"] = bool(
            int(grid["substantive_min_clusters"])
            <= row["cluster_count"]
            <= int(grid["substantive_max_clusters"])
        )
    rows.sort(
        key=lambda row: (
            int(row["dimension"]),
            row["selection_method"],
            int(row["min_cluster_size"]),
            int(row["min_samples"]),
        )
    )
    write_csv(RESULTS / f"pca{pca_dimensions}-hdbscan-grid.csv", rows)
    summary = []
    for dimension in dimensions:
        candidates = [
            row
            for row in rows
            if row["dimension"] == dimension and row["selection_method"] == "eom"
        ]
        nondegenerate = [row for row in candidates if row["substantive_non_degenerate"]]
        best_dbcv = max(nondegenerate, key=lambda row: row["relative_validity"])
        pareto_count = sum(bool(row["pareto_within_dimension"]) for row in nondegenerate)
        summary.append(
            {
                "pca_dimensions": pca_dimensions,
                "dimension": dimension,
                "grid_cells": len(candidates),
                "nondegenerate_cells": len(nondegenerate),
                "pareto_cells": pareto_count,
                "best_dbcv_mcs": best_dbcv["min_cluster_size"],
                "best_dbcv_ms": best_dbcv["min_samples"],
                "best_dbcv": best_dbcv["relative_validity"],
                "best_dbcv_clusters": best_dbcv["cluster_count"],
                "best_dbcv_noise_pct": best_dbcv["noise_pct"],
            }
        )
    write_csv(RESULTS / f"pca{pca_dimensions}-hdbscan-grid-summary.csv", summary)
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Verify sources, order, lineage, and exact reproduction")
    audit.set_defaults(function=command_audit)

    seeds = subparsers.add_parser("umap-seeds", help="Run the preregistered UMAP seed matrix")
    seeds.add_argument("--pca-dimensions", type=int, choices=(50, 100), default=100)
    seeds.add_argument("--jobs", type=int, default=3)
    seeds.add_argument("--force", action="store_true")
    seeds.set_defaults(function=command_umap_seeds)

    summarize_seeds = subparsers.add_parser("summarize-seeds", help="Summarize completed seed runs")
    summarize_seeds.add_argument("--pca-dimensions", type=int, choices=(50, 100), default=100)
    summarize_seeds.set_defaults(function=command_summarize_seeds)

    grid = subparsers.add_parser("hdbscan-grid", help="Run the preregistered HDBSCAN grid")
    grid.add_argument("--pca-dimensions", type=int, choices=(50, 100), default=100)
    grid.add_argument("--jobs", type=int, default=3)
    grid.add_argument("--force", action="store_true")
    grid.set_defaults(function=command_hdbscan_grid)

    summarize_grid = subparsers.add_parser("summarize-grid", help="Summarize completed HDBSCAN grid runs")
    summarize_grid.add_argument("--pca-dimensions", type=int, choices=(50, 100), default=100)
    summarize_grid.set_defaults(function=command_summarize_grid)
    return parser


def main() -> None:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
