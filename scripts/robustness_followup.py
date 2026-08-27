#!/usr/bin/env python3
"""Parameter-sensitivity and resampling stages for the robustness study."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

from robustness_metrics import (
    NOISE,
    atomic_json,
    atomic_npy,
    best_match_jaccard,
    hdbscan_summary,
    pairwise_partition_rows,
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
CONFIG = yaml.safe_load((STUDY / "config.yaml").read_text(encoding="utf-8"))
FINALISTS = yaml.safe_load((STUDY / "finalists.yaml").read_text(encoding="utf-8"))["finalists"]
N_PAPERS = 69_400


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(sorted(set().union(*(row.keys() for row in rows)) - set(fields)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def md_tag(value: float) -> str:
    return str(value).replace(".", "p")


def fit_hdbscan(
    coordinates: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[np.ndarray, dict]:
    import hdbscan

    started = perf_counter()
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method="eom",
        gen_min_span_tree=True,
        prediction_data=False,
        core_dist_n_jobs=1,
    ).fit(coordinates)
    labels = model.labels_.astype(np.int16, copy=False)
    return labels, hdbscan_summary(model, labels, perf_counter() - started)


def unique_umap_hdbscan_configs() -> dict[int, list[tuple[int, int]]]:
    configs: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for finalist in FINALISTS:
        if finalist["kind"] == "umap":
            configs[int(finalist["umap_dimensions"])].add(
                (int(finalist["min_cluster_size"]), int(finalist["min_samples"]))
            )
    return {dimension: sorted(values) for dimension, values in configs.items()}


def sensitivity_directory(dimension: int, neighbors: int, min_dist: float, seed: int) -> Path:
    return (
        WORK
        / "runs"
        / "umap-sensitivity"
        / f"d{dimension}"
        / f"nn{neighbors}-md{md_tag(min_dist)}"
        / f"seed{seed}"
    )


def run_sensitivity_cell(
    dimension: int,
    neighbors: int,
    min_dist: float,
    seed: int,
    hdbscan_configs: list[tuple[int, int]],
    force: bool,
) -> dict:
    import umap

    directory = sensitivity_directory(dimension, neighbors, min_dist, seed)
    meta_path = directory / "run.json"
    embedding_path = directory / "embedding.npy"
    expected_labels = [directory / f"labels-mcs{mcs}-ms{ms}.npy" for mcs, ms in hdbscan_configs]
    if not force and meta_path.exists() and embedding_path.exists() and all(path.exists() for path in expected_labels):
        row = json.loads(meta_path.read_text(encoding="utf-8"))
        row["cached"] = True
        return row
    pca = np.load(SOURCE / "canonical" / "pca100.npy", mmap_mode="r")
    started = perf_counter()
    embedding = umap.UMAP(
        n_components=dimension,
        n_neighbors=neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        low_memory=False,
        verbose=False,
    ).fit_transform(pca).astype(np.float32, copy=False)
    umap_seconds = perf_counter() - started
    if embedding.shape != (N_PAPERS, dimension) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid UMAP sensitivity output: {embedding.shape}")
    atomic_npy(embedding_path, embedding)
    fits = []
    for mcs, ms in hdbscan_configs:
        labels, summary = fit_hdbscan(embedding, mcs, ms)
        labels_path = directory / f"labels-mcs{mcs}-ms{ms}.npy"
        atomic_npy(labels_path, labels)
        fits.append(
            {
                "min_cluster_size": mcs,
                "min_samples": ms,
                "labels_sha256": sha256_file(labels_path),
                **summary,
            }
        )
    row = {
        "dimension": dimension,
        "n_neighbors": neighbors,
        "min_dist": min_dist,
        "seed": seed,
        "umap_seconds": umap_seconds,
        "embedding_sha256": sha256_file(embedding_path),
        "fits": fits,
    }
    atomic_json(meta_path, row)
    row["cached"] = False
    return row


def command_sensitivity(args: argparse.Namespace) -> None:
    settings = CONFIG["umap_parameter_sensitivity"]
    hdbscan_by_dimension = unique_umap_hdbscan_configs()
    cells = [
        (dimension, int(neighbors), float(min_dist), int(seed), hdbscan_configs)
        for dimension, hdbscan_configs in sorted(hdbscan_by_dimension.items())
        for neighbors in settings["n_neighbors"]
        for min_dist in settings["min_dist"]
        for seed in settings["seeds"]
    ]
    print(f"Running/checking {len(cells)} UMAP sensitivity cells", flush=True)
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(run_sensitivity_cell)(*cell, args.force) for cell in cells
    )
    print(
        f"Complete: {sum(not row['cached'] for row in rows)} fitted, "
        f"{sum(bool(row['cached']) for row in rows)} cached",
        flush=True,
    )


def command_summarize_sensitivity(_: argparse.Namespace) -> None:
    settings = CONFIG["umap_parameter_sensitivity"]
    hdbscan_by_dimension = unique_umap_hdbscan_configs()
    run_rows: list[dict] = []
    cell_rows: list[dict] = []
    medoid_labels: dict[tuple[int, int, float, int, int], np.ndarray] = {}
    for dimension, configs in sorted(hdbscan_by_dimension.items()):
        for neighbors in settings["n_neighbors"]:
            for min_dist in settings["min_dist"]:
                for mcs, ms in configs:
                    names, labels, metadata = [], [], []
                    for seed in settings["seeds"]:
                        directory = sensitivity_directory(
                            dimension, int(neighbors), float(min_dist), int(seed)
                        )
                        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                        fit = next(
                            value
                            for value in run["fits"]
                            if value["min_cluster_size"] == mcs and value["min_samples"] == ms
                        )
                        row = {
                            "dimension": dimension,
                            "n_neighbors": int(neighbors),
                            "min_dist": float(min_dist),
                            "seed": int(seed),
                            "min_cluster_size": mcs,
                            "min_samples": ms,
                            **fit,
                        }
                        run_rows.append(row)
                        metadata.append(row)
                        names.append(f"seed{seed}")
                        labels.append(np.load(directory / f"labels-mcs{mcs}-ms{ms}.npy"))
                    comparisons = pairwise_partition_rows(names, labels)
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
                    medoid_labels[(dimension, int(neighbors), float(min_dist), mcs, ms)] = labels[
                        names.index(medoid_name)
                    ]
                    cell_rows.append(
                        {
                            "dimension": dimension,
                            "n_neighbors": int(neighbors),
                            "min_dist": float(min_dist),
                            "min_cluster_size": mcs,
                            "min_samples": ms,
                            "medoid_seed": int(medoid_name.removeprefix("seed")),
                            "cluster_count_median": float(
                                np.median([row["cluster_count"] for row in metadata])
                            ),
                            "cluster_count_min": min(row["cluster_count"] for row in metadata),
                            "cluster_count_max": max(row["cluster_count"] for row in metadata),
                            "noise_pct_median": float(np.median([row["noise_pct"] for row in metadata])),
                            "relative_validity_median": float(
                                np.median([row["relative_validity"] for row in metadata])
                            ),
                            "pairwise_seed_ari_mean": float(
                                np.mean([row["ari_all"] for row in comparisons])
                            ),
                            "pairwise_seed_ari_min": float(
                                np.min([row["ari_all"] for row in comparisons])
                            ),
                            "pairwise_seed_cluster_jaccard_weighted_mean": float(
                                np.mean([row["cluster_jaccard_weighted"] for row in comparisons])
                            ),
                        }
                    )

    parameter_rows = []
    for dimension, configs in sorted(hdbscan_by_dimension.items()):
        for mcs, ms in configs:
            keys = [
                key
                for key in medoid_labels
                if key[0] == dimension and key[3] == mcs and key[4] == ms
            ]
            names = [f"nn{key[1]}-md{key[2]}" for key in keys]
            comparisons = pairwise_partition_rows(names, [medoid_labels[key] for key in keys])
            for row in comparisons:
                row.update(
                    {
                        "dimension": dimension,
                        "min_cluster_size": mcs,
                        "min_samples": ms,
                    }
                )
            parameter_rows.extend(comparisons)
    write_csv(RESULTS / "umap-sensitivity-runs.csv", run_rows)
    write_csv(RESULTS / "umap-sensitivity-cells.csv", cell_rows)
    write_csv(RESULTS / "umap-sensitivity-pairwise-parameters.csv", parameter_rows)
    print(json.dumps(cell_rows, indent=2))


def resample_directory(dimension: int, repeat: int) -> Path:
    return WORK / "runs" / "resampling" / f"d{dimension}" / f"repeat{repeat:02d}"


def resample_indices(repeat: int) -> np.ndarray:
    settings = CONFIG["resampling"]
    rng = np.random.default_rng(int(settings["seed_start"]) + repeat)
    count = int(round(N_PAPERS * float(settings["fraction"])))
    return np.sort(rng.choice(N_PAPERS, size=count, replace=False)).astype(np.int32)


def run_resample_umap(
    dimension: int,
    repeat: int,
    configs: list[tuple[int, int]],
    force: bool,
) -> dict:
    import umap

    directory = resample_directory(dimension, repeat)
    meta_path = directory / "run.json"
    embedding_path = directory / "embedding.npy"
    indices_path = directory / "indices.npy"
    expected = [
        directory / f"labels-{strategy}-mcs{mcs}-ms{ms}.npy"
        for mcs, ms in configs
        for strategy in ("scaled", "fixed")
    ]
    if not force and meta_path.exists() and embedding_path.exists() and indices_path.exists() and all(
        path.exists() for path in expected
    ):
        row = json.loads(meta_path.read_text(encoding="utf-8"))
        row["cached"] = True
        return row
    indices = resample_indices(repeat)
    pca = np.load(SOURCE / "canonical" / "pca100.npy", mmap_mode="r")
    seed = int(CONFIG["resampling"]["seed_start"]) + repeat
    started = perf_counter()
    embedding = umap.UMAP(
        n_components=dimension,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
        low_memory=False,
        verbose=False,
    ).fit_transform(pca[indices]).astype(np.float32, copy=False)
    umap_seconds = perf_counter() - started
    if embedding.shape != (len(indices), dimension) or not np.isfinite(embedding).all():
        raise ValueError(f"Invalid resample UMAP output: {embedding.shape}")
    atomic_npy(indices_path, indices)
    atomic_npy(embedding_path, embedding)
    fits = []
    fraction = float(CONFIG["resampling"]["fraction"])
    for mcs, ms in configs:
        for strategy, fitted_mcs in (("scaled", max(2, round(mcs * fraction))), ("fixed", mcs)):
            labels, summary = fit_hdbscan(embedding, fitted_mcs, ms)
            labels_path = directory / f"labels-{strategy}-mcs{mcs}-ms{ms}.npy"
            atomic_npy(labels_path, labels)
            fits.append(
                {
                    "base_min_cluster_size": mcs,
                    "fitted_min_cluster_size": fitted_mcs,
                    "min_samples": ms,
                    "strategy": strategy,
                    **summary,
                }
            )
    row = {
        "dimension": dimension,
        "repeat": repeat,
        "seed": seed,
        "sample_count": len(indices),
        "sample_fraction": fraction,
        "umap_seconds": umap_seconds,
        "indices_sha256": sha256_file(indices_path),
        "embedding_sha256": sha256_file(embedding_path),
        "fits": fits,
    }
    atomic_json(meta_path, row)
    row["cached"] = False
    return row


def run_resample_fixed_display(repeat: int, force: bool) -> dict:
    from sklearn.cluster import HDBSCAN

    finalist = next(value for value in FINALISTS if value["kind"] == "fixed_display")
    directory = WORK / "runs" / "resampling-fixed-display" / f"repeat{repeat:02d}"
    meta_path = directory / "run.json"
    indices_path = directory / "indices.npy"
    expected = [directory / f"labels-{strategy}.npy" for strategy in ("scaled", "fixed")]
    if not force and meta_path.exists() and indices_path.exists() and all(path.exists() for path in expected):
        row = json.loads(meta_path.read_text(encoding="utf-8"))
        row["cached"] = True
        return row
    indices = resample_indices(repeat)
    coordinates = np.load(ROOT / finalist["reference_embedding"])[indices]
    fraction = float(CONFIG["resampling"]["fraction"])
    fits = []
    for strategy, mcs in (
        ("scaled", max(2, round(int(finalist["min_cluster_size"]) * fraction))),
        ("fixed", int(finalist["min_cluster_size"])),
    ):
        started = perf_counter()
        model = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=int(finalist["min_samples"]),
            metric="euclidean",
            cluster_selection_method="eom",
            algorithm="kd_tree",
            leaf_size=60,
            n_jobs=1,
            copy=True,
        ).fit(coordinates)
        labels = model.labels_.astype(np.int16, copy=False)
        labels_path = directory / f"labels-{strategy}.npy"
        atomic_npy(labels_path, labels)
        assigned = labels != NOISE
        fits.append(
            {
                "strategy": strategy,
                "fitted_min_cluster_size": mcs,
                "min_samples": int(finalist["min_samples"]),
                "cluster_count": int(np.unique(labels[assigned]).size),
                "noise_pct": float((~assigned).mean() * 100),
                "coverage": float(assigned.mean()),
                "membership_mean_assigned": float(model.probabilities_[assigned].mean()),
                "fit_seconds": perf_counter() - started,
            }
        )
    atomic_npy(indices_path, indices)
    row = {
        "repeat": repeat,
        "seed": int(CONFIG["resampling"]["seed_start"]) + repeat,
        "sample_count": len(indices),
        "sample_fraction": fraction,
        "conditional_on_fixed_projection": True,
        "fits": fits,
    }
    atomic_json(meta_path, row)
    row["cached"] = False
    return row


def command_resample(args: argparse.Namespace) -> None:
    configs = unique_umap_hdbscan_configs()
    repeats = range(int(CONFIG["resampling"]["repeats"]))
    jobs = [(dimension, repeat, values) for dimension, values in sorted(configs.items()) for repeat in repeats]
    print(f"Running/checking {len(jobs)} full-path UMAP resamples", flush=True)
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(run_resample_umap)(dimension, repeat, values, args.force)
        for dimension, repeat, values in jobs
    )
    fixed = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(run_resample_fixed_display)(repeat, args.force) for repeat in repeats
    )
    print(
        f"Complete: {sum(not row['cached'] for row in rows) + sum(not row['cached'] for row in fixed)} fitted, "
        f"{sum(bool(row['cached']) for row in rows) + sum(bool(row['cached']) for row in fixed)} cached",
        flush=True,
    )


def command_summarize_resample(_: argparse.Namespace) -> None:
    repeats = range(int(CONFIG["resampling"]["repeats"]))
    run_rows: list[dict] = []
    cluster_rows: list[dict] = []
    summary_rows: list[dict] = []
    for finalist in FINALISTS:
        reference = np.load(ROOT / finalist["reference_labels"])
        if reference.shape != (N_PAPERS,):
            raise ValueError(f"Invalid reference labels for {finalist['id']}: {reference.shape}")
        for strategy in ("scaled", "fixed"):
            per_cluster: dict[int, list[float]] = defaultdict(list)
            finalist_rows = []
            for repeat in repeats:
                if finalist["kind"] == "fixed_display":
                    directory = WORK / "runs" / "resampling-fixed-display" / f"repeat{repeat:02d}"
                    labels_path = directory / f"labels-{strategy}.npy"
                else:
                    directory = resample_directory(int(finalist["umap_dimensions"]), repeat)
                    labels_path = directory / (
                        f"labels-{strategy}-mcs{finalist['min_cluster_size']}-ms{finalist['min_samples']}.npy"
                    )
                indices = np.load(directory / "indices.npy")
                candidate = np.load(labels_path)
                comparison = partition_comparison(reference[indices], candidate)
                matches = best_match_jaccard(reference[indices], candidate)
                for match in matches:
                    per_cluster[int(match["reference_cluster"])].append(float(match["jaccard"]))
                row = {
                    "finalist": finalist["id"],
                    "strategy": strategy,
                    "repeat": repeat,
                    "pca_refitted": False,
                    "conditional_on_fixed_projection": finalist["kind"] == "fixed_display",
                    "candidate_cluster_count": int(np.unique(candidate[candidate != NOISE]).size),
                    "candidate_noise_pct": float(np.mean(candidate == NOISE) * 100),
                    **comparison,
                }
                run_rows.append(row)
                finalist_rows.append(row)
            for cluster, values in sorted(per_cluster.items()):
                low, high = percentile_interval(values)
                mean = float(np.mean(values))
                cluster_rows.append(
                    {
                        "finalist": finalist["id"],
                        "strategy": strategy,
                        "cluster": cluster,
                        "pca_refitted": False,
                        "full_size": int(np.count_nonzero(reference == cluster)),
                        "subsample_jaccard_mean": mean,
                        "subsample_jaccard_median": float(np.median(values)),
                        "subsample_jaccard_min": float(np.min(values)),
                        "subsample_jaccard_ci_low": low,
                        "subsample_jaccard_ci_high": high,
                        "stability_band": stability_band(mean),
                    }
                )
            summary_rows.append(
                {
                    "finalist": finalist["id"],
                    "strategy": strategy,
                    "pca_refitted": False,
                    "conditional_on_fixed_projection": finalist["kind"] == "fixed_display",
                    "runs": len(finalist_rows),
                    "candidate_cluster_count_median": float(
                        np.median([row["candidate_cluster_count"] for row in finalist_rows])
                    ),
                    "candidate_cluster_count_min": min(
                        row["candidate_cluster_count"] for row in finalist_rows
                    ),
                    "candidate_cluster_count_max": max(
                        row["candidate_cluster_count"] for row in finalist_rows
                    ),
                    "degenerate_runs_lt5_clusters": int(
                        sum(row["candidate_cluster_count"] < 5 for row in finalist_rows)
                    ),
                    "candidate_noise_pct_median": float(
                        np.median([row["candidate_noise_pct"] for row in finalist_rows])
                    ),
                    "ari_all_mean": float(np.mean([row["ari_all"] for row in finalist_rows])),
                    "ari_all_median": float(np.median([row["ari_all"] for row in finalist_rows])),
                    "ari_all_q10": float(np.quantile([row["ari_all"] for row in finalist_rows], 0.10)),
                    "ari_all_min": float(np.min([row["ari_all"] for row in finalist_rows])),
                    "cluster_jaccard_weighted_mean": float(
                        np.mean([row["cluster_jaccard_weighted"] for row in finalist_rows])
                    ),
                    "cluster_jaccard_weighted_median": float(
                        np.median([row["cluster_jaccard_weighted"] for row in finalist_rows])
                    ),
                    "cluster_jaccard_weighted_q10": float(
                        np.quantile([row["cluster_jaccard_weighted"] for row in finalist_rows], 0.10)
                    ),
                    "mapped_assignment_agreement_mean": float(
                        np.mean([row["mapped_assignment_agreement"] for row in finalist_rows])
                    ),
                    "noise_agreement_mean": float(
                        np.mean([row["noise_agreement"] for row in finalist_rows])
                    ),
                }
            )
    write_csv(RESULTS / "resampling-runs.csv", run_rows)
    write_csv(RESULTS / "resampling-cluster-stability.csv", cluster_rows)
    write_csv(RESULTS / "resampling-summary.csv", summary_rows)
    print(json.dumps(summary_rows, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sensitivity = subparsers.add_parser("sensitivity")
    sensitivity.add_argument("--jobs", type=int, default=3)
    sensitivity.add_argument("--force", action="store_true")
    sensitivity.set_defaults(function=command_sensitivity)
    summarize_sensitivity = subparsers.add_parser("summarize-sensitivity")
    summarize_sensitivity.set_defaults(function=command_summarize_sensitivity)
    resample = subparsers.add_parser("resample")
    resample.add_argument("--jobs", type=int, default=3)
    resample.add_argument("--force", action="store_true")
    resample.set_defaults(function=command_resample)
    summarize_resample = subparsers.add_parser("summarize-resample")
    summarize_resample.set_defaults(function=command_summarize_resample)
    return parser


def main() -> None:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
