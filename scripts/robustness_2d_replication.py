#!/usr/bin/env python3
"""Replicate the historical 2D HDBSCAN settings on independent UMAP fits."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN

from robustness_metrics import atomic_json, atomic_npy, pairwise_partition_rows, partition_comparison


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
WORK = STUDY / "work"
RESULTS = STUDY / "results"
SEEDS = [11, 23, 42, 73, 101, 211, 307, 419, 547, 809]
LINUX_SEEDS = [11, 42, 73, 211, 547]
MCS = 150
MS = 50
FRACTION = 0.8


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def fit(coordinates: np.ndarray, min_cluster_size: int) -> tuple[np.ndarray, dict]:
    started = perf_counter()
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MS,
        metric="euclidean",
        cluster_selection_method="eom",
        algorithm="kd_tree",
        leaf_size=60,
        n_jobs=1,
        copy=True,
    ).fit(coordinates)
    labels = model.labels_.astype(np.int16, copy=False)
    assigned = labels != -1
    return labels, {
        "cluster_count": int(np.unique(labels[assigned]).size),
        "noise_pct": float(np.mean(~assigned) * 100),
        "coverage": float(np.mean(assigned)),
        "membership_mean_assigned": float(model.probabilities_[assigned].mean()),
        "fit_seconds": perf_counter() - started,
    }


def seed_directory(platform: str, seed: int) -> Path:
    if platform == "windows":
        return WORK / "runs" / "umap-pca100" / "d2" / f"seed{seed}"
    return WORK / "remote-linux" / "seed-runs" / "d2" / f"seed{seed}"


def fit_seed(platform: str, seed: int, force: bool) -> dict:
    directory = seed_directory(platform, seed)
    output = directory / "labels-sklearn-mcs150-ms50.npy"
    metadata = directory / "sklearn-mcs150-ms50.json"
    if output.exists() and metadata.exists() and not force:
        return json.loads(metadata.read_text(encoding="utf-8"))
    coordinates = np.load(directory / "embedding.npy", mmap_mode="r")
    labels, metrics = fit(coordinates, MCS)
    atomic_npy(output, labels)
    row = {"platform": platform, "seed": seed, "min_cluster_size": MCS, "min_samples": MS, **metrics}
    atomic_json(metadata, row)
    return row


def command_seeds(args: argparse.Namespace) -> None:
    jobs = [("windows", seed) for seed in SEEDS] + [("linux", seed) for seed in LINUX_SEEDS]
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(fit_seed)(platform, seed, args.force) for platform, seed in jobs
    )
    print(json.dumps(rows, indent=2))


def command_summarize(_: argparse.Namespace) -> None:
    historical = np.load(ROOT / "analysis" / "hdbscan-2d-labels.npy")
    rows, names, labels = [], [], []
    for platform, seeds in (("windows", SEEDS), ("linux", LINUX_SEEDS)):
        for seed in seeds:
            directory = seed_directory(platform, seed)
            row = json.loads((directory / "sklearn-mcs150-ms50.json").read_text(encoding="utf-8"))
            candidate = np.load(directory / "labels-sklearn-mcs150-ms50.npy")
            row.update({f"historical_{key}": value for key, value in partition_comparison(historical, candidate).items()})
            rows.append(row)
            names.append(f"{platform}-seed{seed}")
            labels.append(candidate)
    pairwise = pairwise_partition_rows(names, labels)
    mean_ari = {
        name: float(np.mean([row["ari_all"] for row in pairwise if row["left"] == name or row["right"] == name]))
        for name in names
    }
    medoid = max(mean_ari, key=mean_ari.get)
    summaries = []
    for platform in ("windows", "linux", "combined"):
        selected = rows if platform == "combined" else [row for row in rows if row["platform"] == platform]
        selected_names = set(names if platform == "combined" else [f"{platform}-seed{row['seed']}" for row in selected])
        comparisons = [row for row in pairwise if row["left"] in selected_names and row["right"] in selected_names]
        summaries.append(
            {
                "platform": platform,
                "runs": len(selected),
                "cluster_count_median": float(np.median([row["cluster_count"] for row in selected])),
                "cluster_count_min": min(row["cluster_count"] for row in selected),
                "cluster_count_max": max(row["cluster_count"] for row in selected),
                "noise_pct_median": float(np.median([row["noise_pct"] for row in selected])),
                "pairwise_ari_mean": float(np.mean([row["ari_all"] for row in comparisons])),
                "pairwise_ari_min": float(np.min([row["ari_all"] for row in comparisons])),
                "cluster_jaccard_weighted_mean": float(np.mean([row["cluster_jaccard_weighted"] for row in comparisons])),
            }
        )
    medoid_platform, medoid_seed = medoid.split("-seed")
    selected = {
        "medoid": medoid,
        "platform": medoid_platform,
        "seed": int(medoid_seed),
        "mean_ari_to_other_runs": mean_ari[medoid],
        "embedding": str(seed_directory(medoid_platform, int(medoid_seed)) / "embedding.npy"),
        "labels": str(seed_directory(medoid_platform, int(medoid_seed)) / "labels-sklearn-mcs150-ms50.npy"),
    }
    write_csv(RESULTS / "direct2-historical-params-runs.csv", rows)
    write_csv(RESULTS / "direct2-historical-params-pairwise.csv", pairwise)
    write_csv(RESULTS / "direct2-historical-params-summary.csv", summaries)
    atomic_json(RESULTS / "direct2-historical-params-selected.json", selected)
    print(json.dumps({"summary": summaries, "selected": selected}, indent=2))


def fit_resample(repeat: int, force: bool) -> dict:
    directory = WORK / "runs" / "resampling" / "d2" / f"repeat{repeat:02d}"
    rows = []
    coordinates = np.load(directory / "embedding.npy", mmap_mode="r")
    for strategy, fitted_mcs in (("scaled", round(MCS * FRACTION)), ("fixed", MCS)):
        output = directory / f"labels-{strategy}-mcs{MCS}-ms{MS}.npy"
        if output.exists() and not force:
            labels = np.load(output)
            assigned = labels != -1
            rows.append(
                {
                    "repeat": repeat,
                    "strategy": strategy,
                    "cluster_count": int(np.unique(labels[assigned]).size),
                    "noise_pct": float(np.mean(~assigned) * 100),
                    "cached": True,
                }
            )
            continue
        labels, metrics = fit(coordinates, fitted_mcs)
        atomic_npy(output, labels)
        rows.append({"repeat": repeat, "strategy": strategy, **metrics, "cached": False})
    return {"repeat": repeat, "fits": rows}


def command_resamples(args: argparse.Namespace) -> None:
    rows = Parallel(n_jobs=args.jobs, backend="loky", verbose=10)(
        delayed(fit_resample)(repeat, args.force) for repeat in range(20)
    )
    print(json.dumps(rows, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    seeds = commands.add_parser("seeds")
    seeds.add_argument("--jobs", type=int, default=3)
    seeds.add_argument("--force", action="store_true")
    seeds.set_defaults(function=command_seeds)
    summarize = commands.add_parser("summarize")
    summarize.set_defaults(function=command_summarize)
    resamples = commands.add_parser("resamples")
    resamples.add_argument("--jobs", type=int, default=3)
    resamples.add_argument("--force", action="store_true")
    resamples.set_defaults(function=command_resamples)
    return root


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
