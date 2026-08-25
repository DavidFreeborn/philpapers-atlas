#!/usr/bin/env python3
"""Scan HDBSCAN parameters on the fixed two-dimensional display projection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score


MIN_CLUSTER_SIZES = (75, 100, 150, 200, 300, 500)
MIN_SAMPLES = (5, 10, 15, 25, 50)
SELECTION_METHODS = ("eom", "leaf")
RANDOM_STATE = 20260825
SILHOUETTE_SAMPLE = 4_000
MAX_SELECTED_CLUSTERS = 200


def load_coordinates(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    coordinates = np.asarray(data["points"], dtype=np.float32)[:, :2]
    if coordinates.shape != (69_400, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"Unexpected display coordinates: {coordinates.shape}")
    return coordinates


def fit_configuration(
    coordinates: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    selection_method: str,
) -> tuple[dict, np.ndarray]:
    started = perf_counter()
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=selection_method,
        algorithm="kd_tree",
        leaf_size=60,
        n_jobs=1,
        copy=True,
    ).fit(coordinates)
    labels = model.labels_.astype(np.int16, copy=False)
    assigned = labels >= 0
    cluster_count = int(np.unique(labels[assigned]).size)
    silhouette = (
        float(
            silhouette_score(
                coordinates[assigned],
                labels[assigned],
                metric="euclidean",
                sample_size=min(SILHOUETTE_SAMPLE, int(assigned.sum())),
                random_state=RANDOM_STATE,
            )
        )
        if cluster_count > 1
        else -1.0
    )
    result = {
        "minClusterSize": min_cluster_size,
        "minSamples": min_samples,
        "selectionMethod": selection_method,
        "clusterCount": cluster_count,
        "noisePct": float((~assigned).mean() * 100),
        "coverage": float(assigned.mean()),
        "meanMembership": float(model.probabilities_[assigned].mean()),
        "medianMembership": float(np.median(model.probabilities_[assigned])),
        "silhouette2d": silhouette,
        "fitSeconds": perf_counter() - started,
    }
    return result, labels


def normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    span = float(values.max() - minimum)
    return np.ones_like(values) if span == 0 else (values - minimum) / span


def score_results(results: list[dict], labels: list[np.ndarray]) -> dict:
    by_key = {
        (row["selectionMethod"], row["minClusterSize"], row["minSamples"]): index
        for index, row in enumerate(results)
    }
    for index, row in enumerate(results):
        neighbours = []
        method = row["selectionMethod"]
        size_index = MIN_CLUSTER_SIZES.index(row["minClusterSize"])
        samples_index = MIN_SAMPLES.index(row["minSamples"])
        for delta_size, delta_samples in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_size = size_index + delta_size
            next_samples = samples_index + delta_samples
            if 0 <= next_size < len(MIN_CLUSTER_SIZES) and 0 <= next_samples < len(MIN_SAMPLES):
                neighbour_index = by_key[
                    (method, MIN_CLUSTER_SIZES[next_size], MIN_SAMPLES[next_samples])
                ]
                neighbours.append(adjusted_rand_score(labels[index], labels[neighbour_index]))
        row["parameterAgreementAri"] = float(np.mean(neighbours))

    eligible = [row for row in results if 2 <= row["clusterCount"] <= MAX_SELECTED_CLUSTERS]
    if not eligible:
        raise ValueError("No scan configuration produced a usable number of clusters")
    membership = normalise(np.array([row["meanMembership"] for row in eligible]))
    silhouette = normalise(np.array([row["silhouette2d"] for row in eligible]))
    agreement = normalise(np.array([row["parameterAgreementAri"] for row in eligible]))
    coverage = normalise(np.array([row["coverage"] for row in eligible]))
    for index, row in enumerate(eligible):
        row["balancedScore"] = float(
            0.35 * agreement[index]
            + 0.30 * membership[index]
            + 0.20 * silhouette[index]
            + 0.15 * coverage[index]
        )
    selected = max(eligible, key=lambda row: row["balancedScore"])
    for row in results:
        row.setdefault("balancedScore", -1.0)
        row["selected"] = row is selected
    return selected


def write_results(
    output_directory: Path,
    results: list[dict],
    selected: dict,
    selected_labels: np.ndarray,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    fields = [
        "selected",
        "balancedScore",
        "selectionMethod",
        "minClusterSize",
        "minSamples",
        "clusterCount",
        "noisePct",
        "coverage",
        "meanMembership",
        "medianMembership",
        "silhouette2d",
        "parameterAgreementAri",
        "fitSeconds",
    ]
    with (output_directory / "hdbscan-2d-scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda row: row["balancedScore"], reverse=True))
    (output_directory / "hdbscan-2d-selected.json").write_text(
        json.dumps(selected, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(output_directory / "hdbscan-2d-labels.npy", selected_labels, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-data", type=Path, default=Path("public/data/map.json"))
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()

    coordinates = load_coordinates(args.map_data.resolve())
    configurations = [
        (min_cluster_size, min_samples, selection_method)
        for selection_method in SELECTION_METHODS
        for min_cluster_size in MIN_CLUSTER_SIZES
        for min_samples in MIN_SAMPLES
    ]
    fitted = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(fit_configuration)(coordinates, *configuration)
        for configuration in configurations
    )
    results = [row for row, _ in fitted]
    labels = [labels for _, labels in fitted]
    selected = score_results(results, labels)
    selected_index = results.index(selected)
    write_results(args.output.resolve(), results, selected, labels[selected_index])
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
