#!/usr/bin/env python3
"""Summarize the preregistered Explorer/Linux seed-replication branch."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from robustness_metrics import (
    best_match_jaccard,
    pairwise_partition_rows,
    partition_comparison,
    percentile_interval,
    stability_band,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
WORK = STUDY / "work"
RESULTS = STUDY / "results"
REMOTE = WORK / "remote-linux" / "seed-runs"
LOCAL = WORK / "runs" / "umap-pca100"
SOURCE = WORK / "source"
DIMENSIONS = (2, 10, 30)
SEEDS = (11, 42, 73, 211, 547)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    extras = sorted(set().union(*(row.keys() for row in rows)) - set(fields))
    fields.extend(extras)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    run_rows: list[dict] = []
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    cross_platform_rows: list[dict] = []
    cluster_rows: list[dict] = []
    for dimension in DIMENSIONS:
        names = []
        labels = []
        metadata = []
        for seed in SEEDS:
            directory = REMOTE / f"d{dimension}" / f"seed{seed}"
            row = json.loads((directory / "run.json").read_text(encoding="utf-8"))
            row["platform_branch"] = "linux_explorer"
            run_rows.append(row)
            metadata.append(row)
            names.append(f"seed{seed}")
            remote_labels = np.load(directory / "labels.npy")
            labels.append(remote_labels)

            local_directory = LOCAL / f"d{dimension}" / f"seed{seed}"
            local_labels = np.load(local_directory / "labels.npy")
            local_embedding = np.load(local_directory / "embedding.npy")
            remote_embedding = np.load(directory / "embedding.npy")
            comparison = partition_comparison(remote_labels, local_labels)
            cross_platform_rows.append(
                {
                    "dimension": dimension,
                    "seed": seed,
                    "embedding_byte_identical": bool(np.array_equal(remote_embedding, local_embedding)),
                    "embedding_mean_absolute_difference": float(np.abs(remote_embedding - local_embedding).mean()),
                    "embedding_flat_correlation": float(
                        np.corrcoef(remote_embedding.ravel(), local_embedding.ravel())[0, 1]
                    ),
                    **comparison,
                }
            )

        comparisons = pairwise_partition_rows(names, labels)
        for row in comparisons:
            row["dimension"] = dimension
            row["platform_branch"] = "linux_explorer"
        pair_rows.extend(comparisons)
        mean_ari = {
            name: float(
                np.mean(
                    [row["ari_all"] for row in comparisons if row["left"] == name or row["right"] == name]
                )
            )
            for name in names
        }
        medoid = max(mean_ari, key=mean_ari.get)
        medoid_labels = labels[names.index(medoid)]
        matches: dict[int, list[float]] = defaultdict(list)
        for name, candidate in zip(names, labels, strict=True):
            if name == medoid:
                continue
            for row in best_match_jaccard(medoid_labels, candidate):
                matches[int(row["reference_cluster"])].append(float(row["jaccard"]))
        for cluster, values in sorted(matches.items()):
            low, high = percentile_interval(values)
            mean = float(np.mean(values))
            cluster_rows.append(
                {
                    "dimension": dimension,
                    "medoid_seed": int(medoid.removeprefix("seed")),
                    "cluster": cluster,
                    "size": int(np.count_nonzero(medoid_labels == cluster)),
                    "seed_jaccard_mean": mean,
                    "seed_jaccard_median": float(np.median(values)),
                    "seed_jaccard_min": float(np.min(values)),
                    "seed_jaccard_ci_low": low,
                    "seed_jaccard_ci_high": high,
                    "stability_band": stability_band(mean),
                }
            )
        summary_rows.append(
            {
                "dimension": dimension,
                "runs": len(labels),
                "medoid_seed": int(medoid.removeprefix("seed")),
                "cluster_count_median": float(np.median([row["cluster_count"] for row in metadata])),
                "cluster_count_min": min(row["cluster_count"] for row in metadata),
                "cluster_count_max": max(row["cluster_count"] for row in metadata),
                "noise_pct_median": float(np.median([row["noise_pct"] for row in metadata])),
                "relative_validity_median": float(np.median([row["relative_validity"] for row in metadata])),
                "relative_validity_min": min(row["relative_validity"] for row in metadata),
                "relative_validity_max": max(row["relative_validity"] for row in metadata),
                "pairwise_ari_mean": float(np.mean([row["ari_all"] for row in comparisons])),
                "pairwise_ari_median": float(np.median([row["ari_all"] for row in comparisons])),
                "pairwise_ari_min": float(np.min([row["ari_all"] for row in comparisons])),
                "pairwise_ami_mean": float(np.mean([row["ami_all"] for row in comparisons])),
                "pairwise_cluster_jaccard_weighted_mean": float(
                    np.mean([row["cluster_jaccard_weighted"] for row in comparisons])
                ),
            }
        )

    canonical_embedding = np.load(SOURCE / "canonical" / "umap30.npy")
    canonical_labels = np.load(SOURCE / "preferred-labels.npy")
    linux_seed42_embedding = np.load(REMOTE / "d30" / "seed42" / "embedding.npy")
    linux_seed42_labels = np.load(REMOTE / "d30" / "seed42" / "labels.npy")
    canonical_check = {
        "embedding_byte_identical": bool(np.array_equal(canonical_embedding, linux_seed42_embedding)),
        "embedding_mean_absolute_difference": float(
            np.abs(canonical_embedding - linux_seed42_embedding).mean()
        ),
        "embedding_flat_correlation": float(
            np.corrcoef(canonical_embedding.ravel(), linux_seed42_embedding.ravel())[0, 1]
        ),
        "clustering_ari": float(adjusted_rand_score(canonical_labels, linux_seed42_labels)),
    }
    write_csv(RESULTS / "linux-seed-runs.csv", run_rows)
    write_csv(RESULTS / "linux-seed-pairwise.csv", pair_rows)
    write_csv(RESULTS / "linux-seed-summary.csv", summary_rows)
    write_csv(RESULTS / "linux-seed-cluster-stability.csv", cluster_rows)
    write_csv(RESULTS / "cross-platform-seeds.csv", cross_platform_rows)
    (RESULTS / "linux-canonical-reproduction.json").write_text(
        json.dumps(canonical_check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"linux_summary": summary_rows, "canonical_check": canonical_check}, indent=2))


if __name__ == "__main__":
    main()
