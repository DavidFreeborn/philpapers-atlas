#!/usr/bin/env python3
"""Summarize Explorer full-pipeline subsample refits against frozen finalists."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from robustness_metrics import (
    best_match_jaccard,
    partition_comparison,
    percentile_interval,
    stability_band,
)


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
WORK = STUDY / "work"
RESULTS = STUDY / "results"
FINALISTS = yaml.safe_load((STUDY / "finalists.yaml").read_text(encoding="utf-8"))["finalists"]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    base = WORK / "remote-linux" / "full-resampling"
    runs, clusters, summaries = [], [], []
    for finalist in FINALISTS:
        if finalist["kind"] == "fixed_display":
            continue
        reference = np.load(ROOT / finalist["reference_labels"])
        for strategy in ("scaled", "fixed"):
            finalist_rows = []
            per_cluster: dict[int, list[float]] = defaultdict(list)
            per_cluster_nondegenerate: dict[int, list[float]] = defaultdict(list)
            for repeat in range(20):
                directory = base / f"repeat{repeat:02d}"
                metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                indices = np.load(directory / "indices.npy")
                candidate = np.load(directory / f"labels-{finalist['id']}-{strategy}.npy")
                comparison = partition_comparison(reference[indices], candidate)
                assigned = candidate != -1
                row = {
                    "finalist": finalist["id"],
                    "strategy": strategy,
                    "repeat": repeat,
                    "pca_refitted": True,
                    "candidate_cluster_count": int(np.unique(candidate[assigned]).size),
                    "candidate_noise_pct": float(np.mean(~assigned) * 100),
                    "pca_explained_variance_ratio_sum": metadata["pca_explained_variance_ratio_sum"],
                    **comparison,
                }
                finalist_rows.append(row)
                runs.append(row)
                matches = best_match_jaccard(reference[indices], candidate)
                for match in matches:
                    per_cluster[int(match["reference_cluster"])].append(float(match["jaccard"]))
                    if row["candidate_cluster_count"] >= 5:
                        per_cluster_nondegenerate[int(match["reference_cluster"])].append(float(match["jaccard"]))
            for cluster, values in sorted(per_cluster.items()):
                low, high = percentile_interval(values)
                mean = float(np.mean(values))
                nondegenerate_values = per_cluster_nondegenerate[cluster]
                clusters.append(
                    {
                        "finalist": finalist["id"],
                        "strategy": strategy,
                        "cluster": cluster,
                        "full_size": int(np.count_nonzero(reference == cluster)),
                        "subsample_jaccard_mean": mean,
                        "subsample_jaccard_median": float(np.median(values)),
                        "subsample_jaccard_min": float(np.min(values)),
                        "subsample_jaccard_ci_low": low,
                        "subsample_jaccard_ci_high": high,
                        "stability_band": stability_band(mean),
                        "nondegenerate_runs": len(nondegenerate_values),
                        "subsample_jaccard_mean_nondegenerate": float(np.mean(nondegenerate_values)),
                        "subsample_jaccard_median_nondegenerate": float(np.median(nondegenerate_values)),
                    }
                )
            nondegenerate_rows = [row for row in finalist_rows if row["candidate_cluster_count"] >= 5]
            summaries.append(
                {
                    "finalist": finalist["id"],
                    "strategy": strategy,
                    "pca_refitted": True,
                    "runs": 20,
                    "candidate_cluster_count_median": float(np.median([row["candidate_cluster_count"] for row in finalist_rows])),
                    "candidate_cluster_count_min": min(row["candidate_cluster_count"] for row in finalist_rows),
                    "candidate_cluster_count_max": max(row["candidate_cluster_count"] for row in finalist_rows),
                    "degenerate_runs_lt5_clusters": int(sum(row["candidate_cluster_count"] < 5 for row in finalist_rows)),
                    "candidate_noise_pct_median": float(np.median([row["candidate_noise_pct"] for row in finalist_rows])),
                    "ari_all_mean": float(np.mean([row["ari_all"] for row in finalist_rows])),
                    "ari_all_median": float(np.median([row["ari_all"] for row in finalist_rows])),
                    "ari_all_q10": float(np.quantile([row["ari_all"] for row in finalist_rows], 0.10)),
                    "ari_all_min": float(np.min([row["ari_all"] for row in finalist_rows])),
                    "cluster_jaccard_weighted_mean": float(np.mean([row["cluster_jaccard_weighted"] for row in finalist_rows])),
                    "cluster_jaccard_weighted_median": float(np.median([row["cluster_jaccard_weighted"] for row in finalist_rows])),
                    "cluster_jaccard_weighted_q10": float(np.quantile([row["cluster_jaccard_weighted"] for row in finalist_rows], 0.10)),
                    "mapped_assignment_agreement_mean": float(np.mean([row["mapped_assignment_agreement"] for row in finalist_rows])),
                    "noise_agreement_mean": float(np.mean([row["noise_agreement"] for row in finalist_rows])),
                    "nondegenerate_runs": len(nondegenerate_rows),
                    "ari_all_mean_nondegenerate": float(np.mean([row["ari_all"] for row in nondegenerate_rows])),
                    "ari_all_median_nondegenerate": float(np.median([row["ari_all"] for row in nondegenerate_rows])),
                    "cluster_jaccard_weighted_mean_nondegenerate": float(
                        np.mean([row["cluster_jaccard_weighted"] for row in nondegenerate_rows])
                    ),
                    "cluster_jaccard_weighted_median_nondegenerate": float(
                        np.median([row["cluster_jaccard_weighted"] for row in nondegenerate_rows])
                    ),
                }
            )
    write_csv(RESULTS / "full-resampling-runs.csv", runs)
    write_csv(RESULTS / "full-resampling-cluster-stability.csv", clusters)
    write_csv(RESULTS / "full-resampling-summary.csv", summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
