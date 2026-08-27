#!/usr/bin/env python3
"""Independent integrity and headline-value audit for the completed study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import hdbscan
import numpy as np
import pandas as pd
import sklearn
import umap
import yaml
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
WORK = STUDY / "work"
RESULTS = STUDY / "results"
FINALISTS = yaml.safe_load((STUDY / "finalists.yaml").read_text(encoding="utf-8"))["finalists"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_labels(finalist: dict) -> np.ndarray:
    coordinates = np.load(ROOT / finalist["reference_embedding"], mmap_mode="r")
    if finalist.get("hdbscan_implementation") == "sklearn":
        return SklearnHDBSCAN(
            min_cluster_size=int(finalist["min_cluster_size"]),
            min_samples=int(finalist["min_samples"]),
            metric="euclidean",
            cluster_selection_method="eom",
            algorithm="kd_tree",
            leaf_size=60,
            n_jobs=-1,
            copy=True,
        ).fit_predict(coordinates)
    return hdbscan.HDBSCAN(
        min_cluster_size=int(finalist["min_cluster_size"]),
        min_samples=int(finalist["min_samples"]),
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    ).fit_predict(coordinates)


def main() -> None:
    started = perf_counter()
    checks: dict[str, object] = {}
    expected_rows = {
        "pca100-seed-runs.csv": 70,
        "pca100-hdbscan-grid.csv": 175,
        "pca50-seed-runs.csv": 9,
        "pca50-hdbscan-grid.csv": 75,
        "umap-sensitivity-runs.csv": 96,
        "direct2-historical-params-runs.csv": 15,
        "resampling-runs.csv": 280,
        "full-resampling-runs.csv": 240,
        "semantic-summary.csv": 7,
        "semantic-null.csv": 700,
        "semantic-clusters.csv": 406,
        "cluster-audit.csv": 406,
    }
    row_counts = {}
    for name, expected in expected_rows.items():
        frame = pd.read_csv(RESULTS / name)
        row_counts[name] = len(frame)
        if len(frame) != expected:
            raise AssertionError(f"{name}: expected {expected} rows, found {len(frame)}")
    checks["row_counts"] = row_counts

    raw = pd.read_csv(RESULTS / "full-resampling-runs.csv")
    summary = pd.read_csv(RESULTS / "full-resampling-summary.csv")
    recomputed = raw.groupby(["finalist", "strategy"], sort=False).agg(
        ari_all_mean=("ari_all", "mean"),
        cluster_jaccard_weighted_mean=("cluster_jaccard_weighted", "mean"),
        candidate_noise_pct_median=("candidate_noise_pct", "median"),
    )
    maximum_difference = 0.0
    for _, row in summary.iterrows():
        values = recomputed.loc[(row["finalist"], row["strategy"])]
        for metric in values.index:
            difference = abs(float(row[metric]) - float(values[metric]))
            maximum_difference = max(maximum_difference, difference)
            if difference > 1e-12:
                raise AssertionError(f"Summary mismatch: {row['finalist']} {row['strategy']} {metric}")
    checks["headline_recomputation_max_abs_difference"] = maximum_difference

    refit_ids = ("current_30d_42", "robust_20d_29", "direct_2d_historical_params")
    refits = {}
    for finalist_id in refit_ids:
        finalist = next(value for value in FINALISTS if value["id"] == finalist_id)
        reference = np.load(ROOT / finalist["reference_labels"])
        candidate = fresh_labels(finalist)
        ari = float(adjusted_rand_score(reference, candidate))
        threshold = 0.999999 if finalist_id != "direct_2d_historical_params" else 0.999
        if ari < threshold:
            raise AssertionError(f"Fresh refit failed for {finalist_id}: ARI={ari}")
        refits[finalist_id] = ari
    checks["fresh_refits"] = refits

    audit = json.loads((WORK / "semantic" / "neighbor-audit.json").read_text(encoding="utf-8"))
    if audit["recall_at_15_mean"] < 0.98 or audit["recall_at_50_mean"] < 0.98:
        raise AssertionError("Approximate-neighbour audit is below the fixed recall floor")
    checks["neighbor_audit"] = audit

    report = (STUDY / "REPORT.md").read_text(encoding="utf-8")
    placeholders = [
        marker
        for marker in ("FINAL_CONCLUSIONS", "RESAMPLING_RESULTS", "CLUSTER_AUDIT_RESULTS", "RECOMMENDATIONS")
        if marker in report
    ]
    if placeholders:
        raise AssertionError(f"Unresolved report placeholders: {placeholders}")
    checks["report_placeholders"] = placeholders

    figure_files = sorted((RESULTS / "figures").glob("*"))
    if len(figure_files) != 10 or any(path.stat().st_size < 10_000 for path in figure_files):
        raise AssertionError("Expected five non-empty PNG/PDF figure pairs")
    checks["figures"] = {path.name: path.stat().st_size for path in figure_files}

    result_hashes = {
        path.relative_to(STUDY).as_posix(): sha256(path)
        for path in sorted(RESULTS.rglob("*"))
        if path.is_file() and path.name != "verification.json"
    }
    checks["result_hashes"] = result_hashes
    checks["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "umap": umap.__version__,
        "hdbscan": importlib.metadata.version("hdbscan"),
    }
    checks["duration_seconds"] = perf_counter() - started
    temporary = RESULTS / "verification.json.tmp"
    temporary.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(RESULTS / "verification.json")
    print(json.dumps({key: value for key, value in checks.items() if key != "result_hashes"}, indent=2))


if __name__ == "__main__":
    main()
