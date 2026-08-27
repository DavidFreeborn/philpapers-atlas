from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "robustness_metrics.py"
SPEC = importlib.util.spec_from_file_location("robustness_metrics", MODULE_PATH)
assert SPEC and SPEC.loader
metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(metrics)


def test_best_match_jaccard_detects_split() -> None:
    reference = np.array([0, 0, 0, 0, 1, 1, 1, -1])
    candidate = np.array([4, 4, 5, 5, 7, 7, 7, -1])
    rows = metrics.best_match_jaccard(reference, candidate)
    assert rows[0]["jaccard"] == 0.5
    assert rows[0]["reference_recall"] == 0.5
    assert rows[1]["jaccard"] == 1.0


def test_partition_comparison_is_label_permutation_invariant() -> None:
    reference = np.array([0, 0, 1, 1, -1, 2, 2])
    candidate = np.array([9, 9, 4, 4, -1, 7, 7])
    row = metrics.partition_comparison(reference, candidate)
    assert row["ari_all"] == 1.0
    assert row["ami_all"] == 1.0
    assert row["mapped_assignment_agreement"] == 1.0
    assert row["cluster_jaccard_weighted"] == 1.0


def test_noise_changes_reduce_agreement() -> None:
    reference = np.array([0, 0, 1, 1, -1, -1])
    candidate = np.array([0, -1, 1, 1, -1, 2])
    row = metrics.partition_comparison(reference, candidate)
    assert 0 < row["noise_agreement"] < 1
    assert 0 < row["mapped_assignment_agreement"] < 1


def test_pareto_flags() -> None:
    rows = [
        {"quality": 1.0, "noise": 0.4},
        {"quality": 0.9, "noise": 0.3},
        {"quality": 0.8, "noise": 0.5},
    ]
    assert metrics.pareto_flags(rows, {"quality": "max", "noise": "min"}) == [True, True, False]


def test_stability_bands_boundaries() -> None:
    assert metrics.stability_band(0.59) == "below_0.60"
    assert metrics.stability_band(0.60) == "0.60_to_0.75"
    assert metrics.stability_band(0.75) == "0.75_to_0.85"
    assert metrics.stability_band(0.85) == "at_least_0.85"


def test_optimal_mapping_handles_permuted_labels_and_noise() -> None:
    reference = np.array([0, 0, 1, 1, 2, 2, -1])
    candidate = np.array([8, 8, 3, 3, 5, 5, -1])
    assert metrics.optimal_label_mapping(reference, candidate) == {8: 0, 3: 1, 5: 2}


def test_percentile_interval_is_order_invariant() -> None:
    left = metrics.percentile_interval([1, 2, 3, 4, 5])
    right = metrics.percentile_interval([5, 3, 1, 4, 2])
    assert left == right
