"""Shared, deterministic metrics for the PhilPapers robustness study."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score


NOISE = -1


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def assigned_cluster_ids(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.unique(labels[labels != NOISE])


def validate_labels(labels: np.ndarray, expected_length: int | None = None) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"Labels must be one-dimensional, got {labels.shape}")
    if expected_length is not None and len(labels) != expected_length:
        raise ValueError(f"Expected {expected_length} labels, got {len(labels)}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Labels must be integers, got {labels.dtype}")
    return labels


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    labels = validate_labels(labels)
    ids, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(ids, counts, strict=True)}


def hdbscan_summary(model: object, labels: np.ndarray, duration_seconds: float) -> dict[str, float | int]:
    labels = validate_labels(labels)
    assigned = labels != NOISE
    ids, counts = np.unique(labels[assigned], return_counts=True)
    probabilities = np.asarray(model.probabilities_)
    persistence = np.asarray(model.cluster_persistence_)
    if len(ids) != len(persistence):
        raise ValueError("HDBSCAN persistence and cluster-count lengths differ")

    coverage = float(assigned.mean())
    weighted_persistence_coverage = (
        float(np.dot(counts.astype(float), persistence) / len(labels)) if len(ids) else 0.0
    )
    size_share = counts.astype(float) / counts.sum() if len(counts) else np.array([], dtype=float)
    size_gini = (
        float(np.abs(size_share[:, None] - size_share[None, :]).mean() / (2 * size_share.mean()))
        if len(size_share)
        else 0.0
    )
    return {
        "n_papers": int(len(labels)),
        "cluster_count": int(len(ids)),
        "noise_count": int((~assigned).sum()),
        "noise_pct": float((~assigned).mean() * 100),
        "coverage": coverage,
        "relative_validity": float(model.relative_validity_),
        "persistence_mean": float(persistence.mean()) if len(persistence) else 0.0,
        "persistence_median": float(np.median(persistence)) if len(persistence) else 0.0,
        "persistence_weighted_coverage": weighted_persistence_coverage,
        "membership_mean_assigned": float(probabilities[assigned].mean()) if assigned.any() else 0.0,
        "membership_median_assigned": float(np.median(probabilities[assigned])) if assigned.any() else 0.0,
        "cluster_size_min": int(counts.min()) if len(counts) else 0,
        "cluster_size_median": float(np.median(counts)) if len(counts) else 0.0,
        "cluster_size_max": int(counts.max()) if len(counts) else 0,
        "cluster_size_gini": size_gini,
        "fit_seconds": float(duration_seconds),
    }


def _contingency(
    reference: np.ndarray,
    candidate: np.ndarray,
    include_noise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = validate_labels(reference)
    candidate = validate_labels(candidate, len(reference))
    if include_noise:
        mask = np.ones(len(reference), dtype=bool)
    else:
        mask = (reference != NOISE) & (candidate != NOISE)
    ref_ids = np.unique(reference[mask])
    cand_ids = np.unique(candidate[mask])
    matrix = np.zeros((len(ref_ids), len(cand_ids)), dtype=np.int64)
    if not mask.any() or not len(ref_ids) or not len(cand_ids):
        return ref_ids, cand_ids, matrix
    ref_index = np.searchsorted(ref_ids, reference[mask])
    cand_index = np.searchsorted(cand_ids, candidate[mask])
    np.add.at(matrix, (ref_index, cand_index), 1)
    return ref_ids, cand_ids, matrix


def best_match_jaccard(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> list[dict[str, float | int]]:
    """Best candidate match for every non-noise reference cluster.

    Matching is deliberately many-to-one, as in cluster-wise bootstrap assessment:
    a split or merge can therefore be diagnosed rather than hidden by a one-to-one
    assignment constraint.
    """

    reference = validate_labels(reference)
    candidate = validate_labels(candidate, len(reference))
    ref_ids = assigned_cluster_ids(reference)
    cand_ids = assigned_cluster_ids(candidate)
    ref_sizes_array = np.array([np.count_nonzero(reference == i) for i in ref_ids], dtype=np.int64)
    cand_sizes_array = np.array([np.count_nonzero(candidate == i) for i in cand_ids], dtype=np.int64)
    intersections = np.zeros((len(ref_ids), len(cand_ids)), dtype=np.int64)
    common = (reference != NOISE) & (candidate != NOISE)
    if common.any() and len(ref_ids) and len(cand_ids):
        ref_positions = np.searchsorted(ref_ids, reference[common])
        cand_positions = np.searchsorted(cand_ids, candidate[common])
        np.add.at(intersections, (ref_positions, cand_positions), 1)
        unions = ref_sizes_array[:, None] + cand_sizes_array[None, :] - intersections
        scores = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
    else:
        scores = np.zeros_like(intersections, dtype=float)
    output: list[dict[str, float | int]] = []
    for ref_position, ref_id in enumerate(ref_ids):
        if not len(cand_ids):
            best_id, intersection, score = NOISE, 0, 0.0
        else:
            best_position = int(np.argmax(scores[ref_position]))
            best_id = int(cand_ids[best_position])
            intersection = int(intersections[ref_position, best_position])
            score = float(scores[ref_position, best_position])
        output.append(
            {
                "reference_cluster": int(ref_id),
                "candidate_cluster": best_id,
                "reference_size": int(ref_sizes_array[ref_position]),
                "candidate_size": int(cand_sizes_array[best_position]) if best_id != NOISE else 0,
                "intersection": intersection,
                "jaccard": score,
                "reference_recall": float(intersection / ref_sizes_array[ref_position]),
            }
        )
    return output


def optimal_label_mapping(reference: np.ndarray, candidate: np.ndarray) -> dict[int, int]:
    """One-to-one candidate→reference mapping maximizing shared assigned papers."""

    ref_ids, cand_ids, intersections = _contingency(reference, candidate)
    if not intersections.size:
        return {}
    rows, columns = linear_sum_assignment(-intersections)
    return {int(cand_ids[column]): int(ref_ids[row]) for row, column in zip(rows, columns, strict=True)}


def mapped_assignment_agreement(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = validate_labels(reference)
    candidate = validate_labels(candidate, len(reference))
    mapping = optimal_label_mapping(reference, candidate)
    mapped = np.full(len(candidate), NOISE, dtype=np.int32)
    for source, target in mapping.items():
        mapped[candidate == source] = target
    return float(np.mean(mapped == reference))


def partition_comparison(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference = validate_labels(reference)
    candidate = validate_labels(candidate, len(reference))
    common_assigned = (reference != NOISE) & (candidate != NOISE)
    matches = best_match_jaccard(reference, candidate)
    weights = np.array([row["reference_size"] for row in matches], dtype=float)
    jaccards = np.array([row["jaccard"] for row in matches], dtype=float)
    return {
        "ari_all": float(adjusted_rand_score(reference, candidate)),
        "ami_all": float(adjusted_mutual_info_score(reference, candidate)),
        "ari_common_assigned": (
            float(adjusted_rand_score(reference[common_assigned], candidate[common_assigned]))
            if common_assigned.sum() > 1
            else 0.0
        ),
        "mapped_assignment_agreement": mapped_assignment_agreement(reference, candidate),
        "noise_agreement": float(np.mean((reference == NOISE) == (candidate == NOISE))),
        "cluster_jaccard_mean": float(jaccards.mean()) if len(jaccards) else 0.0,
        "cluster_jaccard_median": float(np.median(jaccards)) if len(jaccards) else 0.0,
        "cluster_jaccard_weighted": (
            float(np.average(jaccards, weights=weights)) if weights.sum() else 0.0
        ),
    }


def pairwise_partition_rows(
    names: Sequence[str],
    labels: Sequence[np.ndarray],
) -> list[dict[str, float | str]]:
    if len(names) != len(labels):
        raise ValueError("Names and label arrays differ in length")
    rows: list[dict[str, float | str]] = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            row: dict[str, float | str] = {"left": names[left], "right": names[right]}
            row.update(partition_comparison(labels[left], labels[right]))
            rows.append(row)
    return rows


def pareto_flags(rows: Sequence[dict], objectives: dict[str, str]) -> list[bool]:
    """Return nondominance flags for maximized/minimized numeric objectives."""

    directions = {key: (1.0 if value == "max" else -1.0) for key, value in objectives.items()}
    if any(value not in {"max", "min"} for value in objectives.values()):
        raise ValueError("Objective direction must be 'max' or 'min'")
    values = np.array(
        [[float(row[key]) * directions[key] for key in objectives] for row in rows],
        dtype=float,
    )
    flags = np.ones(len(rows), dtype=bool)
    for index in range(len(rows)):
        dominates = np.all(values >= values[index], axis=1) & np.any(values > values[index], axis=1)
        dominates[index] = False
        if dominates.any():
            flags[index] = False
    return flags.tolist()


def percentile_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    if not len(values):
        return (float("nan"), float("nan"))
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(values, tail)), float(np.quantile(values, 1.0 - tail))


def stability_band(jaccard: float) -> str:
    if jaccard < 0.60:
        return "below_0.60"
    if jaccard < 0.75:
        return "0.60_to_0.75"
    if jaccard < 0.85:
        return "0.75_to_0.85"
    return "at_least_0.85"
