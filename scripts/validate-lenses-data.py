#!/usr/bin/env python3
"""Validate the generated atlas lens catalogue and compact label arrays."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
CATALOG_PATH = PUBLIC / "data" / "lenses" / "catalog.json"
PAPER_COUNT = 69_400
BACKGROUND = "#12121a"
MIN_CONTRAST = 4.5
COLOUR_MATCH_JACCARD = 0.80
STUDY_SOURCES = {
    "study_u30_mcs200_ms15_seed73": ROOT / "analysis" / "robustness-study" / "work" / "remote-linux" / "seed-runs" / "d30" / "seed73" / "labels.npy",
    "study_u20_mcs300_ms50_seed42": ROOT / "analysis" / "robustness-study" / "work" / "runs" / "hdbscan-grid-pca100" / "d20" / "mcs300-ms50-eom" / "labels.npy",
}


def relative_luminance(colour: str) -> float:
    values = np.array([int(colour[index:index + 2], 16) / 255 for index in (1, 3, 5)])
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return float(np.dot(linear, [0.2126, 0.7152, 0.0722]))


def contrast_ratio(left: str, right: str) -> float:
    lighter, darker = sorted((relative_luminance(left), relative_luminance(right)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def validate_pairwise_colour_matches(catalog: dict, labels: dict[str, np.ndarray]) -> int:
    matches = 0
    lenses = catalog["lenses"]
    for left_index, left in enumerate(lenses):
        left_labels = labels[left["id"]]
        left_clusters = {int(row["id"]): row for row in left["clusters"]}
        for right in lenses[left_index + 1:]:
            right_labels = labels[right["id"]]
            right_clusters = {int(row["id"]): row for row in right["clusters"]}
            valid = (left_labels >= 0) & (right_labels >= 0)
            width = max(right_clusters) + 1
            intersections = np.bincount(
                left_labels[valid].astype(np.int64) * width + right_labels[valid],
                minlength=(max(left_clusters) + 1) * width,
            ).reshape(max(left_clusters) + 1, width)
            for left_id, left_cluster in left_clusters.items():
                for right_id, right_cluster in right_clusters.items():
                    intersection = int(intersections[left_id, right_id])
                    if not intersection:
                        continue
                    union = left_cluster["count"] + right_cluster["count"] - intersection
                    if intersection / union >= COLOUR_MATCH_JACCARD:
                        if left_cluster["color"] != right_cluster["color"]:
                            raise ValueError(
                                f"Strong overlap has inconsistent colours: {left['id']}:{left_id} and "
                                f"{right['id']}:{right_id}"
                            )
                        matches += 1
    return matches


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if catalog["paperCount"] != PAPER_COUNT:
        raise ValueError("Unexpected catalogue paper count")
    lens_ids = [lens["id"] for lens in catalog["lenses"]]
    if len(lens_ids) != len(set(lens_ids)):
        raise ValueError("Duplicate lens identifiers")
    if catalog["defaultLens"] not in lens_ids:
        raise ValueError("Default lens is absent")

    labels_by_lens: dict[str, np.ndarray] = {}
    colour_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
    for lens in catalog["lenses"]:
        path = PUBLIC / lens["labelsFile"]
        labels = np.fromfile(path, dtype="<i2")
        if labels.shape != (PAPER_COUNT,):
            raise ValueError(f"{lens['id']} has {labels.size} labels")
        labels_by_lens[lens["id"]] = labels
        clusters = {int(row["id"]): row for row in lens["clusters"]}
        observed_ids, observed_counts = np.unique(labels[labels >= 0], return_counts=True)
        if observed_ids.tolist() != sorted(clusters):
            raise ValueError(f"{lens['id']} cluster identifiers disagree with its labels")
        if lens["clusterCount"] != len(clusters):
            raise ValueError(f"{lens['id']} cluster count is inconsistent")
        if lens["noiseCount"] != int(np.count_nonzero(labels < 0)):
            raise ValueError(f"{lens['id']} noise count is inconsistent")
        for cluster_id, count in zip(observed_ids.tolist(), observed_counts.tolist(), strict=True):
            if clusters[cluster_id]["count"] != count:
                raise ValueError(f"{lens['id']}:{cluster_id} paper count is inconsistent")
        colours = [row["color"] for row in clusters.values()]
        if len(colours) != len(set(colours)):
            raise ValueError(f"{lens['id']} contains duplicate colours")
        if any(not colour_pattern.fullmatch(colour) for colour in colours):
            raise ValueError(f"{lens['id']} contains an invalid colour")
        if any(contrast_ratio(colour, BACKGROUND) < MIN_CONTRAST for colour in colours):
            raise ValueError(f"{lens['id']} contains a low-contrast colour")

    for lens_id, source in STUDY_SOURCES.items():
        if lens_id not in labels_by_lens:
            raise ValueError(f"Missing robustness-study lens {lens_id}")
        if source.exists() and not np.array_equal(labels_by_lens[lens_id], np.load(source, allow_pickle=False)):
            raise ValueError(f"Published labels do not exactly match {source}")
        lens = next(item for item in catalog["lenses"] if item["id"] == lens_id)
        metrics = lens["metrics"]
        if metrics.get("resamplingRuns") != 20 or metrics.get("collapsedRuns") != 3:
            raise ValueError(f"{lens_id} robustness metadata is incomplete")

    matches = validate_pairwise_colour_matches(catalog, labels_by_lens)
    print(
        f"Validated {len(catalog['lenses'])} lenses, {PAPER_COUNT:,} aligned papers, "
        f"and {matches} strong cross-lens colour matches."
    )


if __name__ == "__main__":
    main()
