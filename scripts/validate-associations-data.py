#!/usr/bin/env python3
"""Validate metadata lens binaries and every published association score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
CATALOG_PATH = PUBLIC / "data" / "lenses" / "catalog.json"
OVERVIEW_PATH = PUBLIC / "data" / "lenses" / "associations.json"
PAPER_COUNT = 69_400


def corrected_cramers_v(left: np.ndarray, right: np.ndarray) -> float:
    left_ids, left_inverse = np.unique(left, return_inverse=True)
    right_ids, right_inverse = np.unique(right, return_inverse=True)
    rows, columns = left_ids.size, right_ids.size
    if left.size <= 1 or rows <= 1 or columns <= 1:
        return 0.0
    table = np.bincount(
        left_inverse * columns + right_inverse,
        minlength=rows * columns,
    ).reshape(rows, columns)
    total = int(table.sum())
    expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / total
    phi_squared = float(np.sum(np.square(table - expected) / expected)) / total
    correction = ((columns - 1) * (rows - 1)) / (total - 1)
    corrected_phi = max(0.0, phi_squared - correction)
    corrected_rows = rows - ((rows - 1) ** 2) / (total - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (total - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return (corrected_phi / denominator) ** 0.5 if denominator > 0 else 0.0


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    overview = json.loads(OVERVIEW_PATH.read_text(encoding="utf-8"))
    lenses = catalog["lenses"]
    if overview["paperCount"] != PAPER_COUNT:
        raise ValueError("Association overview has the wrong paper count")
    if overview["measure"] != "bias-corrected Cramers V":
        raise ValueError("Unexpected association measure")
    if [row["id"] for row in overview["lenses"]] != [lens["id"] for lens in lenses]:
        raise ValueError("Association lens order differs from the catalogue")
    expected_pairs = len(lenses) * (len(lenses) - 1) // 2
    if len(overview["pairs"]) != expected_pairs:
        raise ValueError("Association overview is missing lens pairs")

    labels = {
        lens["id"]: np.fromfile(PUBLIC / lens["labelsFile"], dtype="<i2")
        for lens in lenses
    }
    pair_lookup = {
        frozenset((pair["left"], pair["right"])): pair
        for pair in overview["pairs"]
    }
    if len(pair_lookup) != expected_pairs:
        raise ValueError("Association overview contains duplicate pairs")

    checked = 0
    for left_index, left_lens in enumerate(lenses):
        left = labels[left_lens["id"]]
        for right_lens in lenses[left_index + 1 :]:
            right = labels[right_lens["id"]]
            eligible = np.ones(PAPER_COUNT, dtype=bool)
            if left_lens["algorithm"] == "metadata":
                eligible &= left >= 0
            if right_lens["algorithm"] == "metadata":
                eligible &= right >= 0
            pair = pair_lookup[frozenset((left_lens["id"], right_lens["id"]))]
            if pair["eligiblePapers"] != int(np.count_nonzero(eligible)):
                raise ValueError(f"Eligible count mismatch for {left_lens['id']} × {right_lens['id']}")
            actual = corrected_cramers_v(left[eligible], right[eligible])
            if abs(actual - pair["cramersV"]) > 5e-7:
                raise ValueError(f"Cramér’s V mismatch for {left_lens['id']} × {right_lens['id']}")
            checked += 1

    metadata = {lens["id"]: lens for lens in lenses if lens["algorithm"] == "metadata"}
    if set(metadata) != {"metadata_publication_type", "metadata_publication_period"}:
        raise ValueError("The expected metadata lenses are absent")
    if metadata["metadata_publication_type"]["noiseCount"] != 3:
        raise ValueError("Publication-type missing count changed unexpectedly")
    if metadata["metadata_publication_period"]["noiseCount"] != 391:
        raise ValueError("Publication-period missing count changed unexpectedly")
    print(f"Validated {checked} association pairs and 2 aligned metadata lenses.")


if __name__ == "__main__":
    main()
