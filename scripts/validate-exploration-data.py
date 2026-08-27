#!/usr/bin/env python3
"""Validate semantic-neighbour shards and characteristic-paper rankings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "public" / "data"


def decode_uint24(path: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    if raw.size % 3:
        raise ValueError(f"{path} does not contain complete uint24 values")
    raw = raw.reshape(-1, 3).astype(np.uint32)
    return raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)


def main() -> None:
    metadata = json.loads((PUBLIC_DATA / "exploration" / "neighbors.json").read_text(encoding="utf-8"))
    representatives = json.loads(
        (PUBLIC_DATA / "exploration" / "representatives.json").read_text(encoding="utf-8")
    )
    catalog = json.loads((PUBLIC_DATA / "lenses" / "catalog.json").read_text(encoding="utf-8"))
    paper_count = int(metadata["paperCount"])
    per_paper = int(metadata["neighborsPerPaper"])
    chunk_size = int(metadata["chunkSize"])
    if paper_count != int(catalog["paperCount"]) or representatives["paperCount"] != paper_count:
        raise ValueError("Exploration data and lens catalogue paper counts differ")

    decoded_rows = 0
    for chunk_index, start in enumerate(range(0, paper_count, chunk_size)):
        stop = min(paper_count, start + chunk_size)
        path = PUBLIC_DATA / "exploration" / "neighbors" / f"{chunk_index:03d}.bin"
        values = decode_uint24(path).reshape(stop - start, per_paper)
        queries = np.arange(start, stop, dtype=np.uint32)
        if np.any(values >= paper_count):
            raise ValueError(f"{path} contains an out-of-range neighbour")
        if np.any(values == queries[:, None]):
            raise ValueError(f"{path} contains a self-neighbour")
        if any(len(set(row.tolist())) != per_paper for row in values):
            raise ValueError(f"{path} contains duplicate neighbours")
        decoded_rows += len(values)
    if decoded_rows != paper_count:
        raise ValueError("Neighbour shards contain the wrong number of rows")

    representative_lists = 0
    representative_papers = 0
    for lens in catalog["lenses"]:
        rows = representatives["lenses"].get(lens["id"])
        if rows is None or len(rows) != int(lens["clusterCount"]):
            raise ValueError(f"Representative lists do not match {lens['id']}")
        labels = np.fromfile(ROOT / "public" / lens["labelsFile"], dtype="<i2")
        for cluster_id, ranked in enumerate(rows):
            indices = [int(item[0]) for item in ranked]
            scores = [float(item[1]) for item in ranked]
            if not 1 <= len(indices) <= int(representatives["method"]["representativesPerCluster"]):
                raise ValueError(f"Unexpected representative count for {lens['id']}:{cluster_id}")
            if len(indices) != len(set(indices)) or any(labels[index] != cluster_id for index in indices):
                raise ValueError(f"Invalid representative membership for {lens['id']}:{cluster_id}")
            if scores != sorted(scores, reverse=True):
                raise ValueError(f"Representative scores are not descending for {lens['id']}:{cluster_id}")
            representative_lists += 1
            representative_papers += len(indices)

    print(
        f"Validated {decoded_rows} neighbour rows, {representative_lists} cluster lists, "
        f"and {representative_papers} representative papers."
    )


if __name__ == "__main__":
    main()
