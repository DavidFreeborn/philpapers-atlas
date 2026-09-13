#!/usr/bin/env python3
"""Build compact data for semantic-neighbour and representative-paper exploration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
PAPER_COUNT = 69_400
EMBEDDING_DIMENSIONS = 768
NEIGHBOURS_PER_PAPER = 50
NEIGHBOUR_CHUNK_SIZE = 200
REPRESENTATIVES_PER_CLUSTER = 10


def uint24_le(values: np.ndarray) -> bytes:
    """Encode non-negative integers below 2**24 as packed little-endian uint24."""
    flat = np.asarray(values, dtype=np.uint32).reshape(-1)
    if flat.size and int(flat.max()) >= 2**24:
        raise ValueError("A neighbour index cannot be represented as uint24")
    encoded = np.empty((flat.size, 3), dtype=np.uint8)
    encoded[:, 0] = flat & 0xFF
    encoded[:, 1] = (flat >> 8) & 0xFF
    encoded[:, 2] = (flat >> 16) & 0xFF
    return encoded.tobytes()


def neighbour_rows(indices: np.ndarray) -> np.ndarray:
    """Remove each query from its approximate-neighbour row and retain the first 50."""
    if indices.shape[0] != PAPER_COUNT or indices.shape[1] < NEIGHBOURS_PER_PAPER + 1:
        raise ValueError(f"Unexpected neighbour graph shape: {indices.shape}")
    result = np.empty((PAPER_COUNT, NEIGHBOURS_PER_PAPER), dtype=np.int32)
    for paper_index in range(PAPER_COUNT):
        row = np.asarray(indices[paper_index], dtype=np.int32)
        row = row[row != paper_index]
        unique = dict.fromkeys(int(value) for value in row)
        if len(unique) < NEIGHBOURS_PER_PAPER:
            raise ValueError(
                f"Paper {paper_index} has fewer than {NEIGHBOURS_PER_PAPER} "
                "unique non-self neighbours"
            )
        result[paper_index] = np.fromiter(
            list(unique)[:NEIGHBOURS_PER_PAPER],
            dtype=np.int32,
            count=NEIGHBOURS_PER_PAPER,
        )
    if np.any(result < 0) or np.any(result >= PAPER_COUNT):
        raise ValueError("The neighbour graph contains an out-of-range paper index")
    return result


def build_neighbour_shards(
    semantic_dir: Path,
    output_dir: Path,
) -> dict:
    source_path = semantic_dir / "neighbors-51.npy"
    audit_path = semantic_dir / "neighbor-audit.json"
    indices = np.load(source_path, mmap_mode="r")
    neighbours = neighbour_rows(indices)
    shards_dir = output_dir / "neighbors"
    shards_dir.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for chunk_index, start in enumerate(range(0, PAPER_COUNT, NEIGHBOUR_CHUNK_SIZE)):
        name = f"{chunk_index:03d}.bin"
        expected_names.add(name)
        stop = min(PAPER_COUNT, start + NEIGHBOUR_CHUNK_SIZE)
        (shards_dir / name).write_bytes(uint24_le(neighbours[start:stop]))
    unexpected = [path for path in shards_dir.glob("*.bin") if path.name not in expected_names]
    if unexpected:
        raise ValueError(f"Unexpected existing neighbour shards: {[path.name for path in unexpected]}")

    exact_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metadata = {
        "version": "1.0.0",
        "paperCount": PAPER_COUNT,
        "embedding": "AllenAI SPECTER",
        "embeddingDimensions": EMBEDDING_DIMENSIONS,
        "metric": "cosine",
        "neighborsPerPaper": NEIGHBOURS_PER_PAPER,
        "chunkSize": NEIGHBOUR_CHUNK_SIZE,
        "encoding": "uint24-le",
        "filePattern": "data/exploration/neighbors/{chunk}.bin",
        "exactAudit": {
            "queries": int(exact_audit["audit_queries"]),
            "recallAt15Mean": float(exact_audit["recall_at_15_mean"]),
            "recallAt50Mean": float(exact_audit["recall_at_50_mean"]),
        },
    }
    (output_dir / "neighbors.json").write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return metadata


def cluster_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
) -> list[list[list[float | int]]]:
    """Rank cluster members by cosine similarity to their SPECTER centroid."""
    if labels.shape != (PAPER_COUNT,):
        raise ValueError(f"Unexpected label shape: {labels.shape}")
    assigned = labels >= 0
    rows = np.flatnonzero(assigned)
    assigned_labels = labels[assigned].astype(np.int32)
    if assigned_labels.size and (
        int(assigned_labels.min()) != 0 or int(assigned_labels.max()) != cluster_count - 1
    ):
        raise ValueError("Cluster identifiers are not contiguous")

    membership = sparse.csr_matrix(
        (
            np.ones(rows.size, dtype=np.float32),
            (assigned_labels, rows),
        ),
        shape=(cluster_count, PAPER_COUNT),
        dtype=np.float32,
    )
    centroids = np.asarray(membership @ embeddings, dtype=np.float32)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.maximum(centroid_norms, 1e-12)

    scores = np.full(PAPER_COUNT, -np.inf, dtype=np.float32)
    for start in range(0, PAPER_COUNT, 2048):
        stop = min(PAPER_COUNT, start + 2048)
        block_labels = labels[start:stop]
        block_assigned = block_labels >= 0
        if not np.any(block_assigned):
            continue
        block = np.asarray(embeddings[start:stop], dtype=np.float32)
        scores[start:stop][block_assigned] = np.einsum(
            "ij,ij->i",
            block[block_assigned],
            centroids[block_labels[block_assigned]],
            optimize=True,
        )

    result: list[list[list[float | int]]] = []
    for cluster_id in range(cluster_count):
        members = np.flatnonzero(labels == cluster_id)
        count = min(REPRESENTATIVES_PER_CLUSTER, members.size)
        if count == 0:
            raise ValueError(f"Cluster {cluster_id} has no papers")
        if count < members.size:
            chosen = members[np.argpartition(scores[members], -count)[-count:]]
        else:
            chosen = members
        chosen = chosen[np.lexsort((chosen, -scores[chosen]))]
        result.append(
            [[int(index), round(float(scores[index]), 6)] for index in chosen]
        )
    return result


def build_representatives(
    public_data: Path,
    semantic_dir: Path,
    output_dir: Path,
) -> dict:
    catalog = json.loads((public_data / "lenses" / "catalog.json").read_text(encoding="utf-8"))
    if int(catalog["paperCount"]) != PAPER_COUNT:
        raise ValueError("The lens catalogue has an unexpected paper count")
    embeddings = np.load(semantic_dir / "specter-normalized.npy", mmap_mode="r")
    if embeddings.shape != (PAPER_COUNT, EMBEDDING_DIMENSIONS):
        raise ValueError(f"Unexpected normalised SPECTER shape: {embeddings.shape}")

    lens_rows: dict[str, list[list[list[float | int]]]] = {}
    for lens in catalog["lenses"]:
        if lens["algorithm"] == "metadata":
            continue
        labels = np.fromfile(ROOT / "public" / lens["labelsFile"], dtype="<i2")
        rows = cluster_representatives(embeddings, labels, int(lens["clusterCount"]))
        lens_rows[lens["id"]] = rows
        print(f"Ranked representatives for {lens['id']} ({lens['clusterCount']} clusters)")

    payload = {
        "version": "1.0.0",
        "paperCount": PAPER_COUNT,
        "method": {
            "name": "SPECTER centroid cosine similarity",
            "embedding": "AllenAI SPECTER",
            "embeddingDimensions": EMBEDDING_DIMENSIONS,
            "representativesPerCluster": REPRESENTATIVES_PER_CLUSTER,
        },
        "lenses": lens_rows,
    }
    (output_dir / "representatives.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-data",
        type=Path,
        default=ROOT / "public" / "data",
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=ROOT / "analysis" / "robustness-study" / "work" / "semantic",
    )
    args = parser.parse_args()
    output_dir = args.public_data / "exploration"
    output_dir.mkdir(parents=True, exist_ok=True)
    neighbour_metadata = build_neighbour_shards(args.semantic_dir, output_dir)
    representative_payload = build_representatives(args.public_data, args.semantic_dir, output_dir)
    print(
        f"Wrote {neighbour_metadata['paperCount']} × {neighbour_metadata['neighborsPerPaper']} "
        f"neighbours and {sum(len(rows) for rows in representative_payload['lenses'].values())} "
        f"cluster representative lists to {output_dir}"
    )


if __name__ == "__main__":
    main()
