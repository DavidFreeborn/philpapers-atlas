#!/usr/bin/env python3
"""Build compact, fixed-projection clustering lenses for the PhilPapers Atlas."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


PAPER_COUNT = 69_400
DEFAULT_LENS = "pca100_u30_mcs200_ms15"

HDBSCAN_LENSES = [
    "pca100_u30_mcs200_ms15",
    "pca100_u25_mcs200_ms5",
    "pca100_u20_mcs200_ms15",
    "pca100_u20_mcs100_ms15",
    "pca100_u30_mcs100_ms15",
    "pca100_u30_mcs100_ms5",
    "pca100_u25_mcs100_ms5",
    "pca100_u30_mcs75_ms15",
    "pca100_u20_mcs75_ms15",
    "pca100_u30_mcs75_ms5",
]

KMEANS_NAMES = {
    0: "Education & Learning",
    1: "Mind, Perception & Consciousness",
    2: "Science, Physics & Biology",
    3: "Religion & History of Philosophy",
    4: "Politics, Law & Democracy",
    5: "Ethics, Value & Agency",
    6: "Aesthetics, Culture & Media",
    7: "Bioethics, Health & Disability",
    8: "Environment, Technology & Public Affairs",
    9: "Language, Logic & Epistemology",
}

# The first 42 entries preserve the palette already used by the preferred lens.
BASE_COLOURS = [
    "#82aa3a", "#b262c2", "#5ab22a", "#2aaac2", "#009e73", "#e292ea", "#e26aba",
    "#aa822a", "#ea4292", "#ea6a8a", "#2a8ae2", "#d55e00", "#2ada82", "#ea523a",
    "#baaa6a", "#7a8a52", "#2a9a92", "#eab28a", "#8c72cb", "#e4d34f", "#5a7aea",
    "#5ad2d2", "#aaca82", "#d27a42", "#7a8ac2", "#92922a", "#eab262", "#a2c22a",
    "#56b4e9", "#da52ca", "#ba6a7a", "#ea425a", "#cc79a7", "#e69f00", "#c2c262",
    "#5a9242", "#2aaa4a", "#aa6aea", "#7ad2a2", "#a27a4a", "#82d272", "#ea8a82",
]

KMEANS_COLOURS = [
    "#56b4e9", "#e69f00", "#009e73", "#cc79a7", "#f0e442",
    "#d55e00", "#8f7be8", "#66c2a5", "#f781bf", "#b3a36b",
]


def colour_for(cluster_id: int) -> str:
    if cluster_id < len(BASE_COLOURS):
        return BASE_COLOURS[cluster_id]
    hue = ((cluster_id - len(BASE_COLOURS)) * 0.61803398875 + 0.08) % 1.0
    saturation = (0.62, 0.76, 0.68)[cluster_id % 3]
    lightness = (0.62, 0.69)[cluster_id % 2]
    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def read_csv_by_key(path: Path, key: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("config_key") == key:
                return row
    raise ValueError(f"No metrics found for {key} in {path}")


def parse_config(key: str) -> dict[str, int]:
    parts = key.split("_")
    return {
        "pcaDimensions": int(parts[0].removeprefix("pca")),
        "umapDimensions": int(parts[1].removeprefix("u")),
        "minClusterSize": int(parts[2].removeprefix("mcs")),
        "minSamples": int(parts[3].removeprefix("ms")),
    }


def write_labels(labels: np.ndarray, destination: Path) -> None:
    if labels.shape != (PAPER_COUNT,):
        raise ValueError(f"Expected {PAPER_COUNT} labels, found {labels.shape}")
    if labels.min() < -1 or labels.max() > np.iinfo(np.int16).max:
        raise ValueError("Labels do not fit the compact Int16 format")
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels.astype("<i2", copy=False).tofile(destination)


def cluster_rows(labels: np.ndarray, descriptions: dict[str, dict]) -> list[dict]:
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    rows = []
    for cluster_id, count in zip(ids.tolist(), counts.tolist(), strict=True):
        description = descriptions[str(cluster_id)]
        rows.append(
            {
                "id": cluster_id,
                "label": description["label"],
                "count": count,
                "terms": description.get("tfidf_terms", [])[:10],
                "color": colour_for(cluster_id),
            }
        )
    return rows


def kmeans_terms(public_data: Path, labels: np.ndarray) -> dict[int, list[str]]:
    search_rows = json.loads((public_data / "search.json").read_text(encoding="utf-8"))
    abstracts: list[str] = []
    for shard in sorted((public_data / "details").glob("*.json")):
        abstracts.extend(row[0] or "" for row in json.loads(shard.read_text(encoding="utf-8")))
    if len(search_rows) != PAPER_COUNT or len(abstracts) != PAPER_COUNT:
        raise ValueError("The search and abstract data must match the lens labels")

    texts = [f"{row[0]}. {row[0]}. {abstract}" for row, abstract in zip(search_rows, abstracts, strict=True)]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=12,
        max_df=0.42,
        max_features=40_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    vocabulary = vectorizer.get_feature_names_out()
    result: dict[int, list[str]] = {}
    for cluster_id in sorted(np.unique(labels).tolist()):
        scores = np.asarray(matrix[labels == cluster_id].mean(axis=0)).ravel()
        best = scores.argsort()[-14:][::-1]
        result[cluster_id] = vocabulary[best].tolist()
    return result


def build(source: Path, public_data: Path) -> None:
    output = public_data / "lenses"
    labels_output = output / "labels"
    labels_output.mkdir(parents=True, exist_ok=True)

    hdbscan_metrics = source / "hdbscan_metrics_v3_pca100_20260712_082428.csv"
    lenses: list[dict] = []
    for key in HDBSCAN_LENSES:
        source_dir = source / "hdbscan" / key
        labels = np.load(source_dir / "labels.npy", allow_pickle=False)
        metrics = read_csv_by_key(hdbscan_metrics, key)
        descriptions = json.loads((source_dir / "cluster_labels_with_tfidf.json").read_text(encoding="utf-8"))
        label_file = labels_output / f"{key}.bin"
        write_labels(labels, label_file)
        method = parse_config(key)
        noise_count = int((labels < 0).sum())
        cluster_count = len(np.unique(labels[labels >= 0]))
        lenses.append(
            {
                "id": key,
                "name": f"HDBSCAN · {cluster_count} clusters",
                "optionLabel": f"{cluster_count} clusters · UMAP {method['umapDimensions']}D",
                "algorithm": "hdbscan",
                "preferred": key == DEFAULT_LENS,
                "labelsFile": f"data/lenses/labels/{key}.bin",
                "clusterCount": cluster_count,
                "noiseCount": noise_count,
                "noisePct": round(noise_count / PAPER_COUNT * 100, 2),
                "method": method,
                "metrics": {
                    "relativeValidity": float(metrics["relative_validity"]),
                    "meanPersistence": float(metrics["cluster_persistence_mean"]),
                    "stabilityWeightedCoverage": float(metrics["stability_weighted_cov"]),
                },
                "clusters": cluster_rows(labels, descriptions),
            }
        )

    kmeans_labels = np.load(source / "kmeans10" / "labels.npy", allow_pickle=False)
    kmeans_file = labels_output / "kmeans_pca100_k10.bin"
    write_labels(kmeans_labels, kmeans_file)
    terms = kmeans_terms(public_data, kmeans_labels)
    with (source / "kmeans_metrics_20260614_103401.csv").open(newline="", encoding="utf-8") as handle:
        kmeans_metric_rows = list(csv.DictReader(handle))
    kmeans_metrics = next(
        row for row in kmeans_metric_rows
        if row["pooling"] == "first_pool" and row["pca_dim"] == "100" and row["kmeans_k"] == "10"
    )
    kmeans_clusters = []
    ids, counts = np.unique(kmeans_labels, return_counts=True)
    for cluster_id, count in zip(ids.tolist(), counts.tolist(), strict=True):
        kmeans_clusters.append(
            {
                "id": cluster_id,
                "label": KMEANS_NAMES[cluster_id],
                "count": count,
                "terms": terms[cluster_id][:10],
                "color": KMEANS_COLOURS[cluster_id],
            }
        )
    lenses.append(
        {
            "id": "kmeans_pca100_k10",
            "name": "K-means · 10 clusters",
            "optionLabel": "10 clusters · PCA 100D",
            "algorithm": "kmeans",
            "preferred": False,
            "labelsFile": "data/lenses/labels/kmeans_pca100_k10.bin",
            "clusterCount": 10,
            "noiseCount": 0,
            "noisePct": 0.0,
            "method": {"pcaDimensions": 100, "k": 10},
            "metrics": {
                "silhouetteCosine": float(kmeans_metrics["silhouette_cosine"]),
                "daviesBouldin": float(kmeans_metrics["davies_bouldin"]),
                "calinskiHarabasz": float(kmeans_metrics["calinski_harabasz"]),
            },
            "clusters": kmeans_clusters,
        }
    )

    catalog = {
        "version": "2.0.0",
        "paperCount": PAPER_COUNT,
        "defaultLens": DEFAULT_LENS,
        "projection": {
            "sourceLens": DEFAULT_LENS,
            "dimensions": 2,
            "nNeighbors": 15,
            "minDist": 0.1,
            "metric": "cosine",
        },
        "lenses": lenses,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {len(lenses)} lenses to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public-data", type=Path, default=Path("public/data"))
    args = parser.parse_args()
    build(args.source.resolve(), args.public_data.resolve())


if __name__ == "__main__":
    main()
