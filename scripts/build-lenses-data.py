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
COLOUR_MATCH_JACCARD = 0.80
MAP_BACKGROUND = "#12121a"
MIN_POINT_CONTRAST = 4.5

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

# These preserve the palette already used by the selected 42-cluster lens.
BASE_COLOURS = [
    "#82aa3a", "#b262c2", "#5ab22a", "#2aaac2", "#009e73", "#e292ea", "#e26aba",
    "#aa822a", "#ea4292", "#ea6a8a", "#2a8ae2", "#d55e00", "#2ada82", "#ea523a",
    "#baaa6a", "#7a8a52", "#2a9a92", "#eab28a", "#8c72cb", "#e4d34f", "#5a7aea",
    "#5ad2d2", "#aaca82", "#d27a42", "#7a8ac2", "#92922a", "#eab262", "#a2c22a",
    "#56b4e9", "#da52ca", "#ba6a7a", "#ea425a", "#cc79a7", "#e69f00", "#c2c262",
    "#5a9242", "#2aaa4a", "#aa6aea", "#7ad2a2", "#a27a4a", "#82d272", "#ea8a82",
]

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
                "color": "",
            }
        )
    return rows


def hex_to_oklab(colour: str) -> np.ndarray:
    """Convert an sRGB hex colour to OKLab for perceptual distance scoring."""
    rgb = np.array([int(colour[index:index + 2], 16) / 255 for index in (1, 3, 5)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    red, green, blue = linear
    l_value = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m_value = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s_value = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l_root, m_root, s_root = np.cbrt([l_value, m_value, s_value])
    return np.array(
        [
            0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
            1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
            0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
        ]
    )


def relative_luminance(colour: str) -> float:
    rgb = np.array([int(colour[index:index + 2], 16) / 255 for index in (1, 3, 5)])
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(np.dot(linear, [0.2126, 0.7152, 0.0722]))


def contrast_ratio(left: str, right: str) -> float:
    lighter, darker = sorted((relative_luminance(left), relative_luminance(right)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def palette_candidates() -> list[str]:
    """Create a large, deterministic pool of saturated colours for the dark map."""
    colours: list[str] = []
    seen = set(BASE_COLOURS)
    for lightness in (0.48, 0.57, 0.66, 0.74):
        for saturation in (0.58, 0.72, 0.86):
            for hue_degrees in range(0, 360, 3):
                red, green, blue = colorsys.hls_to_rgb(hue_degrees / 360, lightness, saturation)
                colour = f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
                if colour not in seen and contrast_ratio(colour, MAP_BACKGROUND) >= MIN_POINT_CONTRAST:
                    seen.add(colour)
                    colours.append(colour)
    return colours


def overlap_edges(lenses: list[dict], labels_by_lens: dict[str, np.ndarray]) -> list[tuple[float, tuple[str, int], tuple[str, int]]]:
    """Return every cross-lens cluster pair meeting the colour-equivalence threshold."""
    edges: list[tuple[float, tuple[str, int], tuple[str, int]]] = []
    for left_index, left in enumerate(lenses):
        left_labels = labels_by_lens[left["id"]]
        left_counts = {cluster["id"]: cluster["count"] for cluster in left["clusters"]}
        for right in lenses[left_index + 1:]:
            right_labels = labels_by_lens[right["id"]]
            right_counts = {cluster["id"]: cluster["count"] for cluster in right["clusters"]}
            valid = (left_labels >= 0) & (right_labels >= 0)
            right_width = max(right_counts) + 1
            intersections = np.bincount(
                left_labels[valid].astype(np.int64) * right_width + right_labels[valid],
                minlength=(max(left_counts) + 1) * right_width,
            ).reshape(max(left_counts) + 1, right_width)
            for left_id, left_count in left_counts.items():
                for right_id, right_count in right_counts.items():
                    intersection = int(intersections[left_id, right_id])
                    if not intersection:
                        continue
                    jaccard = intersection / (left_count + right_count - intersection)
                    if jaccard >= COLOUR_MATCH_JACCARD:
                        edges.append(
                            (jaccard, (left["id"], left_id), (right["id"], right_id))
                        )
    return sorted(edges, reverse=True)


def assign_consistent_colours(lenses: list[dict], labels_by_lens: dict[str, np.ndarray]) -> None:
    """Share colours for stable memberships and separate all other coexisting clusters."""
    nodes = [
        (lens["id"], cluster["id"])
        for lens in lenses
        for cluster in lens["clusters"]
    ]
    parent = {node: node for node in nodes}
    component_lenses = {node: {node[0]} for node in nodes}

    def find(node: tuple[str, int]) -> tuple[str, int]:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: tuple[str, int], right: tuple[str, int]) -> bool:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return True
        if not component_lenses[left_root].isdisjoint(component_lenses[right_root]):
            return False
        if len(component_lenses[left_root]) < len(component_lenses[right_root]):
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        component_lenses[left_root].update(component_lenses.pop(right_root))
        return True

    edges = overlap_edges(lenses, labels_by_lens)
    rejected_edges = [edge for edge in edges if not union(edge[1], edge[2])]
    if rejected_edges:
        raise ValueError(
            f"Cannot honour {len(rejected_edges)} colour matches without duplicating a colour within a lens"
        )

    components: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in nodes:
        components.setdefault(find(node), []).append(node)

    lens_components: dict[str, set[tuple[str, int]]] = {}
    for lens in lenses:
        lens_components[lens["id"]] = {
            find((lens["id"], cluster["id"])) for cluster in lens["clusters"]
        }
    conflicts = {root: set() for root in components}
    for roots in lens_components.values():
        for root in roots:
            conflicts[root].update(roots - {root})

    assigned: dict[tuple[str, int], str] = {}
    selected = next(lens for lens in lenses if lens["id"] == DEFAULT_LENS)
    if selected["clusterCount"] > len(BASE_COLOURS):
        raise ValueError("The selected lens contains more clusters than the reference palette")
    for cluster in selected["clusters"]:
        assigned[find((selected["id"], cluster["id"]))] = BASE_COLOURS[cluster["id"]]

    counts = {
        (lens["id"], cluster["id"]): cluster["count"]
        for lens in lenses
        for cluster in lens["clusters"]
    }
    unassigned = [root for root in components if root not in assigned]
    unassigned.sort(
        key=lambda root: (
            max(counts[node] for node in components[root]),
            len(components[root]),
        ),
        reverse=True,
    )

    candidates = palette_candidates()
    candidate_labs = np.vstack([hex_to_oklab(colour) for colour in candidates])
    available = np.ones(len(candidates), dtype=bool)
    assigned_labs = {root: hex_to_oklab(colour) for root, colour in assigned.items()}
    for root in unassigned:
        neighbouring_labs = [assigned_labs[other] for other in conflicts[root] if other in assigned_labs]
        all_labs = list(assigned_labs.values())
        if neighbouring_labs:
            neighbour_matrix = np.vstack(neighbouring_labs)
            neighbour_distance = np.linalg.norm(
                candidate_labs[:, None, :] - neighbour_matrix[None, :, :], axis=2
            ).min(axis=1)
        else:
            neighbour_distance = np.full(len(candidates), 1.0)
        global_matrix = np.vstack(all_labs)
        global_distance = np.linalg.norm(
            candidate_labs[:, None, :] - global_matrix[None, :, :], axis=2
        ).min(axis=1)
        score = neighbour_distance + 0.12 * global_distance
        score[~available] = -np.inf
        candidate_index = int(np.argmax(score))
        if not np.isfinite(score[candidate_index]):
            raise ValueError("The generated colour palette is too small")
        colour = candidates[candidate_index]
        assigned[root] = colour
        assigned_labs[root] = candidate_labs[candidate_index]
        available[candidate_index] = False

    for lens in lenses:
        colours = []
        for cluster in lens["clusters"]:
            colour = assigned[find((lens["id"], cluster["id"]))]
            cluster["color"] = colour
            colours.append(colour)
        if len(colours) != len(set(colours)):
            raise ValueError(f"Lens {lens['id']} contains duplicate cluster colours")

    for _, left, right in edges:
        if assigned[find(left)] != assigned[find(right)]:
            raise ValueError("A strong overlap did not receive a consistent colour")
    print(
        f"Matched {len(edges)} cross-lens pairs at Jaccard >= {COLOUR_MATCH_JACCARD:.2f}; "
        f"assigned {len(components)} colour families"
    )


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
    labels_by_lens: dict[str, np.ndarray] = {}
    for key in HDBSCAN_LENSES:
        source_dir = source / "hdbscan" / key
        labels = np.load(source_dir / "labels.npy", allow_pickle=False)
        labels_by_lens[key] = labels
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
    labels_by_lens["kmeans_pca100_k10"] = kmeans_labels
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
                "color": "",
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

    assign_consistent_colours(lenses, labels_by_lens)

    catalog = {
        "version": "2.1.0",
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
