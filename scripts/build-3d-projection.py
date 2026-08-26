#!/usr/bin/env python3
"""Build and evaluate a lazy-loaded 3D UMAP display for the PhilPapers Atlas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from umap import UMAP


PAPER_COUNT = 69_400
RANDOM_STATE = 20260826
SAMPLE_SIZE = 5_000
PAIR_SAMPLE_SIZE = 100_000
MODEL_NAME = "allenai/specter"
MODEL_REVISION = "81cbfb43d4fc2e728d5b4201ce14987db8d0854c"
PCA_DIMENSIONS = 100
SOURCE_UMAP_DIMENSIONS = 30
SOURCE_UMAP_NEIGHBOURS = 15
SOURCE_UMAP_MIN_DIST = 0.0
DISPLAY_EPOCHS = 300
DISPLAY_CONFIGS = (
    (10, 0.05),
    (10, 0.10),
    (15, 0.00),
    (15, 0.05),
    (15, 0.10),
    (15, 0.20),
    (30, 0.05),
    (30, 0.10),
    (50, 0.05),
    (50, 0.10),
)


def load_texts(public_data: Path) -> tuple[list[str], str]:
    search_rows = json.loads((public_data / "search.json").read_text(encoding="utf-8"))
    abstracts: list[str] = []
    for shard in sorted((public_data / "details").glob("*.json")):
        abstracts.extend(row[0] or "" for row in json.loads(shard.read_text(encoding="utf-8")))
    if len(search_rows) != PAPER_COUNT or len(abstracts) != PAPER_COUNT:
        raise ValueError("The title and abstract data do not match the atlas paper count")
    texts = [
        f"{row[0]}[SEP]{abstract}"
        for row, abstract in zip(search_rows, abstracts, strict=True)
    ]
    digest = hashlib.sha256()
    for index, text in enumerate(texts):
        digest.update(index.to_bytes(4, "little"))
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return texts, digest.hexdigest()


def embed_texts(
    texts: list[str],
    destination: Path,
    progress_path: Path,
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, str]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device).eval()
    dimensions = int(model.config.hidden_size)
    if dimensions != 768:
        raise ValueError(f"Expected 768-dimensional SPECTER output, found {dimensions}")
    revision = str(getattr(model.config, "_commit_hash", None) or MODEL_REVISION)
    if revision != MODEL_REVISION:
        raise ValueError(f"Expected SPECTER revision {MODEL_REVISION}, found {revision}")

    start_index = 0
    if destination.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("papers") == len(texts)
            and progress.get("dimensions") == dimensions
            and progress.get("model") == MODEL_NAME
            and progress.get("revision") == MODEL_REVISION
        ):
            start_index = int(progress.get("nextIndex", 0))
    mode = "r+" if destination.exists() else "w+"
    embeddings = np.lib.format.open_memmap(
        destination,
        mode=mode,
        dtype=np.float32,
        shape=(len(texts), dimensions),
    )
    if start_index >= len(texts):
        print(f"Using complete cached SPECTER embeddings at {destination}", flush=True)
        return embeddings, revision

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    started = perf_counter()
    for first in range(start_index, len(texts), batch_size):
        last = min(first + batch_size, len(texts))
        batch_texts = [
            text.replace("[SEP]", tokenizer.sep_token, 1)
            for text in texts[first:last]
        ]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokens = {name: value.to(device, non_blocking=True) for name, value in tokens.items()}
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(**tokens).last_hidden_state[:, 0, :]
            else:
                output = model(**tokens).last_hidden_state[:, 0, :]
        embeddings[first:last] = output.float().cpu().numpy()
        embeddings.flush()
        progress_path.write_text(
            json.dumps(
                {
                    "model": MODEL_NAME,
                    "revision": revision,
                    "papers": len(texts),
                    "dimensions": dimensions,
                    "nextIndex": last,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if last == len(texts) or last % (batch_size * 50) == 0:
            elapsed = perf_counter() - started
            rate = (last - start_index) / max(elapsed, 1e-9)
            remaining = (len(texts) - last) / max(rate, 1e-9)
            print(
                f"Embedded {last:,}/{len(texts):,} papers | {rate:.1f}/s | "
                f"about {remaining / 60:.1f} min remaining",
                flush=True,
            )
    return embeddings, revision


def reduce_to_source_space(embeddings: np.ndarray, work: Path) -> np.ndarray:
    pca_path = work / "pca100.npy"
    source_path = work / "umap30.npy"
    if source_path.exists():
        source = np.load(source_path, mmap_mode="r")
        if source.shape == (PAPER_COUNT, SOURCE_UMAP_DIMENSIONS):
            print(f"Using cached 30D source projection at {source_path}", flush=True)
            return source
    if pca_path.exists():
        pca_values = np.load(pca_path, mmap_mode="r")
        if pca_values.shape != (PAPER_COUNT, PCA_DIMENSIONS):
            raise ValueError(f"Unexpected cached PCA shape: {pca_values.shape}")
    else:
        print("Fitting PCA 768D -> 100D", flush=True)
        pca_values = PCA(
            n_components=PCA_DIMENSIONS,
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        ).fit_transform(np.asarray(embeddings))
        pca_values = pca_values.astype(np.float32)
        np.save(pca_path, pca_values, allow_pickle=False)
    print("Fitting source UMAP 100D -> 30D", flush=True)
    source = UMAP(
        n_components=SOURCE_UMAP_DIMENSIONS,
        n_neighbors=SOURCE_UMAP_NEIGHBOURS,
        min_dist=SOURCE_UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
        n_jobs=1,
        low_memory=True,
        verbose=True,
    ).fit_transform(pca_values).astype(np.float32)
    np.save(source_path, source, allow_pickle=False)
    return source


def nearest_indices(values: np.ndarray, neighbours: int) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=neighbours + 1, metric="euclidean", n_jobs=-1)
    indices = model.fit(values).kneighbors(return_distance=False)
    return indices[:, 1:]


def neighbour_recall(reference: np.ndarray, candidate: np.ndarray, neighbours: int) -> float:
    candidate_indices = nearest_indices(candidate, neighbours)
    return float(
        np.mean(
            [
                len(set(left[:neighbours]).intersection(right)) / neighbours
                for left, right in zip(reference, candidate_indices, strict=True)
            ]
        )
    )


def sampled_distance_correlation(
    source: np.ndarray,
    display: np.ndarray,
    random: np.random.Generator,
) -> float:
    left = random.integers(0, len(source), size=PAIR_SAMPLE_SIZE)
    right = random.integers(0, len(source), size=PAIR_SAMPLE_SIZE)
    different = left != right
    source_distances = np.linalg.norm(source[left[different]] - source[right[different]], axis=1)
    display_distances = np.linalg.norm(display[left[different]] - display[right[different]], axis=1)
    return float(spearmanr(source_distances, display_distances).statistic)


def shape_balance(values: np.ndarray) -> float:
    variances = np.linalg.eigvalsh(np.cov(values, rowvar=False))
    return float(max(0.0, variances[0] / max(variances[-1], 1e-12)))


def evaluate_projection(
    source_sample: np.ndarray,
    display_sample: np.ndarray,
    reference_15: np.ndarray,
    reference_50: np.ndarray,
    random: np.random.Generator,
) -> dict[str, float]:
    recall_15 = neighbour_recall(reference_15, display_sample, 15)
    recall_50 = neighbour_recall(reference_50, display_sample, 50)
    result = {
        "trustworthiness15": float(
            trustworthiness(source_sample, display_sample, n_neighbors=15, metric="euclidean")
        ),
        "neighbourRecall15": recall_15,
        "neighbourRecall50": recall_50,
        "distanceSpearman": sampled_distance_correlation(source_sample, display_sample, random),
        "shapeBalance": shape_balance(display_sample),
    }
    result["qualityScore"] = float(
        0.35 * result["trustworthiness15"]
        + 0.35 * recall_15
        + 0.15 * recall_50
        + 0.10 * max(0.0, result["distanceSpearman"])
        + 0.05 * min(1.0, result["shapeBalance"] * 3)
    )
    return result


def config_key(neighbours: int, min_dist: float) -> str:
    return f"nn{neighbours}_md{str(min_dist).replace('.', 'p')}"


def fit_display(source: np.ndarray, neighbours: int, min_dist: float, seed: int) -> np.ndarray:
    return UMAP(
        n_components=3,
        n_neighbors=neighbours,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
        n_epochs=DISPLAY_EPOCHS,
        low_memory=True,
        verbose=False,
    ).fit_transform(source).astype(np.float32)


def stability_score(projections: list[np.ndarray], sample_indices: np.ndarray) -> float:
    neighbours = [nearest_indices(values[sample_indices], 15) for values in projections]
    scores = []
    for left in range(len(neighbours)):
        for right in range(left + 1, len(neighbours)):
            scores.append(
                np.mean(
                    [
                        len(set(a).intersection(b)) / 15
                        for a, b in zip(neighbours[left], neighbours[right], strict=True)
                    ]
                )
            )
    return float(np.mean(scores))


def normalize_projection(values: np.ndarray) -> tuple[np.ndarray, dict]:
    center = values.mean(axis=0)
    centered = values - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("The selected 3D projection is degenerate")
    normalized = (centered / radius).astype("<f4")
    return normalized, {
        "center": center.astype(float).tolist(),
        "radius": radius,
        "bounds": {
            "min": normalized.min(axis=0).astype(float).tolist(),
            "max": normalized.max(axis=0).astype(float).tolist(),
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-data", type=Path, default=Path("public/data"))
    parser.add_argument("--analysis", type=Path, default=Path("analysis"))
    parser.add_argument("--source-30d", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    public_data = args.public_data.resolve()
    analysis = args.analysis.resolve()
    work = analysis / "3d-work"
    work.mkdir(parents=True, exist_ok=True)
    texts, corpus_hash = load_texts(public_data)

    model_revision = "saved-source"
    if args.source_30d:
        source = np.load(args.source_30d.resolve(), mmap_mode="r")
        if source.shape != (PAPER_COUNT, SOURCE_UMAP_DIMENSIONS):
            raise ValueError(f"Unexpected source 30D shape: {source.shape}")
    else:
        embeddings, model_revision = embed_texts(
            texts,
            work / "specter-embeddings.npy",
            work / "specter-embedding-progress.json",
            args.device,
            args.batch_size,
        )
        source = reduce_to_source_space(embeddings, work)

    if not np.isfinite(source).all():
        raise ValueError("The 30D source contains non-finite values")
    random = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(random.choice(PAPER_COUNT, size=SAMPLE_SIZE, replace=False))
    remaining_indices = np.setdiff1d(np.arange(PAPER_COUNT), sample_indices, assume_unique=True)
    test_indices = np.sort(random.choice(remaining_indices, size=SAMPLE_SIZE, replace=False))
    source_sample = np.asarray(source[sample_indices])
    reference_15 = nearest_indices(source_sample, 15)
    reference_50 = nearest_indices(source_sample, 50)

    map_data = json.loads((public_data / "map.json").read_text(encoding="utf-8"))
    display_2d = np.asarray(map_data["points"], dtype=object)[:, :2].astype(np.float32)
    baseline = evaluate_projection(
        source_sample,
        display_2d[sample_indices],
        reference_15,
        reference_50,
        np.random.default_rng(RANDOM_STATE + 9_000),
    )
    print(f"Existing 2D baseline: {json.dumps(baseline)}", flush=True)

    rows: list[dict] = []
    primary_projections: dict[str, np.ndarray] = {}
    for neighbours, min_dist in DISPLAY_CONFIGS:
        key = config_key(neighbours, min_dist)
        cache = work / f"{key}_seed{RANDOM_STATE}.npy"
        started = perf_counter()
        if cache.exists():
            projection = np.load(cache, mmap_mode="r")
        else:
            projection = fit_display(source, neighbours, min_dist, RANDOM_STATE)
            np.save(cache, projection, allow_pickle=False)
        primary_projections[key] = projection
        metrics = evaluate_projection(
            source_sample,
            np.asarray(projection[sample_indices]),
            reference_15,
            reference_50,
            np.random.default_rng(RANDOM_STATE + neighbours * 100 + round(min_dist * 100)),
        )
        row = {
            "key": key,
            "nNeighbors": neighbours,
            "minDist": min_dist,
            "seed": RANDOM_STATE,
            "fitSeconds": perf_counter() - started,
            **metrics,
        }
        rows.append(row)
        print(f"{key}: {json.dumps(metrics)}", flush=True)

    finalists = sorted(rows, key=lambda row: row["qualityScore"], reverse=True)[:2]
    for row in finalists:
        key = row["key"]
        projections = [np.asarray(primary_projections[key])]
        for seed in (RANDOM_STATE + 1, RANDOM_STATE + 2):
            cache = work / f"{key}_seed{seed}.npy"
            if cache.exists():
                projection = np.load(cache, mmap_mode="r")
            else:
                projection = fit_display(
                    source,
                    int(row["nNeighbors"]),
                    float(row["minDist"]),
                    seed,
                )
                np.save(cache, projection, allow_pickle=False)
            projections.append(np.asarray(projection))
        row["seedNeighbourAgreement15"] = stability_score(projections, sample_indices)
        row["selectionScore"] = (
            0.82 * float(row["qualityScore"])
            + 0.18 * float(row["seedNeighbourAgreement15"])
        )

    selected = max(finalists, key=lambda row: row["selectionScore"])
    selected_projection = np.asarray(primary_projections[selected["key"]])
    test_source = np.asarray(source[test_indices])
    test_reference_15 = nearest_indices(test_source, 15)
    test_reference_50 = nearest_indices(test_source, 50)
    test_metrics = evaluate_projection(
        test_source,
        selected_projection[test_indices],
        test_reference_15,
        test_reference_50,
        np.random.default_rng(RANDOM_STATE + 12_000),
    )
    baseline_test = evaluate_projection(
        test_source,
        display_2d[test_indices],
        test_reference_15,
        test_reference_50,
        np.random.default_rng(RANDOM_STATE + 13_000),
    )
    normalized, normalization = normalize_projection(selected_projection)
    analysis_projection = analysis / "projection-3d.npy"
    np.save(analysis_projection, normalized, allow_pickle=False)
    binary_path = public_data / "projection-3d.bin"
    normalized.tofile(binary_path)
    binary_sha256 = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    metadata = {
        "version": "1.0.0",
        "paperCount": PAPER_COUNT,
        "dimensions": 3,
        "file": "data/projection-3d.bin",
        "format": "little-endian Float32 xyz",
        "corpusSha256": corpus_hash,
        "binarySha256": binary_sha256,
        "source": {
            "model": MODEL_NAME,
            "modelRevision": model_revision,
            "embeddingDimensions": 768,
            "pcaDimensions": PCA_DIMENSIONS,
            "umapDimensions": SOURCE_UMAP_DIMENSIONS,
            "umapNNeighbors": SOURCE_UMAP_NEIGHBOURS,
            "umapMinDist": SOURCE_UMAP_MIN_DIST,
            "umapMetric": "cosine",
        },
        "display": {
            "nNeighbors": int(selected["nNeighbors"]),
            "minDist": float(selected["minDist"]),
            "metric": "euclidean",
            "epochs": DISPLAY_EPOCHS,
            "randomState": RANDOM_STATE,
        },
        "metrics": {
            key: float(value)
            for key, value in selected.items()
            if key in {
                "trustworthiness15",
                "neighbourRecall15",
                "neighbourRecall50",
                "distanceSpearman",
                "shapeBalance",
                "qualityScore",
                "seedNeighbourAgreement15",
                "selectionScore",
            }
        },
        "baseline2d": baseline,
        "testMetrics": test_metrics,
        "baseline2dTest": baseline_test,
        "normalization": normalization,
    }
    write_csv(analysis / "projection-3d-scan.csv", rows)
    (analysis / "projection-3d-selected.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (public_data / "projection-3d.json").write_text(
        json.dumps(metadata, separators=(",", ":")),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
