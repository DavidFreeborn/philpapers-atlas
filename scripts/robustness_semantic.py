#!/usr/bin/env python3
"""Held-out semantic validation and consensus diagnostics for frozen finalists."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

from robustness_metrics import NOISE, atomic_json, atomic_npy, optimal_label_mapping


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
WORK = STUDY / "work"
SOURCE = WORK / "source"
RESULTS = STUDY / "results"
CONFIG = yaml.safe_load((STUDY / "config.yaml").read_text(encoding="utf-8"))
FINALISTS = yaml.safe_load((STUDY / "finalists.yaml").read_text(encoding="utf-8"))["finalists"]
SEMANTIC = WORK / "semantic"
N_PAPERS = 69_400


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    fields.extend(sorted(set().union(*(row.keys() for row in rows)) - set(fields)))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atlas_records() -> list[dict[str, str]]:
    search = json.loads((ROOT / "public" / "data" / "search.json").read_text(encoding="utf-8"))
    details: list[list[str]] = []
    for path in sorted((ROOT / "public" / "data" / "details").glob("*.json")):
        details.extend(json.loads(path.read_text(encoding="utf-8")))
    if len(search) != N_PAPERS or len(details) != N_PAPERS:
        raise ValueError(f"Atlas metadata length mismatch: search={len(search)}, details={len(details)}")
    records = []
    for compact, detail in zip(search, details, strict=True):
        title, authors, year, url = (str(value or "") for value in compact[:4])
        abstract = str(detail[0] or "") if detail else ""
        records.append(
            {
                "title": title,
                "authors": authors,
                "year": year,
                "url": url,
                "paper_id": url.rsplit("/", 1)[-1],
                "abstract": abstract,
                "text": f"{title}. {abstract}".strip(),
            }
        )
    return records


def deterministic_splits(records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    settings = CONFIG["semantic_validation"]
    seed = int(settings["random_seed"])
    scores = np.fromiter(
        (
            int.from_bytes(
                hashlib.sha256(f"{seed}:{record['paper_id']}".encode("utf-8")).digest()[:8],
                "big",
            )
            for record in records
        ),
        dtype=np.uint64,
        count=N_PAPERS,
    )
    order = np.argsort(scores, kind="stable")
    development_n = int(settings["development_sample"])
    validation_n = int(settings["validation_sample"])
    return (
        np.sort(order[:development_n]).astype(np.int32),
        np.sort(order[development_n : development_n + validation_n]).astype(np.int32),
    )


def normalized_specter(force: bool = False) -> np.ndarray:
    output = SEMANTIC / "specter-normalized.npy"
    if output.exists() and not force:
        result = np.load(output, mmap_mode="r")
        if result.shape == (N_PAPERS, 768):
            return result
    embeddings = np.load(SOURCE / "embeddings.npy", mmap_mode="r")
    english = np.load(SOURCE / "english_indices.npy")
    if embeddings.shape != (73_440, 768) or english.shape != (N_PAPERS,):
        raise ValueError("Unexpected source embedding shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npy.tmp")
    result = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(N_PAPERS, 768))
    for start in range(0, N_PAPERS, 2048):
        block = np.asarray(embeddings[english[start : start + 2048]], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        result[start : start + len(block)] = block / np.maximum(norms, 1e-12)
    result.flush()
    del result
    os.replace(temporary, output)
    return np.load(output, mmap_mode="r")


def without_self(indices: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    output = np.empty((len(queries), k), dtype=np.int32)
    for row, query in enumerate(queries):
        values = indices[row]
        values = values[values != query]
        if len(values) < k:
            raise ValueError(f"Only {len(values)} non-self neighbours for query {query}")
        output[row] = values[:k]
    return output


def build_neighbors(force: bool = False) -> dict:
    from pynndescent import NNDescent

    SEMANTIC.mkdir(parents=True, exist_ok=True)
    indices_path = SEMANTIC / "neighbors-51.npy"
    distances_path = SEMANTIC / "neighbor-distances-51.npy"
    audit_path = SEMANTIC / "neighbor-audit.json"
    development_path = SEMANTIC / "development-indices.npy"
    validation_path = SEMANTIC / "validation-indices.npy"
    if all(path.exists() for path in (indices_path, distances_path, audit_path, development_path, validation_path)) and not force:
        return json.loads(audit_path.read_text(encoding="utf-8"))

    records = atlas_records()
    development, validation = deterministic_splits(records)
    atomic_npy(development_path, development)
    atomic_npy(validation_path, validation)
    x = normalized_specter(force=force)
    # PyNNDescent's angular RP-tree kernels require a writable C-contiguous array;
    # an mmap opened read-only is rejected by Numba even though the algorithm does
    # not intentionally modify the data.
    neighbor_data = np.array(x, dtype=np.float32, order="C", copy=True)
    started = perf_counter()
    index = NNDescent(
        neighbor_data,
        n_neighbors=51,
        metric="cosine",
        random_state=int(CONFIG["semantic_validation"]["random_seed"]),
        n_jobs=-1,
        low_memory=True,
        verbose=True,
    )
    indices, distances = index.neighbor_graph
    atomic_npy(indices_path, indices.astype(np.int32, copy=False))
    atomic_npy(distances_path, distances.astype(np.float32, copy=False))

    audit_n = int(CONFIG["semantic_validation"]["exact_neighbor_audit_sample"])
    audit_rng = np.random.default_rng(int(CONFIG["semantic_validation"]["random_seed"]) + 17)
    audit_queries = np.sort(audit_rng.choice(validation, size=audit_n, replace=False)).astype(np.int32)
    approximate = without_self(indices[audit_queries], audit_queries, 50)
    recalls: dict[int, list[float]] = {15: [], 50: []}
    exact_chunks = 25
    for start in range(0, audit_n, exact_chunks):
        query_ids = audit_queries[start : start + exact_chunks]
        similarities = np.asarray(x[query_ids]) @ np.asarray(x).T
        top = np.argpartition(-similarities, kth=50, axis=1)[:, :51]
        top_scores = np.take_along_axis(similarities, top, axis=1)
        top = np.take_along_axis(top, np.argsort(-top_scores, axis=1), axis=1)
        exact = without_self(top, query_ids, 50)
        for row in range(len(query_ids)):
            for k in recalls:
                recalls[k].append(len(set(approximate[start + row, :k]) & set(exact[row, :k])) / k)
    audit = {
        "index_seconds": perf_counter() - started,
        "papers": N_PAPERS,
        "dimensions": 768,
        "neighbors_stored": 51,
        "audit_queries": audit_n,
        "recall_at_15_mean": float(np.mean(recalls[15])),
        "recall_at_15_min": float(np.min(recalls[15])),
        "recall_at_50_mean": float(np.mean(recalls[50])),
        "recall_at_50_min": float(np.min(recalls[50])),
    }
    atomic_json(audit_path, audit)
    return audit


def centroid_diagnostics(
    x: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    cluster_ids = np.unique(labels[labels != NOISE])
    positions = {int(cluster): position for position, cluster in enumerate(cluster_ids)}
    centroids = np.zeros((len(cluster_ids), x.shape[1]), dtype=np.float64)
    counts = np.zeros(len(cluster_ids), dtype=np.int64)
    train_labels = labels[train_indices]
    for cluster in cluster_ids:
        mask = train_labels == cluster
        position = positions[int(cluster)]
        counts[position] = int(mask.sum())
        centroids[position] = np.asarray(x[train_indices[mask]], dtype=np.float64).mean(axis=0)
    centroids = normalize(centroids).astype(np.float32)
    validation_labels = labels[validation_indices]
    assigned = validation_labels != NOISE
    query = np.asarray(x[validation_indices[assigned]], dtype=np.float32)
    similarities = query @ centroids.T
    own_positions = np.array([positions[int(label)] for label in validation_labels[assigned]])
    own = similarities[np.arange(len(similarities)), own_positions]
    masked = similarities.copy()
    masked[np.arange(len(masked)), own_positions] = -np.inf
    alternative = masked.max(axis=1)
    predictions = cluster_ids[np.argmax(similarities, axis=1)]
    metrics = {
        "specter_centroid_accuracy": float(np.mean(predictions == validation_labels[assigned])),
        "specter_cosine_cohesion_mean": float(np.mean(own)),
        "specter_centroid_margin_mean": float(np.mean(own - alternative)),
        "specter_centroid_margin_positive_pct": float(np.mean(own > alternative) * 100),
    }
    return metrics, cluster_ids, centroids, assigned


def local_purity(
    labels: np.ndarray,
    validation: np.ndarray,
    neighbor_indices: np.ndarray,
    k: int,
) -> tuple[float, float, np.ndarray]:
    neighbors = without_self(neighbor_indices[validation], validation, k)
    focal = labels[validation]
    assigned = focal != NOISE
    same = labels[neighbors] == focal[:, None]
    per_paper = same.mean(axis=1)
    micro = float(per_paper[assigned].mean()) if assigned.any() else 0.0
    cluster_means = [float(per_paper[focal == cluster].mean()) for cluster in np.unique(focal[assigned])]
    macro = float(np.mean(cluster_means)) if cluster_means else 0.0
    return micro, macro, per_paper


def permutation_null(
    labels: np.ndarray,
    validation: np.ndarray,
    neighbors: np.ndarray,
    repeats: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        permuted = rng.permutation(labels)
        focal = permuted[validation]
        assigned = focal != NOISE
        values.append(float(np.mean(permuted[neighbors][assigned] == focal[assigned, None])))
    return values


def npmi_for_terms(matrix: sparse.csr_matrix, term_indices: np.ndarray) -> float:
    if matrix.shape[0] < 2 or len(term_indices) < 2:
        return float("nan")
    binary = (matrix[:, term_indices] > 0).astype(np.int8).toarray()
    probabilities = binary.mean(axis=0)
    values = []
    for left in range(len(term_indices)):
        for right in range(left + 1, len(term_indices)):
            joint = float(np.mean(binary[:, left] & binary[:, right]))
            if joint <= 0 or probabilities[left] <= 0 or probabilities[right] <= 0:
                values.append(-1.0)
            else:
                values.append(float(np.log(joint / (probabilities[left] * probabilities[right])) / -np.log(joint)))
    return float(np.mean(values))


def refit_membership(finalist: dict, reference: np.ndarray) -> tuple[np.ndarray, dict[int, float], float]:
    coordinates = np.load(ROOT / finalist["reference_embedding"], mmap_mode="r")
    if finalist["kind"] == "fixed_display" or finalist.get("hdbscan_implementation") == "sklearn":
        from sklearn.cluster import HDBSCAN

        model = HDBSCAN(
            min_cluster_size=int(finalist["min_cluster_size"]),
            min_samples=int(finalist["min_samples"]),
            metric="euclidean",
            cluster_selection_method="eom",
            algorithm="kd_tree",
            leaf_size=60,
            n_jobs=-1,
            copy=True,
        ).fit(coordinates)
        raw_persistence: dict[int, float] = {}
    else:
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(finalist["min_cluster_size"]),
            min_samples=int(finalist["min_samples"]),
            metric="euclidean",
            cluster_selection_method="eom",
            core_dist_n_jobs=-1,
        ).fit(coordinates)
        raw_persistence = {
            int(cluster): float(value) for cluster, value in enumerate(model.cluster_persistence_)
        }
    ari = float(adjusted_rand_score(reference, model.labels_))
    if ari < 0.98:
        raise ValueError(f"Refit for {finalist['id']} did not reproduce labels: ARI={ari}")
    if ari < 0.999:
        return np.full(len(reference), np.nan, dtype=np.float32), {}, ari
    mapping = optimal_label_mapping(reference, np.asarray(model.labels_))
    persistence = {
        mapping[model_cluster]: value
        for model_cluster, value in raw_persistence.items()
        if model_cluster in mapping
    }
    return np.asarray(model.probabilities_), persistence, ari


def command_prepare(args: argparse.Namespace) -> None:
    audit = build_neighbors(force=args.force)
    print(json.dumps(audit, indent=2))


def command_evaluate(_: argparse.Namespace) -> None:
    records = atlas_records()
    development = np.load(SEMANTIC / "development-indices.npy")
    validation = np.load(SEMANTIC / "validation-indices.npy")
    held_out = np.zeros(N_PAPERS, dtype=bool)
    held_out[validation] = True
    train = np.flatnonzero(~held_out).astype(np.int32)
    x = normalized_specter()
    neighbor_indices = np.load(SEMANTIC / "neighbors-51.npy", mmap_mode="r")

    texts = [record["text"] for record in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_df=0.85,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    started = perf_counter()
    train_matrix = vectorizer.fit_transform([texts[index] for index in train]).tocsr()
    validation_matrix = vectorizer.transform([texts[index] for index in validation]).tocsr()
    terms = vectorizer.get_feature_names_out()

    summary_rows: list[dict] = []
    cluster_rows: list[dict] = []
    term_rows: list[dict] = []
    representative_rows: list[dict] = []
    null_rows: list[dict] = []
    rng = np.random.default_rng(int(CONFIG["semantic_validation"]["random_seed"]))

    for finalist_number, finalist in enumerate(FINALISTS):
        finalist_id = finalist["id"]
        print(f"Evaluating {finalist_id}", flush=True)
        labels = np.load(ROOT / finalist["reference_labels"])
        cluster_ids = np.unique(labels[labels != NOISE])
        metrics, specter_cluster_ids, specter_centroids, validation_assigned = centroid_diagnostics(
            x, labels, train, validation
        )
        if not np.array_equal(cluster_ids, specter_cluster_ids):
            raise ValueError("Centroid cluster order mismatch")
        probabilities, persistence, refit_ari = refit_membership(finalist, labels)

        purity_by_k: dict[int, np.ndarray] = {}
        for k in CONFIG["semantic_validation"]["nearest_neighbors"]:
            micro, macro, per_paper = local_purity(labels, validation, neighbor_indices, int(k))
            metrics[f"specter_knn{k}_purity_micro"] = micro
            metrics[f"specter_knn{k}_purity_macro"] = macro
            purity_by_k[int(k)] = per_paper

        silhouette_mask = labels[validation] != NOISE
        silhouette_indices = validation[silhouette_mask]
        sample_n = min(2000, len(silhouette_indices))
        metrics["specter_silhouette_cosine"] = float(
            silhouette_score(
                np.asarray(x[silhouette_indices]),
                labels[silhouette_indices],
                metric="cosine",
                sample_size=sample_n,
                random_state=int(CONFIG["semantic_validation"]["random_seed"]),
            )
        )
        metrics["specter_silhouette_sample_n"] = sample_n

        train_labels = labels[train]
        validation_labels = labels[validation]
        tfidf_centroids = np.zeros((len(cluster_ids), train_matrix.shape[1]), dtype=np.float32)
        tfidf_counts = np.zeros(len(cluster_ids), dtype=np.int64)
        for position, cluster in enumerate(cluster_ids):
            mask = train_labels == cluster
            tfidf_counts[position] = int(mask.sum())
            if mask.any():
                tfidf_centroids[position] = np.asarray(train_matrix[mask].mean(axis=0)).ravel()
        tfidf_centroids = normalize(tfidf_centroids)
        assigned_rows = np.flatnonzero(validation_labels != NOISE)
        tfidf_similarities = validation_matrix[assigned_rows] @ tfidf_centroids.T
        tfidf_similarities = np.asarray(tfidf_similarities)
        own_positions = np.searchsorted(cluster_ids, validation_labels[assigned_rows])
        own_tfidf = tfidf_similarities[np.arange(len(assigned_rows)), own_positions]
        masked = tfidf_similarities.copy()
        masked[np.arange(len(masked)), own_positions] = -np.inf
        tfidf_alternative = masked.max(axis=1)
        tfidf_predictions = cluster_ids[np.argmax(tfidf_similarities, axis=1)]
        metrics.update(
            {
                "tfidf_centroid_accuracy": float(np.mean(tfidf_predictions == validation_labels[assigned_rows])),
                "tfidf_centroid_cohesion_mean": float(np.mean(own_tfidf)),
                "tfidf_centroid_margin_mean": float(np.mean(own_tfidf - tfidf_alternative)),
                "tfidf_centroid_margin_positive_pct": float(np.mean(own_tfidf > tfidf_alternative) * 100),
            }
        )

        total_term_weight = tfidf_centroids.sum(axis=0)
        cluster_npmi: dict[int, float] = {}
        cluster_exclusivity: dict[int, float] = {}
        for position, cluster in enumerate(cluster_ids):
            top_count = min(12, tfidf_centroids.shape[1])
            top = np.argpartition(-tfidf_centroids[position], top_count - 1)[:top_count]
            top = top[np.argsort(-tfidf_centroids[position, top])]
            exclusivities = tfidf_centroids[position, top] / np.maximum(total_term_weight[top], 1e-12)
            training_cluster = train_matrix[train_labels == cluster]
            coherence = npmi_for_terms(training_cluster, top[:10])
            cluster_npmi[int(cluster)] = coherence
            cluster_exclusivity[int(cluster)] = float(np.mean(exclusivities))
            for rank, (term_index, exclusivity) in enumerate(zip(top, exclusivities, strict=True), start=1):
                term_rows.append(
                    {
                        "finalist": finalist_id,
                        "cluster": int(cluster),
                        "rank": rank,
                        "term": terms[term_index],
                        "tfidf_centroid_weight": float(tfidf_centroids[position, term_index]),
                        "term_exclusivity": float(exclusivity),
                    }
                )

        metrics["text_npmi_macro"] = float(np.nanmean(list(cluster_npmi.values())))
        metrics["text_term_exclusivity_macro"] = float(np.mean(list(cluster_exclusivity.values())))

        neighbors50 = without_self(neighbor_indices[validation], validation, 50)
        null_values = permutation_null(
            labels,
            validation,
            neighbors50,
            int(CONFIG["semantic_validation"]["label_permutations"]),
            int(CONFIG["semantic_validation"]["random_seed"]) + finalist_number,
        )
        observed = metrics["specter_knn50_purity_micro"]
        null_mean = float(np.mean(null_values))
        null_sd = float(np.std(null_values, ddof=1))
        metrics["specter_knn50_null_mean"] = null_mean
        metrics["specter_knn50_null_sd"] = null_sd
        metrics["specter_knn50_null_z"] = float((observed - null_mean) / null_sd) if null_sd else float("inf")
        metrics["specter_knn50_null_empirical_p"] = float((1 + np.sum(np.asarray(null_values) >= observed)) / (1 + len(null_values)))
        for repeat, value in enumerate(null_values):
            null_rows.append({"finalist": finalist_id, "repeat": repeat, "knn50_purity": value})

        current_labels = np.load(ROOT / FINALISTS[0]["reference_labels"])
        current_matches: dict[int, tuple[int, float]] = {}
        for cluster in cluster_ids:
            own_mask = labels == cluster
            candidate_ids, counts = np.unique(current_labels[own_mask & (current_labels != NOISE)], return_counts=True)
            if len(candidate_ids):
                best = int(np.argmax(counts))
                candidate = int(candidate_ids[best])
                intersection = int(counts[best])
                union = int(own_mask.sum() + np.count_nonzero(current_labels == candidate) - intersection)
                current_matches[int(cluster)] = (candidate, intersection / union)
            else:
                current_matches[int(cluster)] = (NOISE, 0.0)

        validation_specter_own: dict[int, list[float]] = defaultdict(list)
        assigned_labels = validation_labels[validation_assigned]
        validation_vectors = np.asarray(x[validation[validation_assigned]])
        own_sims = np.sum(validation_vectors * specter_centroids[np.searchsorted(cluster_ids, assigned_labels)], axis=1)
        for cluster, value in zip(assigned_labels, own_sims, strict=True):
            validation_specter_own[int(cluster)].append(float(value))

        for position, cluster in enumerate(cluster_ids):
            full_mask = labels == cluster
            validation_mask = validation_labels == cluster
            best_current, overlap = current_matches[int(cluster)]
            cluster_rows.append(
                {
                    "finalist": finalist_id,
                    "cluster": int(cluster),
                    "size": int(full_mask.sum()),
                    "coverage_share": float(full_mask.mean()),
                    "persistence": persistence.get(int(cluster), ""),
                    "membership_mean": (
                        float(np.nanmean(probabilities[full_mask]))
                        if np.isfinite(probabilities[full_mask]).any()
                        else ""
                    ),
                    "validation_n": int(validation_mask.sum()),
                    "specter_knn15_purity": float(purity_by_k[15][validation_mask].mean()) if validation_mask.any() else "",
                    "specter_knn50_purity": float(purity_by_k[50][validation_mask].mean()) if validation_mask.any() else "",
                    "specter_cosine_cohesion": float(np.mean(validation_specter_own[int(cluster)])) if validation_specter_own[int(cluster)] else "",
                    "text_npmi": cluster_npmi[int(cluster)],
                    "text_term_exclusivity": cluster_exclusivity[int(cluster)],
                    "best_current_cluster": best_current,
                    "best_current_jaccard": overlap,
                }
            )

            member_indices = np.flatnonzero(full_mask)
            similarities = np.asarray(x[member_indices]) @ specter_centroids[position]
            representatives = member_indices[np.argsort(-similarities)[:3]]
            random_examples = rng.choice(member_indices, size=min(2, len(member_indices)), replace=False)
            for role, chosen in (("centroid", representatives), ("random", random_examples)):
                for rank, paper_index in enumerate(chosen, start=1):
                    record = records[int(paper_index)]
                    representative_rows.append(
                        {
                            "finalist": finalist_id,
                            "cluster": int(cluster),
                            "role": role,
                            "rank": rank,
                            "paper_index": int(paper_index),
                            "title": record["title"],
                            "authors": record["authors"],
                            "year": record["year"],
                            "url": record["url"],
                            "specter_centroid_similarity": float(similarities[np.searchsorted(member_indices, paper_index)]),
                        }
                    )

        summary_rows.append(
            {
                "finalist": finalist_id,
                "clusters": int(len(cluster_ids)),
                "noise_pct": float(np.mean(labels == NOISE) * 100),
                "validation_n": len(validation),
                "validation_assigned_n": int(validation_assigned.sum()),
                "refit_ari": refit_ari,
                **metrics,
            }
        )

    write_csv(RESULTS / "semantic-summary.csv", summary_rows)
    write_csv(RESULTS / "semantic-clusters.csv", cluster_rows)
    write_csv(RESULTS / "semantic-terms.csv", term_rows)
    write_csv(RESULTS / "semantic-null.csv", null_rows)
    write_csv(RESULTS / "representative-papers.csv", representative_rows)
    atomic_json(
        RESULTS / "semantic-validation-manifest.json",
        {
            "development_sample": len(development),
            "validation_sample": len(validation),
            "validation_selected_before_semantic_evaluation": True,
            "finalists_frozen_before_semantic_evaluation": True,
            "tfidf_training_documents": len(train),
            "tfidf_features": int(train_matrix.shape[1]),
            "tfidf_fit_and_transform_seconds": perf_counter() - started,
            "silhouette_sample_cap": 2000,
            "text_coherence_basis": "training documents; validation is reserved for centroid separation",
        },
    )
    print(json.dumps(summary_rows, indent=2))


def resample_paths(finalist: dict, repeat: int, strategy: str = "scaled") -> tuple[Path, Path]:
    if finalist["kind"] == "fixed_display":
        directory = WORK / "runs" / "resampling-fixed-display" / f"repeat{repeat:02d}"
        return directory / "indices.npy", directory / f"labels-{strategy}.npy"
    directory = WORK / "remote-linux" / "full-resampling" / f"repeat{repeat:02d}"
    return directory / "indices.npy", directory / f"labels-{finalist['id']}-{strategy}.npy"


def command_consensus(_: argparse.Namespace) -> None:
    validation = np.load(SEMANTIC / "validation-indices.npy")
    repeats = int(CONFIG["resampling"]["repeats"])
    paper_rows: list[dict] = []
    cluster_rows: list[dict] = []
    coassignment_rows: list[dict] = []
    rng = np.random.default_rng(int(CONFIG["semantic_validation"]["random_seed"]) + 991)

    for finalist in FINALISTS:
        reference = np.load(ROOT / finalist["reference_labels"])
        correct = np.zeros(N_PAPERS, dtype=np.int16)
        seen = np.zeros(N_PAPERS, dtype=np.int16)
        assigned_consistent = np.zeros(N_PAPERS, dtype=np.int16)
        validation_position = {int(value): position for position, value in enumerate(validation)}
        candidate_on_validation: list[np.ndarray] = []
        present_on_validation: list[np.ndarray] = []
        for repeat in range(repeats):
            indices_path, labels_path = resample_paths(finalist, repeat)
            indices = np.load(indices_path)
            candidate = np.load(labels_path)
            mapping = optimal_label_mapping(reference[indices], candidate)
            mapped = np.full(len(candidate), NOISE, dtype=np.int32)
            for source, target in mapping.items():
                mapped[candidate == source] = target
            seen[indices] += 1
            correct[indices] += mapped == reference[indices]
            assigned_consistent[indices] += (mapped == reference[indices]) & (mapped != NOISE)

            values = np.full(len(validation), -32768, dtype=np.int32)
            positions = np.array([validation_position.get(int(index), -1) for index in indices])
            keep = positions >= 0
            values[positions[keep]] = candidate[keep]
            candidate_on_validation.append(values)
            present_on_validation.append(values != -32768)

        consistency = np.divide(correct, seen, out=np.zeros(N_PAPERS, dtype=float), where=seen > 0)
        for cluster in np.unique(reference[reference != NOISE]):
            members = reference == cluster
            values = consistency[members & (seen > 0)]
            cluster_rows.append(
                {
                    "finalist": finalist["id"],
                    "cluster": int(cluster),
                    "size": int(members.sum()),
                    "paper_consistency_mean": float(values.mean()),
                    "paper_consistency_median": float(np.median(values)),
                    "paper_consistency_p10": float(np.quantile(values, 0.10)),
                    "paper_consistency_below_half_pct": float(np.mean(values < 0.5) * 100),
                }
            )
        paper_rows.append(
            {
                "finalist": finalist["id"],
                "pca_refitted": finalist["kind"] != "fixed_display",
                "conditional_on_fixed_projection": finalist["kind"] == "fixed_display",
                "papers_seen": int(np.sum(seen > 0)),
                "mean_resamples_seen": float(seen.mean()),
                "assignment_consistency_mean": float(consistency[seen > 0].mean()),
                "assignment_consistency_p10": float(np.quantile(consistency[seen > 0], 0.10)),
                "assignment_consistency_below_half_pct": float(np.mean(consistency[seen > 0] < 0.5) * 100),
            }
        )

        assigned_validation = validation[reference[validation] != NOISE]
        within_pairs: list[tuple[int, int]] = []
        for cluster in np.unique(reference[assigned_validation]):
            members = assigned_validation[reference[assigned_validation] == cluster]
            if len(members) >= 2:
                for _ in range(min(100, len(members) * 2)):
                    left, right = rng.choice(members, size=2, replace=False)
                    within_pairs.append((validation_position[int(left)], validation_position[int(right)]))
        between_pairs: list[tuple[int, int]] = []
        while len(between_pairs) < min(5000, len(within_pairs)):
            left, right = rng.choice(assigned_validation, size=2, replace=False)
            if reference[left] != reference[right]:
                between_pairs.append((validation_position[int(left)], validation_position[int(right)]))

        for pair_type, pairs in (("within_reference", within_pairs), ("between_reference", between_pairs)):
            agreements, denominators = [], []
            for left, right in pairs:
                same = 0
                available = 0
                for values, present in zip(candidate_on_validation, present_on_validation, strict=True):
                    if present[left] and present[right]:
                        available += 1
                        same += values[left] != NOISE and values[left] == values[right]
                if available:
                    agreements.append(same / available)
                    denominators.append(available)
            coassignment_rows.append(
                {
                    "finalist": finalist["id"],
                    "pca_refitted": finalist["kind"] != "fixed_display",
                    "conditional_on_fixed_projection": finalist["kind"] == "fixed_display",
                    "pair_type": pair_type,
                    "pairs": len(agreements),
                    "mean_resamples_per_pair": float(np.mean(denominators)),
                    "coassignment_mean": float(np.mean(agreements)),
                    "coassignment_median": float(np.median(agreements)),
                    "coassignment_p10": float(np.quantile(agreements, 0.10)),
                    "coassignment_p90": float(np.quantile(agreements, 0.90)),
                }
            )

    write_csv(RESULTS / "consensus-paper-summary.csv", paper_rows)
    write_csv(RESULTS / "consensus-cluster-summary.csv", cluster_rows)
    write_csv(RESULTS / "consensus-coassignment-summary.csv", coassignment_rows)
    print(json.dumps({"papers": paper_rows, "coassignment": coassignment_rows}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(function=command_prepare)
    evaluate = commands.add_parser("evaluate")
    evaluate.set_defaults(function=command_evaluate)
    consensus = commands.add_parser("consensus")
    consensus.set_defaults(function=command_consensus)
    return parser


def main() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
