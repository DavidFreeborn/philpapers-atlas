#!/usr/bin/env python3
"""Scan, evaluate, and export several defensible LDA topic-model lenses."""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import adjusted_rand_score


PAPER_COUNT = 69_400
RANDOM_STATE = 20260825
VALIDATION_SIZE = 6_940
TEST_SIZE = 6_940
TOP_WORDS = 10
TOPIC_COUNTS = (12, 20, 30, 40, 50, 60, 75, 100)
PRIOR_SCAN_COUNTS = (20, 40, 60, 75)
STAGE_ONE_ITERATIONS = 10
STABILITY_ITERATIONS = 15
FINAL_ITERATIONS = 25

DOMAIN_STOP_WORDS = {
    "account", "accounts", "approach", "approaches", "argument", "arguments", "article",
    "argue", "argued", "argues", "book", "chapter", "conclusion", "consider", "considers",
    "discuss", "discusses", "does", "examine", "examines", "explore", "explores",
    "introduction", "new", "paper", "papers", "present", "presents", "press", "propose",
    "proposes", "question", "questions", "reply", "research", "response", "result", "results",
    "review", "section", "study", "suggest", "suggests", "university", "use", "used", "using",
    "view", "views", "way", "work", "works",
}


def load_texts(public_data: Path) -> list[str]:
    search_rows = json.loads((public_data / "search.json").read_text(encoding="utf-8"))
    abstracts: list[str] = []
    for shard in sorted((public_data / "details").glob("*.json")):
        abstracts.extend(row[0] or "" for row in json.loads(shard.read_text(encoding="utf-8")))
    if len(search_rows) != PAPER_COUNT or len(abstracts) != PAPER_COUNT:
        raise ValueError("The title and abstract data do not match the atlas paper count")
    return [
        f"{row[0]}. {row[0]}. {abstract}"
        for row, abstract in zip(search_rows, abstracts, strict=True)
    ]


def configurations() -> list[dict]:
    rows = []
    for topic_count in TOPIC_COUNTS:
        default_prior = 1 / topic_count
        rows.append(
            {
                "topicCount": topic_count,
                "priorVariant": "default",
                "docTopicPrior": default_prior,
                "topicWordPrior": default_prior,
            }
        )
    for topic_count in PRIOR_SCAN_COUNTS:
        default_prior = 1 / topic_count
        rows.extend(
            [
                {
                    "topicCount": topic_count,
                    "priorVariant": "sparse-documents",
                    "docTopicPrior": 0.1 / topic_count,
                    "topicWordPrior": default_prior,
                },
                {
                    "topicCount": topic_count,
                    "priorVariant": "sparse-words",
                    "docTopicPrior": default_prior,
                    "topicWordPrior": 0.01,
                },
                {
                    "topicCount": topic_count,
                    "priorVariant": "sparse-both",
                    "docTopicPrior": 0.1 / topic_count,
                    "topicWordPrior": 0.01,
                },
            ]
        )
    return rows


def config_key(config: dict) -> str:
    return f"k{config['topicCount']}_{config['priorVariant']}"


def topic_indices(components: np.ndarray, count: int = TOP_WORDS) -> np.ndarray:
    return np.argsort(components, axis=1)[:, -count:][:, ::-1]


def npmi_coherence(components: np.ndarray, validation_matrix) -> float:
    binary = validation_matrix.copy().astype(np.int32)
    binary.data.fill(1)
    document_count = binary.shape[0]
    topic_scores = []
    for indices in topic_indices(components):
        subset = binary[:, indices]
        cooccurrence = (subset.T @ subset).toarray().astype(np.float64)
        frequencies = np.diag(cooccurrence)
        pair_scores = []
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                joint = (cooccurrence[left, right] + 1) / (document_count + 1)
                left_probability = (frequencies[left] + 1) / (document_count + 1)
                right_probability = (frequencies[right] + 1) / (document_count + 1)
                pmi = math.log(joint / (left_probability * right_probability))
                pair_scores.append(pmi / -math.log(joint))
        topic_scores.append(float(np.mean(pair_scores)))
    return float(np.mean(topic_scores))


def topic_diversity(components: np.ndarray) -> float:
    indices = topic_indices(components)
    return len(np.unique(indices)) / indices.size


def topic_exclusivity(components: np.ndarray) -> float:
    topic_word = components / components.sum(axis=1, keepdims=True)
    word_totals = topic_word.sum(axis=0)
    scores = []
    for topic_id, indices in enumerate(topic_indices(components)):
        scores.extend((topic_word[topic_id, indices] / word_totals[indices]).tolist())
    return float(np.mean(scores))


def evaluate_distribution(distribution: np.ndarray) -> dict:
    dominant = distribution.argmax(axis=1)
    counts = np.bincount(dominant, minlength=distribution.shape[1])
    prevalence = counts / counts.sum()
    nonzero = prevalence > 0
    usage_entropy = -float(np.sum(prevalence[nonzero] * np.log(prevalence[nonzero])))
    usage_entropy /= math.log(distribution.shape[1])
    document_entropy = -np.sum(distribution * np.log(np.clip(distribution, 1e-12, None)), axis=1)
    return {
        "meanDominantProbability": float(distribution.max(axis=1).mean()),
        "meanNormalisedDocumentEntropy": float(
            np.mean(document_entropy / math.log(distribution.shape[1]))
        ),
        "topicUsageEntropy": usage_entropy,
        "activeTopics": int(np.count_nonzero(counts)),
        "dominantLabels": dominant.astype(np.int16),
    }


def fit_configuration(
    config: dict,
    train_matrix,
    validation_matrix,
    seed: int,
    iterations: int,
    return_model: bool = False,
) -> dict:
    started = perf_counter()
    model = LatentDirichletAllocation(
        n_components=config["topicCount"],
        doc_topic_prior=config["docTopicPrior"],
        topic_word_prior=config["topicWordPrior"],
        learning_method="online",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=iterations,
        batch_size=2_048,
        evaluate_every=-1,
        random_state=seed,
        n_jobs=1,
    ).fit(train_matrix)
    validation_distribution = model.transform(validation_matrix)
    distribution_metrics = evaluate_distribution(validation_distribution)
    result = {
        **config,
        "key": config_key(config),
        "seed": seed,
        "iterations": iterations,
        "heldOutPerplexity": float(model.perplexity(validation_matrix)),
        "npmiCoherence": npmi_coherence(model.components_, validation_matrix),
        "topicDiversity": topic_diversity(model.components_),
        "topicExclusivity": topic_exclusivity(model.components_),
        "meanDominantProbability": distribution_metrics["meanDominantProbability"],
        "meanNormalisedDocumentEntropy": distribution_metrics["meanNormalisedDocumentEntropy"],
        "topicUsageEntropy": distribution_metrics["topicUsageEntropy"],
        "activeTopics": distribution_metrics["activeTopics"],
        "fitSeconds": perf_counter() - started,
        "components": model.components_.astype(np.float32),
        "dominantLabels": distribution_metrics["dominantLabels"],
    }
    if return_model:
        result["model"] = model
    return result


def normalise(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    span = float(values.max() - minimum)
    return np.ones_like(values) if span == 0 else (values - minimum) / span


def preliminary_scores(results: list[dict]) -> None:
    inverse_perplexity = 1 - normalise(np.array([row["heldOutPerplexity"] for row in results]))
    coherence = normalise(np.array([row["npmiCoherence"] for row in results]))
    diversity = normalise(np.array([row["topicDiversity"] for row in results]))
    exclusivity = normalise(np.array([row["topicExclusivity"] for row in results]))
    dominance = normalise(np.array([row["meanDominantProbability"] for row in results]))
    usage = normalise(np.array([row["topicUsageEntropy"] for row in results]))
    for index, row in enumerate(results):
        row["preliminaryScore"] = float(
            0.30 * coherence[index]
            + 0.25 * inverse_perplexity[index]
            + 0.15 * diversity[index]
            + 0.15 * exclusivity[index]
            + 0.10 * dominance[index]
            + 0.05 * usage[index]
        )


def granularity_band(topic_count: int) -> str:
    if topic_count <= 30:
        return "broad"
    if topic_count <= 60:
        return "balanced"
    return "detailed"


def component_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = left / np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = right / np.linalg.norm(right, axis=1, keepdims=True)
    similarities = left_norm @ right_norm.T
    rows, columns = linear_sum_assignment(-similarities)
    return float(similarities[rows, columns].mean())


def stability_metrics(fits: list[dict]) -> tuple[float, float, int]:
    topic_scores = []
    assignment_scores = []
    medoid_scores = np.zeros(len(fits))
    for left in range(len(fits)):
        for right in range(left + 1, len(fits)):
            topic_score = component_similarity(fits[left]["components"], fits[right]["components"])
            assignment_score = adjusted_rand_score(
                fits[left]["dominantLabels"], fits[right]["dominantLabels"]
            )
            topic_scores.append(topic_score)
            assignment_scores.append(assignment_score)
            medoid_scores[left] += topic_score
            medoid_scores[right] += topic_score
    return float(np.mean(topic_scores)), float(np.mean(assignment_scores)), int(np.argmax(medoid_scores))


def final_scores(candidates: list[dict]) -> None:
    inverse_perplexity = 1 - normalise(
        np.array([row["meanHeldOutPerplexity"] for row in candidates])
    )
    coherence = normalise(np.array([row["meanNpmiCoherence"] for row in candidates]))
    diversity = normalise(np.array([row["meanTopicDiversity"] for row in candidates]))
    exclusivity = normalise(np.array([row["meanTopicExclusivity"] for row in candidates]))
    topic_stability = normalise(np.array([row["topicStability"] for row in candidates]))
    assignment_stability = normalise(
        np.array([row["assignmentStabilityAri"] for row in candidates])
    )
    dominance = normalise(
        np.array([row["meanSeedDominantProbability"] for row in candidates])
    )
    for index, row in enumerate(candidates):
        row["finalScore"] = float(
            0.22 * coherence[index]
            + 0.18 * inverse_perplexity[index]
            + 0.12 * diversity[index]
            + 0.12 * exclusivity[index]
            + 0.18 * topic_stability[index]
            + 0.13 * assignment_stability[index]
            + 0.05 * dominance[index]
        )


def topic_words_overlap(left: set[str], right: set[str]) -> bool:
    for left_word in left:
        for right_word in right:
            left_singular = left_word[:-1] if left_word.endswith("s") else left_word
            right_singular = right_word[:-1] if right_word.endswith("s") else right_word
            if left_singular == right_singular:
                return True
            shortest = min(len(left_word), len(right_word))
            similarity = SequenceMatcher(None, left_word, right_word).ratio()
            if shortest >= 3 and left_word[:min(4, shortest)] == right_word[:min(4, shortest)]:
                if similarity >= 0.55:
                    return True
            if shortest >= 4 and similarity >= 0.68:
                return True
    return False


def format_term(term: str) -> str:
    acronyms = {"ai": "AI", "ocr": "OCR", "vr": "VR"}
    return " ".join(acronyms.get(word, word.capitalize()) for word in term.split())


def topic_label(terms: list[str], used: set[str]) -> str:
    selected = []
    selected_tokens = []
    for term in terms:
        tokens = {"artificial", "intelligence"} if term == "ai" else set(term.split())
        if any(topic_words_overlap(tokens, existing) for existing in selected_tokens):
            continue
        selected.append(format_term(term))
        selected_tokens.append(tokens)
        if len(selected) == 2:
            break
    label = " · ".join(selected) if selected else "Unlabelled topic"
    if label in used:
        for term in terms:
            candidate = f"{label} · {format_term(term)}"
            if candidate not in used:
                label = candidate
                break
    used.add(label)
    return label


def fit_final_model(config: dict, full_matrix, vocabulary: np.ndarray, seed: int) -> dict:
    model = LatentDirichletAllocation(
        n_components=config["topicCount"],
        doc_topic_prior=config["docTopicPrior"],
        topic_word_prior=config["topicWordPrior"],
        learning_method="online",
        learning_decay=0.7,
        learning_offset=10.0,
        max_iter=FINAL_ITERATIONS,
        batch_size=2_048,
        evaluate_every=-1,
        random_state=seed,
        n_jobs=1,
    ).fit(full_matrix)
    distribution = model.transform(full_matrix)
    labels = distribution.argmax(axis=1).astype(np.int16)
    indices = topic_indices(model.components_, count=14)
    used_labels: set[str] = set()
    topics = []
    for topic_id, word_indices in enumerate(indices):
        terms = vocabulary[word_indices].tolist()
        topics.append(
            {
                "id": topic_id,
                "label": topic_label(terms, used_labels),
                "terms": terms[:10],
                "count": int((labels == topic_id).sum()),
            }
        )
    return {"labels": labels, "topics": topics}


def serialisable_result(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"components", "dominantLabels", "model"}
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-data", type=Path, default=Path("public/data"))
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()

    texts = load_texts(args.public_data.resolve())
    random = np.random.default_rng(RANDOM_STATE)
    order = random.permutation(PAPER_COUNT)
    train_indices = order[: PAPER_COUNT - VALIDATION_SIZE - TEST_SIZE]
    validation_indices = order[PAPER_COUNT - VALIDATION_SIZE - TEST_SIZE: PAPER_COUNT - TEST_SIZE]
    test_indices = order[PAPER_COUNT - TEST_SIZE:]

    vectorizer = CountVectorizer(
        stop_words=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS),
        lowercase=True,
        ngram_range=(1, 2),
        min_df=20,
        max_df=0.5,
        max_features=20_000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
    )
    train_matrix = vectorizer.fit_transform([texts[index] for index in train_indices])
    validation_matrix = vectorizer.transform([texts[index] for index in validation_indices])
    test_matrix = vectorizer.transform([texts[index] for index in test_indices])
    vocabulary = vectorizer.get_feature_names_out()
    print(
        f"Vocabulary {len(vocabulary):,}; training documents {train_matrix.shape[0]:,}; "
        f"non-zero counts {train_matrix.nnz:,}",
        flush=True,
    )

    configs = configurations()
    stage_one = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(fit_configuration)(
            config,
            train_matrix,
            validation_matrix,
            RANDOM_STATE,
            STAGE_ONE_ITERATIONS,
        )
        for config in configs
    )
    preliminary_scores(stage_one)

    stability_candidates = []
    for band in ("broad", "balanced", "detailed"):
        band_rows = [row for row in stage_one if granularity_band(row["topicCount"]) == band]
        stability_candidates.extend(
            sorted(band_rows, key=lambda row: row["preliminaryScore"], reverse=True)[:2]
        )
    extra_fits = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(fit_configuration)(
            {key: candidate[key] for key in (
                "topicCount", "priorVariant", "docTopicPrior", "topicWordPrior"
            )},
            train_matrix,
            validation_matrix,
            seed,
            STABILITY_ITERATIONS,
            True,
        )
        for candidate in stability_candidates
        for seed in (RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2)
    )
    extra_by_key: dict[str, list[dict]] = {}
    for fit in extra_fits:
        extra_by_key.setdefault(fit["key"], []).append(fit)
    seed_for_key = {}
    for candidate in stability_candidates:
        fits = extra_by_key[candidate["key"]]
        topic_stability, assignment_stability, medoid_index = stability_metrics(fits)
        medoid = fits[medoid_index]
        candidate["topicStability"] = topic_stability
        candidate["assignmentStabilityAri"] = assignment_stability
        candidate["meanHeldOutPerplexity"] = float(
            np.mean([fit["heldOutPerplexity"] for fit in fits])
        )
        candidate["meanNpmiCoherence"] = float(
            np.mean([fit["npmiCoherence"] for fit in fits])
        )
        candidate["meanTopicDiversity"] = float(
            np.mean([fit["topicDiversity"] for fit in fits])
        )
        candidate["meanTopicExclusivity"] = float(
            np.mean([fit["topicExclusivity"] for fit in fits])
        )
        candidate["meanSeedDominantProbability"] = float(
            np.mean([fit["meanDominantProbability"] for fit in fits])
        )
        test_distribution = medoid["model"].transform(test_matrix)
        test_distribution_metrics = evaluate_distribution(test_distribution)
        candidate["testHeldOutPerplexity"] = float(medoid["model"].perplexity(test_matrix))
        candidate["testNpmiCoherence"] = npmi_coherence(medoid["components"], test_matrix)
        candidate["testMeanDominantProbability"] = test_distribution_metrics[
            "meanDominantProbability"
        ]
        candidate["testTopicUsageEntropy"] = test_distribution_metrics["topicUsageEntropy"]
        candidate["medoidSeed"] = medoid["seed"]
        seed_for_key[candidate["key"]] = medoid["seed"]
    final_scores(stability_candidates)

    selected = []
    for band in ("broad", "balanced", "detailed"):
        selected.append(
            max(
                (row for row in stability_candidates if granularity_band(row["topicCount"]) == band),
                key=lambda row: row["finalScore"],
            )
        )

    full_matrix = vectorizer.transform(texts)
    final_models = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(fit_final_model)(row, full_matrix, vocabulary, seed_for_key[row["key"]])
        for row in selected
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_lenses = []
    for row, final_model in zip(selected, final_models, strict=True):
        lens_id = f"lda_{row['key'].replace('-', '_')}"
        labels_file = f"{lens_id}-labels.npy"
        np.save(output / labels_file, final_model["labels"], allow_pickle=False)
        selected_lenses.append(
            {
                "id": lens_id,
                "granularity": granularity_band(row["topicCount"]),
                "labelsFile": labels_file,
                "method": {
                    "topicCount": row["topicCount"],
                    "docTopicPrior": row["docTopicPrior"],
                    "topicWordPrior": row["topicWordPrior"],
                    "priorVariant": row["priorVariant"],
                    "learningMethod": "online",
                    "iterations": FINAL_ITERATIONS,
                    "vocabularySize": len(vocabulary),
                    "minDocumentFrequency": 20,
                    "maxDocumentFrequency": 0.5,
                    "ngramRange": [1, 2],
                },
                "metrics": {
                    "heldOutPerplexity": row["testHeldOutPerplexity"],
                    "npmiCoherence": row["testNpmiCoherence"],
                    "topicDiversity": row["meanTopicDiversity"],
                    "topicExclusivity": row["meanTopicExclusivity"],
                    "meanDominantProbability": row["testMeanDominantProbability"],
                    "topicStability": row["topicStability"],
                    "assignmentStabilityAri": row["assignmentStabilityAri"],
                    "finalScore": row["finalScore"],
                },
                "topics": final_model["topics"],
            }
        )

    all_rows = []
    candidate_keys = {row["key"] for row in stability_candidates}
    selected_keys = {row["key"] for row in selected}
    for row in stage_one:
        serialised = serialisable_result(row)
        serialised["stabilityCandidate"] = row["key"] in candidate_keys
        serialised["selected"] = row["key"] in selected_keys
        all_rows.append(serialised)
    fields = sorted({key for row in all_rows for key in row})
    with (output / "lda-scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: row["preliminaryScore"], reverse=True))
    (output / "lda-selected.json").write_text(
        json.dumps(
            {
                "version": "1.0.0",
                "randomState": RANDOM_STATE,
                "trainingDocuments": len(train_indices),
                "validationDocuments": len(validation_indices),
                "testDocumentsReserved": TEST_SIZE,
                "lenses": selected_lenses,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            [
                {
                    "id": lens["id"],
                    "granularity": lens["granularity"],
                    "method": lens["method"],
                    "metrics": lens["metrics"],
                }
                for lens in selected_lenses
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
