#!/usr/bin/env python3
"""Merge robustness evidence, flag weak clusters, and generate publication-ready figures."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "analysis" / "robustness-study"
RESULTS = STUDY / "results"
FIGURES = RESULTS / "figures"


LABELS = {
    "current_30d_42": "Current 30D (42)",
    "typical_30d_49": "Typical 30D (49)",
    "robust_20d_49": "20D fine (49)",
    "robust_20d_29": "20D coarse (29)",
    "direct_2d_70": "Direct 2D (70)",
    "direct_2d_historical_params": "Direct 2D historical params (63)",
    "historical_display_2d_104": "Historical fixed 2D (104)*",
}
COLORS = {
    "current_30d_42": "#2b6f9f",
    "typical_30d_49": "#61a5c2",
    "robust_20d_49": "#40916c",
    "robust_20d_29": "#74c69d",
    "direct_2d_70": "#d97706",
    "direct_2d_historical_params": "#e9a23b",
    "historical_display_2d_104": "#b45309",
}


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
        }
    )


def seed_stability_figure() -> None:
    local = pd.read_csv(RESULTS / "pca100-seed-summary.csv")
    linux = pd.read_csv(RESULTS / "linux-seed-summary.csv")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    dimension = local["dimension"].to_numpy()
    median = local["cluster_count_median"].to_numpy()
    lower = median - local["cluster_count_min"].to_numpy()
    upper = local["cluster_count_max"].to_numpy() - median
    axes[0].errorbar(dimension, median, yerr=np.vstack([lower, upper]), marker="o", capsize=3, color="#2b6f9f")
    axes[0].set(xlabel="UMAP dimensions", ylabel="Clusters", title="Seed variation in cluster count")
    axes[0].set_xticks(dimension)
    axes[1].plot(dimension, local["pairwise_ari_mean"], marker="o", label="Windows · 10 seeds", color="#2b6f9f")
    axes[1].plot(linux["dimension"], linux["pairwise_ari_mean"], marker="s", label="Linux · 5 seeds", color="#d97706")
    axes[1].set(xlabel="UMAP dimensions", ylabel="Mean pairwise ARI", title="Partition stability across seeds", ylim=(0, 0.8))
    axes[1].set_xticks(dimension)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, "seed-stability-by-dimension")


def hdbscan_grid_figure() -> None:
    grid = pd.read_csv(RESULTS / "pca100-hdbscan-grid.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
    for column, dimension in enumerate((20, 30)):
        subset = grid[(grid["dimension"] == dimension) & (grid["selection_method"] == "eom")]
        for row, (metric, title, fmt) in enumerate(
            (("relative_validity", "DBCV", ".2f"), ("cluster_count", "Cluster count", ".0f"))
        ):
            pivot = subset.pivot(index="min_samples", columns="min_cluster_size", values=metric)
            sns.heatmap(
                pivot,
                ax=axes[row, column],
                cmap="viridis" if metric == "relative_validity" else "mako",
                annot=True,
                fmt=fmt,
                cbar=False,
            )
            axes[row, column].set_title(f"{dimension}D · {title}")
            axes[row, column].set_xlabel("Minimum cluster size")
            axes[row, column].set_ylabel("Minimum samples")
    figure.tight_layout()
    save_figure(figure, "hdbscan-parameter-surfaces")


def umap_sensitivity_figure() -> None:
    cells = pd.read_csv(RESULTS / "umap-sensitivity-cells.csv")
    selections = [
        (30, 200, 15, "30D current HDBSCAN"),
        (20, 300, 50, "20D coarse HDBSCAN"),
        (2, 200, 15, "Direct 2D HDBSCAN"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, (dimension, mcs, samples, title) in zip(axes, selections, strict=True):
        subset = cells[
            (cells["dimension"] == dimension)
            & (cells["min_cluster_size"] == mcs)
            & (cells["min_samples"] == samples)
        ].copy()
        subset["setting"] = subset.apply(lambda row: f"n={int(row.n_neighbors)}\nmin d={row.min_dist:g}", axis=1)
        collapsed = subset["cluster_count_median"] < 5
        axis.bar(
            subset["setting"],
            subset["pairwise_seed_ari_mean"],
            color=np.where(collapsed, "#d97706", "#2b6f9f"),
        )
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=50, labelsize=8)
        axis.set_xlabel("UMAP setting")
        axis.set_ylim(0, 0.85)
    axes[0].set_ylabel("Mean pairwise seed ARI")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2b6f9f", label="Median ≥5 clusters"),
        plt.Rectangle((0, 0), 1, 1, color="#d97706", label="Collapsed: median <5 clusters"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    save_figure(figure, "umap-parameter-sensitivity")


def primary_resampling_summary() -> pd.DataFrame:
    full = pd.read_csv(RESULTS / "full-resampling-summary.csv")
    full = full[full["strategy"] == "scaled"].drop(columns=["strategy"])
    conditional = pd.read_csv(RESULTS / "resampling-summary.csv")
    conditional = conditional[
        (conditional["strategy"] == "scaled")
        & (conditional["finalist"] == "historical_display_2d_104")
    ].drop(columns=["strategy"])
    conditional["pca_refitted"] = False
    return pd.concat([full, conditional], ignore_index=True, sort=False)


def primary_cluster_stability() -> pd.DataFrame:
    full = pd.read_csv(RESULTS / "full-resampling-cluster-stability.csv")
    full = full[full["strategy"] == "scaled"].drop(columns=["strategy"])
    conditional = pd.read_csv(RESULTS / "resampling-cluster-stability.csv")
    conditional = conditional[
        (conditional["strategy"] == "scaled")
        & (conditional["finalist"] == "historical_display_2d_104")
    ].drop(columns=["strategy"])
    return pd.concat([full, conditional], ignore_index=True, sort=False).rename(
        columns={"full_size": "size"}
    )


def finalist_figure() -> None:
    resampling = primary_resampling_summary().set_index("finalist")
    semantic = pd.read_csv(RESULTS / "semantic-summary.csv").set_index("finalist")
    order = list(LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    panels = [
        (resampling, "cluster_jaccard_weighted_mean", "Subsample cluster Jaccard", (0, 1)),
        (resampling, "ari_all_mean", "Subsample ARI", (0, 1)),
        (semantic, "specter_knn50_purity_micro", "SPECTER 50-NN label purity", (0, 1)),
        (semantic, "tfidf_centroid_accuracy", "Held-out TF–IDF centroid accuracy", (0, 1)),
    ]
    for axis, (data, metric, title, limits) in zip(axes.ravel(), panels, strict=True):
        values = [data.loc[item, metric] for item in order]
        bars = axis.barh([LABELS[item] for item in order], values, color=[COLORS[item] for item in order])
        bars[-1].set_hatch("///")
        bars[-1].set_edgecolor("white")
        axis.set_title(title)
        axis.set_xlim(*limits)
        axis.invert_yaxis()
    figure.text(
        0.5,
        0.01,
        "* Historical resampling holds one fitted 2D projection fixed; its structural bars are conditional, not full-path evidence.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, "finalist-evidence")


def cluster_figure(audit: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 6.2))
    for finalist, subset in audit.groupby("finalist", sort=False):
        conditional = finalist == "historical_display_2d_104"
        axis.scatter(
            subset["subsample_jaccard_mean"],
            subset["specter_knn50_purity"],
            s=np.clip(np.sqrt(subset["size"]) * 3, 12, 100),
            alpha=0.58,
            color=COLORS[finalist],
            label=LABELS[finalist],
            marker="x" if conditional else "o",
            linewidth=1 if conditional else 0,
        )
    axis.axvline(0.60, color="#555", linestyle="--", linewidth=1)
    axis.text(0.605, 0.035, "review threshold", rotation=90, color="#555", fontsize=8, va="bottom")
    axis.set(
        xlabel="Mean best-match Jaccard under 80% resampling",
        ylabel="Held-out SPECTER 50-NN label purity",
        title="Cluster-level structural and semantic evidence",
        xlim=(0, 1.02),
        ylim=(0, 1.02),
    )
    axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.text(
        0.5,
        0.01,
        "Marker area represents cluster size. * Fixed-projection historical values are conditional.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(figure, "cluster-stability-and-coherence")


def merge_evidence() -> pd.DataFrame:
    resampling = primary_resampling_summary()
    semantic = pd.read_csv(RESULTS / "semantic-summary.csv")
    consensus = pd.read_csv(RESULTS / "consensus-paper-summary.csv")
    evidence = semantic.merge(resampling, on="finalist", suffixes=("_semantic", "_resample")).merge(consensus, on="finalist")
    evidence.insert(1, "label", evidence["finalist"].map(LABELS))
    evidence.to_csv(RESULTS / "finalist-evidence.csv", index=False, float_format="%.8f")

    clusters = pd.read_csv(RESULTS / "semantic-clusters.csv")
    subsamples = primary_cluster_stability()
    paper_consensus = pd.read_csv(RESULTS / "consensus-cluster-summary.csv")
    audit = clusters.merge(subsamples, on=["finalist", "cluster", "size"], how="left").merge(
        paper_consensus, on=["finalist", "cluster", "size"], how="left", suffixes=("", "_consensus")
    )
    terms = pd.read_csv(RESULTS / "semantic-terms.csv")
    top_terms = (
        terms[terms["rank"] <= 5]
        .sort_values(["finalist", "cluster", "rank"])
        .groupby(["finalist", "cluster"])["term"]
        .agg(lambda values: "; ".join(values.astype(str)))
        .rename("top_terms")
        .reset_index()
    )
    representatives = pd.read_csv(RESULTS / "representative-papers.csv")
    representative_titles = (
        representatives[(representatives["role"] == "centroid") & (representatives["rank"] <= 3)]
        .sort_values(["finalist", "cluster", "rank"])
        .groupby(["finalist", "cluster"])["title"]
        .agg(lambda values: " | ".join(values.astype(str)))
        .rename("representative_titles")
        .reset_index()
    )
    audit = audit.merge(top_terms, on=["finalist", "cluster"], how="left").merge(
        representative_titles, on=["finalist", "cluster"], how="left"
    )
    catalog = json.loads((ROOT / "public" / "data" / "lenses" / "catalog.json").read_text(encoding="utf-8"))
    current_lens = next(item for item in catalog["lenses"] if item["id"] == "pca100_u30_mcs200_ms15")
    current_names = {int(item["id"]): item["label"] for item in current_lens["clusters"]}
    audit.insert(
        2,
        "cluster_descriptor",
        [
            current_names.get(int(cluster), "")
            if finalist == "current_30d_42"
            else " / ".join(str(terms).split("; ")[:3])
            for finalist, cluster, terms in audit[["finalist", "cluster", "top_terms"]].itertuples(index=False, name=None)
        ],
    )
    audit["knn50_lift_over_cluster_share"] = audit["specter_knn50_purity"] - audit["coverage_share"]
    audit["weak_resampling"] = audit["subsample_jaccard_mean"] < 0.60
    audit["weak_specter_locality"] = audit["specter_knn50_purity"] < 0.40
    audit["low_text_coherence"] = audit["text_npmi"] < -0.10
    audit["weak_semantic_boundary"] = audit[["weak_specter_locality", "low_text_coherence"]].any(axis=1)
    audit["unstable_papers"] = audit["paper_consistency_mean"] < 0.60
    audit["review_flag"] = audit[["weak_resampling", "weak_semantic_boundary", "unstable_papers"]].any(axis=1)
    audit.to_csv(RESULTS / "cluster-audit.csv", index=False, float_format="%.8f")
    return audit


def validation_manifest(audit: pd.DataFrame) -> None:
    required = [
        "pca100-seed-summary.csv",
        "pca100-hdbscan-grid.csv",
        "pca50-seed-summary.csv",
        "linux-seed-summary.csv",
        "cross-platform-seeds.csv",
        "umap-sensitivity-cells.csv",
        "full-resampling-summary.csv",
        "semantic-summary.csv",
        "semantic-null.csv",
        "consensus-coassignment-summary.csv",
        "finalist-evidence.csv",
        "cluster-audit.csv",
    ]
    checks = {}
    for name in required:
        path = RESULTS / name
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        checks[name] = {"rows": len(frame), "columns": len(frame.columns), "missing_cells": int(frame.isna().sum().sum())}
    if len(audit) < 250:
        raise ValueError("Cluster audit unexpectedly short")
    temporary = RESULTS / "result-integrity.json.tmp"
    temporary.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, RESULTS / "result-integrity.json")


def main() -> None:
    style()
    audit = merge_evidence()
    seed_stability_figure()
    hdbscan_grid_figure()
    umap_sensitivity_figure()
    finalist_figure()
    cluster_figure(audit)
    validation_manifest(audit)
    print(f"Wrote {len(audit)} cluster-audit rows and five figures to {FIGURES}")


if __name__ == "__main__":
    main()
