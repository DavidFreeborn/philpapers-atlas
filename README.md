# PhilPapers Atlas

An interactive map of 69,400 English-language philosophy papers. Each point is a paper; proximity reflects similarity in the saved SPECTER/PCA/UMAP representation, while colour shows the saved HDBSCAN cluster assignment.

The atlas is a visual interface to the existing Computational Philosophy Lab analysis. It does not refit the model or alter Prajakta's selected result.

## Run locally

```bash
npm install
npm run dev
```

## Checks

```bash
npm run lint
npx tsc --noEmit
npm run build
```

## Rebuild the browser data

`scripts/build-map-data.py` converts the saved NumPy outputs and PhilPapers metadata workbook into compact map/search files and lazy detail shards under `public/data/`. See `public/data/README.md` for the required source files.

## Analytical pipeline represented

SPECTER embeddings (768D) → PCA (100D) → UMAP (30D, cosine, 15 neighbours, minimum distance 0) → HDBSCAN (`min_cluster_size=200`, `min_samples=15`). The visible coordinates are a further 2D UMAP of the 30D representation (15 neighbours, minimum distance 0.1).

This is a candidate research map: 42 clusters are shown, with 21,281 papers labelled as HDBSCAN noise. Distances and cluster boundaries are exploratory rather than taxonomic claims.
