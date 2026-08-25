# PhilPapers Atlas

An interactive map of 69,400 English-language philosophy papers. Each point is a paper; proximity reflects similarity in the saved SPECTER/PCA/UMAP representation, while colour shows the selected clustering lens.

Live atlas: https://davidfreeborn.github.io/philpapers-atlas/

The atlas is a visual interface to the existing Computational Philosophy Lab analysis. Prajakta's preferred 42-cluster HDBSCAN result remains the default; ten manually named HDBSCAN solutions and a broad 10-cluster k-means solution can be compared without moving the papers.

## Run locally

```bash
npm install
npm run dev:pages
```

## Checks

```bash
npm run lint
npx tsc --noEmit
npm run build:pages
```

## Rebuild the browser data

`scripts/build-map-data.py` converts the saved NumPy outputs and PhilPapers metadata workbook into compact map/search files and lazy detail shards under `public/data/`. `scripts/build-lenses-data.py` converts the reviewed alternative clusterings into a metadata catalogue and 139 KB assignment arrays. See `public/data/README.md` for the required source files.

## Analytical pipeline represented

SPECTER embeddings (768D) → PCA (100D) → UMAP (30D, cosine, 15 neighbours, minimum distance 0) → HDBSCAN (`min_cluster_size=200`, `min_samples=15`). The visible coordinates are a further 2D UMAP of the 30D representation (15 neighbours, minimum distance 0.1).

The preferred view has 42 clusters and labels 21,281 papers as HDBSCAN noise. Alternative HDBSCAN views vary UMAP dimensionality and density parameters; the k-means view assigns every paper. The displayed 2D coordinates remain fixed throughout, so the lenses can be compared directly. Distances and cluster boundaries are exploratory rather than taxonomic claims.

The map uses WebGL 2 to draw the complete point cloud in a single GPU call. Lens changes update existing GPU buffers rather than rebuilding the renderer. Core paper metadata is loaded up front; compact label arrays and smaller abstract-only shards are fetched on demand and cached.
