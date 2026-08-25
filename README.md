# PhilPapers Atlas

An interactive map of 69,400 English-language philosophy papers. Each point is a paper; proximity reflects similarity in the saved SPECTER/PCA/UMAP representation, while colour shows the selected clustering lens.

Live atlas: https://davidfreeborn.github.io/philpapers-atlas/

The atlas is a visual interface to the existing Computational Philosophy Lab analysis. Prajakta's preferred 42-cluster HDBSCAN result remains the default; ten saved high-dimensional HDBSCAN solutions, an optimised HDBSCAN fit on the fixed 2D display, a 10-cluster k-means solution, and two reviewed LDA topic models can be compared without moving the papers.

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

`scripts/build-map-data.py` converts the saved NumPy outputs and PhilPapers metadata workbook into compact map/search files and lazy detail shards under `public/data/`. `scripts/build-lenses-data.py` converts the reviewed alternative clusterings into a metadata catalogue and 139 KB assignment arrays. `scripts/scan-hdbscan-2d.py` reproduces the 60-configuration parameter scan used to select the display-space lens. `scripts/scan-lda.py` reproduces the LDA model scan and exports its evaluation tables and assignments. See `public/data/README.md` for the required source files.

## Analytical pipeline represented

SPECTER embeddings (768D) → PCA (100D) → UMAP (30D, cosine, 15 neighbours, minimum distance 0) → HDBSCAN (`min_cluster_size=200`, `min_samples=15`). The visible coordinates are a further 2D UMAP of the 30D representation (15 neighbours, minimum distance 0.1).

The preferred view has 42 clusters and labels 21,281 papers as HDBSCAN noise. Alternative HDBSCAN views vary UMAP dimensionality and density parameters. The display-space HDBSCAN lens was selected by balancing coverage, membership strength, 2D silhouette, and agreement with adjacent parameter settings; the k-means view assigns every paper. The displayed 2D coordinates remain fixed throughout, so the lenses can be compared directly. Distances and cluster boundaries are exploratory rather than taxonomic claims.

The LDA scan uses titles (weighted twice) and abstracts, represented by a 14,104-term unigram-and-bigram count vocabulary. It evaluates 20 combinations of topic count and document/topic priors on an 80/10/10 train/validation/test split. Six validation finalists are fitted at three random seeds before the sealed test set is opened. Selection considers held-out perplexity and NPMI coherence, topic diversity and exclusivity, dominant-topic strength, matched topic-word stability, and dominant-assignment agreement. The 20- and 60-topic models passed qualitative review; the 100-topic candidate was rejected because it contained too many small, mixed topics and large generic catch-alls. LDA is a mixed-membership model, so the atlas colour is each paper's highest-probability topic rather than a claim that the paper belongs exclusively to one topic.

The map uses WebGL 2 to draw the complete point cloud in a single GPU call. Lens changes update existing GPU buffers rather than rebuilding the renderer. Core paper metadata is loaded up front; compact label arrays and smaller abstract-only shards are fetched on demand and cached.
