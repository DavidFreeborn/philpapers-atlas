# PhilPapers Atlas

An interactive map of 69,400 English-language philosophy papers. Each point is a paper; proximity reflects similarity in the SPECTER/PCA/UMAP representation, while colour shows the selected analytical lens. The fixed 2D projection is the default, with a lazy-loaded 3D view for spatial exploration.

Live atlas: https://davidfreeborn.github.io/philpapers-atlas/

The default lens is a 42-cluster HDBSCAN solution fitted in a 30-dimensional UMAP representation. Alternative high-dimensional HDBSCAN solutions, an optimised HDBSCAN fit on the fixed 2D projection, a 10-cluster k-means solution, two evaluated LDA topic models, publication type, and publication period can be compared without moving the papers. Selected paper sets persist between lenses, and the correlation workspace compares every lens pair using bias-corrected Cramér’s V and Pearson residuals.

## Run locally

```bash
npm install
npm run dev:pages
```

## Checks

```bash
npm run lint
npx tsc --noEmit
npm run test:search
npm run test:3d
npm run test:3d:data
npm run test:lenses
npm run test:exploration
npm run test:associations
npm run build:pages
```

## Rebuild the browser data

`scripts/build-map-data.py` converts the saved NumPy outputs and PhilPapers metadata workbook into compact map/search files and lazy detail shards under `public/data/`. `scripts/build-lenses-data.py` converts the reviewed alternative clusterings into a metadata catalogue and 139 KB assignment arrays. `scripts/build-association-data.py` aligns the two metadata lenses and precomputes the cross-lens association overview. `scripts/build-exploration-data.py` exports SPECTER nearest-neighbour shards and the ten papers nearest each cluster's SPECTER centroid. `scripts/scan-hdbscan-2d.py` reproduces the 60-configuration parameter scan used to select the display-space lens. `scripts/scan-lda.py` reproduces the LDA model scan and exports its evaluation tables and assignments. `scripts/build-3d-projection.py` rebuilds and evaluates the optional 3D projection. See `public/data/README.md` for the required source files.

## Analytical pipeline represented

SPECTER embeddings (768D) → PCA (100D) → UMAP (30D, cosine, 15 neighbours, minimum distance 0) → HDBSCAN (`min_cluster_size=200`, `min_samples=15`). The default visible coordinates are a further 2D UMAP of the 30D representation (15 neighbours, minimum distance 0.1).

The optional display is an independently fitted 3D UMAP from a matching reconstruction of that 30D pipeline. Ten combinations of neighbourhood size and minimum distance were compared on a fixed 5,000-paper sample. The two leading candidates were then checked across three random seeds. The selected fit uses 50 neighbours and minimum distance 0.05. On a disjoint 5,000-paper holdout it achieved trustworthiness@15 of 0.983, neighbour recall@15 of 0.628, and neighbour recall@50 of 0.689. The corresponding values for the fixed 2D projection were 0.769, 0.123, and 0.164. These figures assess projection fidelity to the reconstructed 30D source; they do not score the clustering lenses.

The default view has 42 clusters and labels 21,281 papers as HDBSCAN noise. Alternative HDBSCAN views vary UMAP dimensionality and density parameters. The display-space HDBSCAN lens was selected by balancing coverage, membership strength, 2D silhouette, and agreement with adjacent parameter settings; the k-means view assigns every paper. The displayed 2D coordinates remain fixed throughout, so the lenses can be compared directly. Distances and cluster boundaries are exploratory rather than taxonomic claims.

The LDA scan uses titles (weighted twice) and abstracts, represented by a 14,104-term unigram-and-bigram count vocabulary. It evaluates 20 combinations of topic count and document/topic priors on an 80/10/10 train/validation/test split. Six validation finalists are fitted at three random seeds before the sealed test set is opened. Selection considers held-out perplexity and NPMI coherence, topic diversity and exclusivity, dominant-topic strength, matched topic-word stability, and dominant-assignment agreement. The 20- and 60-topic models passed qualitative review; the 100-topic candidate was rejected because it contained too many small, mixed topics and large generic catch-alls. LDA is a mixed-membership model, so the atlas colour is each paper's highest-probability topic rather than a claim that the paper belongs exclusively to one topic.

The map uses WebGL 2 to draw the complete point cloud in a single GPU call. A dynamic per-point GPU mask supports single-cluster, multi-cluster, retained cross-lens selections, and semantic-neighbour emphasis without rebuilding the renderer. The 813 KiB 3D coordinate file is fetched only when 3D is first selected. Camera changes render on demand rather than through a permanent animation loop, and 3D paper picking uses a capped off-screen GPU identity buffer. Lens changes update existing GPU buffers rather than rebuilding the renderer. Core paper metadata is loaded up front; compact label arrays, abstract shards, and 30 KB semantic-neighbour shards are fetched on demand and cached. The association overview is precomputed; a detailed contingency matrix loads only the two selected label arrays. Search normalises punctuation and diacritics, recognises initials and singular/plural variants, tolerates one small spelling error, and uses order-independent token matching, so author names work in either display order.

The source workbook has near-complete publication type and date coverage but no journal or publisher values. Its creator field is not a reliable complete author list. A conservative Crossref/OpenAlex pilot found insufficient match coverage for journal, publisher, or co-authorship lenses; see `analysis/metadata-enrichment-audit.md`.
