# PhilPapers clustering robustness study

Protocol frozen: 27 August 2026, before any new seed, perturbation, or validation runs.

## Purpose

This study asks whether the cluster structure presented in the PhilPapers Atlas is reproducible, stable under defensible analytical choices, and semantically useful. It is not a search for the parameter combination with the largest single internal score. There is no labelled ground truth for this corpus, and internal validity indices can reward partitions that are not substantively meaningful. Conclusions will therefore rest on converging evidence from density validity, perturbation stability, original-space structure, text coherence, and cluster-level inspection.

The main object of evaluation is the existing 42-cluster lens:

> L2-normalised SPECTER (768D) → PCA (100D) → UMAP (30D; cosine; 15 neighbours; minimum distance 0; seed 42) → HDBSCAN (minimum cluster size 200; minimum samples 15; EOM selection; Euclidean distance).

The fixed 2D and 3D displays are visual representations and are not assumed to preserve the density geometry of the clustering space.

## Questions

1. Can the current 42-cluster result be reproduced exactly from the recovered arrays and declared HDBSCAN settings?
2. How sensitive are the result, its aggregate scores, and its individual clusters to UMAP random seed?
3. How sensitive are they to UMAP output dimension, PCA dimension, and nearby HDBSCAN parameters?
4. Does the apparently strong HDBSCAN result in two dimensions recur across independently fitted 2D projections, or is it specific to one display projection?
5. Which clusters survive resampling, and which split, merge, dissolve, or change their noise boundary?
6. Do stable clusters correspond to coherent neighbourhoods in the unreduced SPECTER space and coherent language in titles and abstracts?
7. Which conclusions are strong enough to guide the atlas, and which should remain exploratory lenses?

## Provenance audit and protocol amendments

The audit found two distinct UMAP lineages with the same visible hyperparameters:

- The May display lineage used the byte-identical PCA100 array but left UMAP unseeded. Its 30D output was then reduced to the fixed 2D display.
- The June/July clustering lineage regenerated UMAP20/25/30 with `random_state=42` and `low_memory=False`. The published 42-cluster labels were fitted on this later 30D array.

The two 30D arrays are numerically different. Fitting HDBSCAN(200, 15) to the later array reproduces the published 42 labels exactly; fitting it to the earlier array does not. The study therefore treats the current clustering and the fixed display as separate, explicitly named branches. It will not describe the display as a direct 2D projection of the exact 30D array used for the 42-cluster fit.

This discovery improved the original plan in four ways:

- UMAP seed sensitivity is now a primary analysis.
- The old display projection is tested as a historical special case, not silently mixed with newly fitted projections.
- The analysis environment is pinned to the server versions that exactly reproduce the 42 labels.
- The workbook `subject` field was rejected as external validation: 69,399 matched records contain only the value “Philosophy”.

These changes were made before any new robustness results were generated.

Before semantic validation, a second design audit identified one further gap. The
historical 104-cluster display had been selected with scikit-learn HDBSCAN at
minimum cluster size 150 and minimum samples 50, whereas the preregistered direct-2D
seed experiment initially carried forward the current lens's settings (200, 15).
The historical settings are therefore replicated on every independent 2D seed and
on the same twenty full-path subsamples. This is a protocol clarification, not a
post-semantic choice: the held-out semantic results had not been computed or
inspected. Both direct-2D configurations are retained, and the fixed historical
projection remains explicitly conditional.

## Frozen inputs

The study will record byte hashes for every input in `results/source-manifest.json`. The critical recovered artefacts are:

- 73,440 original SPECTER first-pool embeddings, 768 dimensions;
- 69,400 English-language row indices;
- the exact PCA100 array used by both recovered branches;
- seeded canonical PCA100→UMAP20/25/30 arrays;
- seeded canonical PCA50→UMAP20/25/30 arrays;
- the unseeded historical UMAP30 and its UMAP2 display;
- the published 42-cluster labels and unified-fit summary;
- document IDs, searchable atlas metadata, and the source metadata workbook.

The atlas order must match all 69,400 filtered document IDs. Any mismatch aborts the study.

## Fixed software environment

The primary environment matches the current server analysis environment where it affects the recovered result: Python 3.12.13 locally, NumPy 1.26.4, SciPy 1.17.1, scikit-learn 1.8.0, umap-learn 0.5.11, hdbscan 0.8.42, and pynndescent 0.6.0. The HDBSCAN reproduction check must return ARI 1.0, 42 clusters, 30.664265% noise, and relative validity 0.4215738883 before new runs begin.

## Experimental design

### A. Baseline and lineage checks

1. Refit the current HDBSCAN configuration on canonical UMAP30 and compare labels exactly.
2. Fit the same configuration on the historical UMAP30 and fixed UMAP2.
3. Quantify neighbourhood overlap between the two UMAP30 arrays at k = 15 and 50.
4. Measure trustworthiness of each representation relative to PCA100 on a fixed evaluation sample.
5. Verify all input hashes, shapes, finite values, and row alignment.

### B. UMAP dimension and seed sensitivity

Fit UMAP to the exact PCA100 array using:

- output dimensions: 2, 3, 5, 10, 20, 25, 30;
- seeds: 11, 23, 42, 73, 101, 211, 307, 419, 547, 809;
- cosine input metric, 15 neighbours, minimum distance 0, `low_memory=False`;
- otherwise the pinned umap-learn defaults.

Fit HDBSCAN(200, 15, EOM) to every result. For each dimension report the distribution of cluster count, noise fraction, DBCV/relative validity, persistence, membership strength, pairwise ARI, adjusted mutual information, and cluster-wise best-match Jaccard. The medoid run is the seed with the largest mean ARI to the other seeds; it is a representative run, not a consensus proof.

The saved canonical seed-42 arrays at dimensions 20, 25, and 30 must be regenerated exactly or, if platform-level numerical differences prevent byte equality, reproduce their clustering with ARI at least 0.99. Failure triggers a documented platform-sensitivity branch rather than silent substitution.

### C. HDBSCAN parameter robustness

On the seed-42 representation at every tested UMAP dimension, scan the Cartesian grid:

- minimum cluster size: 75, 100, 150, 200, 300, 500;
- minimum samples: 5, 15, 30, 50;
- EOM cluster selection.

For each cell calculate the same aggregate metrics and ARI to adjacent grid cells. A separate leaf-selection sensitivity fit is run at (200, 15) for every dimension. Results will be shown as surfaces and Pareto sets. They will not be collapsed into an undocumented or arbitrarily weighted “balanced score”. Degenerate partitions with fewer than five clusters or more than 250 clusters are retained in the results but excluded from substantive candidate selection.

### D. PCA and UMAP-neighbourhood sensitivity

The recovered PCA50 and PCA100 branches provide a controlled PCA-dimension comparison at UMAP dimensions 20, 25, and 30. For the strongest non-degenerate configurations from sections B–C, test UMAP neighbourhood sizes 10, 15, 30, and 50, with minimum distances 0 and 0.1, using seeds 42, 211, and 547. These tests determine whether a conclusion is tied to unusually aggressive local compression.

Candidate configurations for this stage are frozen from structural results before held-out semantic scores are examined. At least the current 30D configuration, direct 2D, the historical fixed 2D display, and the best intermediate-dimensional Pareto candidate must be retained even if one is dominated.

### E. Resampling and cluster-wise stability

For each frozen finalist, run 20 deterministic 80% subsamples. The full reduction-and-clustering path is refitted on each subsample. Primary fits scale minimum cluster size by the sample fraction; a secondary analysis keeps it fixed to distinguish population perturbation from a threshold artefact.

Compare each full-data cluster with its best matching subsample cluster using Jaccard similarity on shared papers, following Hennig’s cluster-wise approach. Report the full distribution, size-weighted and unweighted means, and the proportion of clusters in descriptive bands (<0.60, 0.60–0.75, 0.75–0.85, ≥0.85). No band is treated as an automatic declaration of truth. Also report global ARI, noise retention, split/merge counts, and paper-level assignment consistency after optimal label matching.

### F. Semantic validation, held out until finalists are frozen

Generate deterministic development and validation samples from paper IDs. Do not inspect validation results during structural selection.

For each finalist calculate:

- label agreement among approximate cosine nearest neighbours in the original 768D SPECTER space at k = 15 and 50;
- exact-neighbour recall for the approximate index on a fixed audit subset;
- cosine cohesion to original-space cluster centroids and nearest-centroid separation;
- cosine silhouette in original SPECTER space on the held-out sample, reported cautiously;
- class-based TF-IDF top terms from titles and abstracts;
- document-level NPMI coherence and term exclusivity;
- null distributions from 100 label permutations that preserve cluster sizes and noise prevalence.

The metadata supplies no meaningful external category labels. This limitation will be stated plainly. Semantic metrics reuse the same underlying documents and therefore demonstrate coherence, not discovery of a uniquely correct taxonomy.

### G. Cluster audit and consensus summary

For every cluster in each finalist, produce a row with size, persistence, membership, seed Jaccard, subsample Jaccard, original-space cohesion, text coherence, and its best overlap with the current 42-cluster lens. Flag clusters whose aggregate lens score looks strong but whose individual stability or semantic evidence is weak. Provide representative papers nearest the SPECTER centroid plus deterministic random examples for human inspection.

Construct a co-assignment summary on the held-out sample across seed and subsample runs. Use it to distinguish stable cores from uncertain boundaries. Do not force every paper into a consensus cluster; persistent noise and uncertain assignments are legitimate outcomes.

## Evaluation and decision rules

Evidence will be interpreted in layers:

1. **Reproducibility:** exact or near-exact recovery of the declared reference result.
2. **Global robustness:** distributions across seeds, parameters, and subsamples rather than a best run.
3. **Cluster robustness:** individual-cluster Jaccard and split/merge behaviour.
4. **Semantic utility:** original-space and text evidence substantially above size-preserving nulls.
5. **Practical interpretability:** a defensible resolution and a noise boundary that serve the atlas.

A lens can be recommended only if it is non-degenerate and lies on or close to the empirical Pareto frontier for stability, coverage, density validity, and semantic evidence. A large DBCV value alone is insufficient. DBCV values calculated in different representation spaces will not be treated as directly commensurable measurements of truth. If criteria disagree, the report will preserve the trade-off instead of manufacturing a single winner.

The 2D result will count as genuinely surprising only if its apparent advantage recurs across independently fitted 2D seeds and subsamples, survives comparison in original SPECTER/text space, and is not produced by a tiny number of coarse clusters. Otherwise it will be interpreted as projection-dependent density enhancement, which is compatible with the known behaviour of UMAP followed by density clustering.

## Quality controls

- Unit tests for label matching, Jaccard, ARI summaries, null permutations, and checkpoint resumption.
- Synthetic tests with known stable, merged, split, and noise clusters.
- Checkpointed arrays and atomic result writes; completed runs are never silently overwritten.
- Assertions on shape, dtype, finite values, row order, parameter identity, and label counts.
- Independent recomputation of headline table values from saved raw run files.
- A second-pass audit that samples result rows and refits them from scratch.
- Figures generated only from saved tabular results.
- A machine-readable run manifest with versions, hashes, parameters, timestamps, durations, and failures.

## Outputs

The tracked study output will include:

- `REPORT.md`: methods, results, limitations, interpretation, and atlas recommendations;
- `results/`: compact CSV/JSON tables and figures;
- `scripts/robustness_study.py`: checkpointed runner and analysis commands;
- `tests/test_robustness_study.py`: metric and integrity tests;
- `config.yaml`: the frozen parameter sets and seeds.

Large recovered arrays and fitted embeddings remain under the ignored `work/` directory. The report will contain enough hashes and commands to reproduce them from the authorised source project.

## Methodological basis

- UMAP documentation on [clustering](https://umap-learn.readthedocs.io/en/latest/clustering.html) and [reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html).
- McInnes, Healy and Astels, [*hdbscan: Hierarchical density based clustering*](https://joss.theoj.org/papers/10.21105/joss.00205) (2017).
- Moulavi et al., [*Density-Based Clustering Validation*](https://www.imada.sdu.dk/~zimek/publications/SDM2014/DBCV.pdf) (2014).
- Hennig, [*Cluster-wise assessment of cluster stability*](https://www.homepages.ucl.ac.uk/~ucakche/papers/clusta.pdf) (2007).
- Monti et al., [*Consensus Clustering*](https://www.cs.utexas.edu/~ml/biodm/papers/MLJ-biodm2.pdf) (2003).
- Herrmann et al., [*Enhancing cluster analysis via topological manifold learning*](https://link.springer.com/article/10.1007/s10618-023-00980-2) (2023).
- Gagolewski, Bartoszuk and Cena, [*Are cluster validity measures (in)valid?*](https://www.sciencedirect.com/science/article/pii/S0020025521010082) (2021).
